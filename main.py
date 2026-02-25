"""
Project Alpha — FastAPI Backend
Intelligent student-to-opportunity matching engine for the Northeast region.

Endpoints:
    POST /upload_resume   – Upload a PDF resume, extract skills, generate embeddings, query Qdrant
    GET  /opportunities   – List all regional internship/event opportunities
    GET  /health          – Health check for API & Qdrant connection
"""

import os
import io
import logging
import re
from typing import Dict, List, Optional, Set

import fitz  # PyMuPDF
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "opportunities")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = 384  # Dimension of all-MiniLM-L6-v2 vectors
TOP_K = int(os.getenv("TOP_K", 5))

# Hybrid scoring weights
W_SEMANTIC = 0.6   # weight for cosine similarity from Qdrant
W_SKILL    = 0.4   # weight for discrete skill overlap ratio

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("project_alpha")

# ---------------------------------------------------------------------------
# App Initialisation
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Project Alpha API",
    description="Connects Northeast-region students to internships, startups, and skill-based opportunities.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# ML Model & Qdrant Client  (lazy-loaded on startup)
# ---------------------------------------------------------------------------
model: Optional[SentenceTransformer] = None
qdrant: Optional[QdrantClient] = None


@app.on_event("startup")
async def startup_event():
    """Load the sentence-transformer model and connect to Qdrant."""
    global model, qdrant

    logger.info("Loading embedding model: %s …", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info("Embedding model loaded successfully.")

    logger.info("Connecting to Qdrant at %s:%s …", QDRANT_HOST, QDRANT_PORT)
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Ensure the collection exists
    collections = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in collections:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection '%s'.", COLLECTION_NAME)
    else:
        logger.info("Qdrant collection '%s' already exists.", COLLECTION_NAME)


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class SkillGap(BaseModel):
    """A single skill gap entry."""
    skill: str
    importance: str = "required"  # critical | required | preferred


class MatchResult(BaseModel):
    """A single opportunity match."""
    id: str
    title: str
    organisation: str
    match_score: float
    semantic_score: float
    skill_overlap_ratio: float
    matched_skills: List[str]
    missing_skills: List[str]


class ResumeResponse(BaseModel):
    """Response payload for POST /upload_resume."""
    extracted_skills: List[str]
    matches: List[MatchResult]
    skill_gaps: List[SkillGap]


class Opportunity(BaseModel):
    """Schema for a single opportunity listing."""
    id: str
    title: str
    organisation: str
    type: str  # internship | startup | hackathon | event
    location: str
    description: str
    required_skills: List[str]
    timeline: str
    eligibility: str


class HealthResponse(BaseModel):
    """Response schema for the health endpoint."""
    model_config = {"protected_namespaces": ()}  # allow model_ fields

    status: str
    qdrant_connected: bool
    model_loaded: bool
    collection_count: int


# ---------------------------------------------------------------------------
# Helpers — Skill Synonyms & Normalization
# ---------------------------------------------------------------------------

# Maps abbreviations / alternate spellings → canonical skill name.
# When we see ANY key in text, we record the canonical value.
SKILL_SYNONYMS: Dict[str, str] = {
    # Languages
    "js": "javascript", "javascript": "javascript",
    "ts": "typescript", "typescript": "typescript",
    "py": "python", "python3": "python", "python": "python",
    "c++": "c++", "cpp": "c++",
    "c#": "c#", "csharp": "c#",
    "c": "c",
    "java": "java",
    "golang": "go", "go": "go",
    "rust": "rust",
    "r": "r",
    "swift": "swift",
    "kotlin": "kotlin",
    "ruby": "ruby",
    "php": "php",
    "bash": "bash", "shell": "bash",
    # ML / AI
    "ml": "machine learning", "machine learning": "machine learning",
    "dl": "deep learning", "deep learning": "deep learning",
    "ai": "artificial intelligence", "artificial intelligence": "artificial intelligence",
    "nlp": "nlp", "natural language processing": "nlp",
    "cv": "computer vision", "computer vision": "computer vision",
    "llm": "llm", "large language model": "llm",
    "gen ai": "generative ai", "generative ai": "generative ai", "genai": "generative ai",
    "neural network": "deep learning", "neural networks": "deep learning",
    # Frameworks
    "pytorch": "pytorch", "torch": "pytorch",
    "tensorflow": "tensorflow", "tf": "tensorflow",
    "scikit-learn": "scikit-learn", "sklearn": "scikit-learn",
    "keras": "keras",
    "react": "react", "reactjs": "react", "react.js": "react",
    "next.js": "next.js", "nextjs": "next.js",
    "node.js": "node.js", "nodejs": "node.js", "node": "node.js",
    "express": "express", "expressjs": "express",
    "fastapi": "fastapi",
    "flask": "flask",
    "django": "django",
    "spring": "spring", "spring boot": "spring",
    "flutter": "flutter",
    "angular": "angular",
    "vue": "vue", "vuejs": "vue", "vue.js": "vue",
    # Data
    "sql": "sql", "mysql": "sql", "postgres": "postgresql", "postgresql": "postgresql",
    "mongodb": "mongodb", "mongo": "mongodb",
    "pandas": "pandas", "numpy": "numpy",
    "data analysis": "data analysis", "data analytics": "data analysis",
    "statistics": "statistics", "stats": "statistics",
    "tableau": "tableau", "power bi": "power bi", "powerbi": "power bi",
    # Cloud & DevOps
    "docker": "docker",
    "kubernetes": "kubernetes", "k8s": "kubernetes",
    "aws": "aws", "amazon web services": "aws",
    "gcp": "gcp", "google cloud": "gcp",
    "azure": "azure",
    "terraform": "terraform",
    "ci/cd": "ci/cd", "cicd": "ci/cd",
    "devops": "devops",
    "linux": "linux", "ubuntu": "linux",
    # Web
    "html": "html", "html5": "html",
    "css": "css", "css3": "css",
    "tailwind": "tailwind", "tailwindcss": "tailwind",
    # Tools
    "git": "git", "github": "git",
    "figma": "figma",
    "jira": "jira",
    "rest api": "rest api", "restful": "rest api", "api": "rest api",
    # Mobile
    "android": "android",
    "ios": "ios",
    # Security
    "networking": "networking",
    "firewalls": "firewalls",
    "siem": "siem",
    "encryption": "encryption",
    "penetration testing": "penetration testing", "pen testing": "penetration testing",
    # Other
    "blockchain": "blockchain", "solidity": "solidity", "web3": "web3",
    "agile": "agile", "scrum": "agile",
    "product strategy": "product strategy",
}

# Canonical skill keywords used for extraction (unique values from SKILL_SYNONYMS)
SKILL_KEYWORDS: List[str] = sorted(set(SKILL_SYNONYMS.keys()))


def normalize_skill(raw: str) -> str:
    """Map a raw skill mention to its canonical form."""
    return SKILL_SYNONYMS.get(raw.lower().strip(), raw.lower().strip())


def normalize_skill_set(skills: List[str]) -> List[str]:
    """Normalize and deduplicate a list of skills."""
    return sorted({normalize_skill(s) for s in skills})


# ---------------------------------------------------------------------------
# Helpers — Core Functions
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract raw text from a PDF using PyMuPDF (fitz)."""
    text_parts: list[str] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts)


def extract_skills(text: str) -> List[str]:
    """
    Extract skills from text using keyword matching + synonym normalization.
    Multi-word skills are matched first, then single-word with word-boundary checks.
    """
    text_lower = text.lower()
    found: Set[str] = set()

    # Sort keywords by length descending so multi-word matches take priority
    sorted_keywords = sorted(SKILL_KEYWORDS, key=len, reverse=True)

    for keyword in sorted_keywords:
        # For single-char / very short keywords (c, r, go), use word boundaries
        if len(keyword) <= 2:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text_lower):
                found.add(normalize_skill(keyword))
        else:
            if keyword in text_lower:
                found.add(normalize_skill(keyword))

    return sorted(found)


def generate_embedding(text: str) -> List[float]:
    """Generate a dense vector for the given text using the loaded model."""
    if model is None:
        raise RuntimeError("Embedding model not loaded.")
    embedding = model.encode(text, convert_to_tensor=False)
    if isinstance(embedding, torch.Tensor):
        return embedding.tolist()
    return embedding.tolist()


def compute_skill_overlap(student_skills: List[str], opp_skills: List[str]) -> tuple:
    """
    Compare student skills against opportunity requirements.
    Returns (matched_skills, skill_overlap_ratio).
    """
    student_set = {s.lower() for s in student_skills}
    opp_set = {s.lower() for s in opp_skills}

    if not opp_set:
        return [], 1.0  # no requirements = perfect match

    matched = sorted(student_set & opp_set)
    ratio = len(matched) / len(opp_set)
    return matched, round(ratio, 4)


def compute_hybrid_score(semantic_score: float, skill_ratio: float) -> float:
    """
    Blend semantic similarity with discrete skill overlap.
    final_score = W_SEMANTIC × cosine_sim + W_SKILL × skill_ratio
    """
    return round(W_SEMANTIC * semantic_score + W_SKILL * skill_ratio, 4)


def compute_skill_gaps(
    student_skills: List[str],
    opportunity_skills: List[str],
    opportunity_title: str = "",
) -> List[SkillGap]:
    """
    Return skills required by the opportunity that the student lacks.
    Skills mentioned in the job title are marked 'critical',
    others default to 'preferred'.
    """
    student_set = {s.lower() for s in student_skills}
    title_lower = opportunity_title.lower()

    gaps: List[SkillGap] = []
    for skill in opportunity_skills:
        if skill.lower() not in student_set:
            # If the skill (or a synonym) appears in the job title → critical
            if skill.lower() in title_lower:
                importance = "critical"
            else:
                # Check if any synonym of this skill appears in title
                is_in_title = any(
                    syn in title_lower
                    for syn, canonical in SKILL_SYNONYMS.items()
                    if canonical == skill.lower()
                )
                importance = "critical" if is_in_title else "preferred"
            gaps.append(SkillGap(skill=skill, importance=importance))

    # Sort: critical first, then preferred
    gaps.sort(key=lambda g: (0 if g.importance == "critical" else 1, g.skill))
    return gaps


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/upload_resume", response_model=ResumeResponse, tags=["Resume"])
async def upload_resume(file: UploadFile = File(...)):
    """
    Accept a PDF resume, extract text & skills, embed the resume,
    and query Qdrant for the best-matching opportunities.
    Returns top matches with skill-gap analysis.
    """
    if file.content_type not in ("application/pdf",):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted.",
        )

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # 1. Extract text from PDF
    try:
        resume_text = extract_text_from_pdf(pdf_bytes)
    except Exception as exc:
        logger.error("PDF extraction failed: %s", exc)
        raise HTTPException(status_code=422, detail="Could not parse PDF.") from exc

    if not resume_text.strip():
        raise HTTPException(status_code=422, detail="No text found in the PDF.")

    # 2. Extract skills
    extracted_skills = extract_skills(resume_text)

    # 3. Generate embedding for the resume
    resume_vector = generate_embedding(resume_text)

    # 4. Query Qdrant for top-K matches
    try:
        search_results = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=resume_vector,
            limit=TOP_K,
        )
    except Exception as exc:
        logger.error("Qdrant search failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Vector search unavailable.",
        ) from exc

    # 5. Build response with hybrid scoring + skill-gap analysis
    matches: List[MatchResult] = []
    all_gaps: List[SkillGap] = []

    for hit in search_results:
        payload = hit.payload or {}
        opp_title = payload.get("title", "Unknown")
        opp_org = payload.get("organisation", "Unknown")
        opp_skills = normalize_skill_set(payload.get("required_skills", []))

        # Normalize student skills for comparison
        student_normalized = normalize_skill_set(extracted_skills)

        # Compute discrete skill overlap
        matched, skill_ratio = compute_skill_overlap(student_normalized, opp_skills)

        # Compute hybrid score (semantic + skill overlap)
        semantic = round(hit.score, 4)
        hybrid = compute_hybrid_score(semantic, skill_ratio)

        # Compute weighted skill gaps
        gaps = compute_skill_gaps(student_normalized, opp_skills, opp_title)

        matches.append(
            MatchResult(
                id=str(hit.id),
                title=opp_title,
                organisation=opp_org,
                match_score=hybrid,
                semantic_score=semantic,
                skill_overlap_ratio=skill_ratio,
                matched_skills=matched,
                missing_skills=[g.skill for g in gaps],
            )
        )
        all_gaps.extend(gaps)

    # Re-sort matches by hybrid score (descending)
    matches.sort(key=lambda m: m.match_score, reverse=True)

    # Deduplicate skill gaps across all matches, keeping highest importance
    gap_map: dict[str, SkillGap] = {}
    for gap in all_gaps:
        key = gap.skill.lower()
        if key not in gap_map:
            gap_map[key] = gap
        elif gap.importance == "critical" and gap_map[key].importance != "critical":
            gap_map[key] = gap  # upgrade importance

    unique_gaps = sorted(gap_map.values(), key=lambda g: (0 if g.importance == "critical" else 1, g.skill))

    return ResumeResponse(
        extracted_skills=extracted_skills,
        matches=matches,
        skill_gaps=unique_gaps,
    )


@app.get("/opportunities", response_model=List[Opportunity], tags=["Opportunities"])
async def get_opportunities():
    """
    Return all regional internship, startup, event, and hackathon
    listings stored in the Qdrant collection.
    """
    try:
        result = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            with_payload=True,
            with_vectors=False,
        )
        points, _ = result
    except Exception as exc:
        logger.error("Failed to fetch opportunities: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Could not retrieve opportunities.",
        ) from exc

    opportunities: List[Opportunity] = []
    for point in points:
        p = point.payload or {}
        opportunities.append(
            Opportunity(
                id=str(point.id),
                title=p.get("title", ""),
                organisation=p.get("organisation", ""),
                type=p.get("type", "internship"),
                location=p.get("location", "Northeast, India"),
                description=p.get("description", ""),
                required_skills=p.get("required_skills", []),
                timeline=p.get("timeline", ""),
                eligibility=p.get("eligibility", ""),
            )
        )

    return opportunities


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Basic health check for the API and Qdrant connection."""
    qdrant_ok = False
    collection_count = 0

    try:
        collections = qdrant.get_collections().collections
        qdrant_ok = True
        collection_count = len(collections)
    except Exception:
        pass

    return HealthResponse(
        status="ok" if qdrant_ok and model is not None else "degraded",
        qdrant_connected=qdrant_ok,
        model_loaded=model is not None,
        collection_count=collection_count,
    )


# ---------------------------------------------------------------------------
# Entrypoint (for local development)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
