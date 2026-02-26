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
import uuid
from typing import Dict, List, Optional, Set

import google.generativeai as genai

import fitz  # PyMuPDF
import spacy
from spacy.matcher import PhraseMatcher
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
from src.searching import router as search_router, store_user_context

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

# Load environment variables from .env if present
from dotenv import load_dotenv
load_dotenv()

# LLM Configuration for holistic advice (Gemini)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
LLM_MODEL_NAME = os.getenv("LLM_MODEL", "gemini-flash-latest")

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

app.include_router(search_router)

# ---------------------------------------------------------------------------
# ML Model, NLP & Qdrant Client  (lazy-loaded on startup)
# ---------------------------------------------------------------------------
model: Optional[SentenceTransformer] = None
nlp = None
skill_matcher = None
qdrant: Optional[QdrantClient] = None


@app.on_event("startup")
async def startup_event():
    """Load the sentence-transformer model, spaCy NLP, and connect to Qdrant."""
    global model, qdrant, nlp, skill_matcher

    logger.info("Loading embedding model: %s …", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info("Embedding model loaded successfully.")

    logger.info("Loading spaCy NLP model for NER-based skill extraction…")
    nlp = spacy.load("en_core_web_sm")
    skill_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    # Add all synonym patterns to the matcher
    patterns = [nlp.make_doc(term) for term in SKILL_SYNONYMS.keys()]
    skill_matcher.add("SKILL", patterns)
    logger.info("spaCy NER skill matcher loaded with %d patterns.", len(patterns))

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
    advice: str = ""  # Human-readable learning advice


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
    linkedin_url: str = ""
    advice: str = ""  # Natural language match explanation


class ResumeResponse(BaseModel):
    """Response payload for POST /upload_resume."""
    session_id: str = ""  # Unique session ID for cross-reference in hybrid search
    extracted_skills: List[str]
    matches: List[MatchResult]
    skill_gaps: List[SkillGap]
    overall_advice: str = ""  # Summary recommendations in natural language


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
    linkedin_url: str = ""


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
    NER-based skill extraction using spaCy PhraseMatcher.
    Uses linguistic tokenization for proper word-boundary matching,
    handles multi-word expressions, and normalizes via synonym map.
    Falls back to regex matching if spaCy is not loaded.
    """
    if nlp is None or skill_matcher is None:
        # Fallback to regex if spaCy not loaded
        return _extract_skills_regex(text)

    doc = nlp(text)
    found: Set[str] = set()

    # PhraseMatcher runs NER-style matching with linguistic features
    matches = skill_matcher(doc)
    for match_id, start, end in matches:
        span_text = doc[start:end].text.lower()
        canonical = normalize_skill(span_text)
        found.add(canonical)

    # Also try to extract skills from spaCy's built-in NER entities
    # (ORG, PRODUCT entities can sometimes be frameworks/tools)
    for ent in doc.ents:
        ent_lower = ent.text.lower().strip()
        if ent_lower in SKILL_SYNONYMS:
            found.add(normalize_skill(ent_lower))

    return sorted(found)


def _extract_skills_regex(text: str) -> List[str]:
    """Fallback regex-based skill extraction."""
    text_lower = text.lower()
    found: Set[str] = set()
    sorted_keywords = sorted(SKILL_KEYWORDS, key=len, reverse=True)
    for keyword in sorted_keywords:
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
    Includes human-readable learning advice per gap.
    """
    student_set = {s.lower() for s in student_skills}
    title_lower = opportunity_title.lower()

    gaps: List[SkillGap] = []
    for skill in opportunity_skills:
        if skill.lower() not in student_set:
            # Determine importance
            if skill.lower() in title_lower:
                importance = "critical"
            else:
                is_in_title = any(
                    syn in title_lower
                    for syn, canonical in SKILL_SYNONYMS.items()
                    if canonical == skill.lower()
                )
                importance = "critical" if is_in_title else "preferred"

            advice = _generate_gap_advice(skill, importance)
            gaps.append(SkillGap(skill=skill, importance=importance, advice=advice))

    gaps.sort(key=lambda g: (0 if g.importance == "critical" else 1, g.skill))
    return gaps


# ---------------------------------------------------------------------------
# Natural Language Advice Generation
# ---------------------------------------------------------------------------

# Learning resource suggestions by skill category
SKILL_LEARNING_HINTS: Dict[str, str] = {
    "python": "Practice on LeetCode and build projects with Flask or FastAPI",
    "javascript": "Try freeCodeCamp's JavaScript course and build a portfolio project",
    "typescript": "Start with TypeScript's official handbook and convert a JS project",
    "react": "Follow the official React tutorial and build a CRUD app",
    "next.js": "Try the Next.js 'Learn' course at nextjs.org/learn",
    "node.js": "Build a REST API with Express.js to get hands-on experience",
    "machine learning": "Start with Andrew Ng's ML course on Coursera and implement models in scikit-learn",
    "deep learning": "Take fast.ai's Practical Deep Learning course — it's free and project-based",
    "pytorch": "Follow PyTorch's official 60-minute blitz tutorial and build a classifier",
    "tensorflow": "Try TensorFlow's beginner tutorials on tensorflow.org",
    "nlp": "Explore Hugging Face's NLP course and fine-tune a text classifier",
    "computer vision": "Start with OpenCV basics and move to CNN architectures",
    "docker": "Containerize one of your existing projects — Docker's getting started guide is excellent",
    "kubernetes": "Start with Minikube locally and deploy a simple app",
    "aws": "Get hands-on with AWS Free Tier — start with EC2 and S3",
    "gcp": "Try Google Cloud's Qwiklabs for guided hands-on labs",
    "sql": "Practice on SQLZoo or LeetCode SQL problems",
    "git": "Use Git daily in your projects and learn branching strategies",
    "linux": "Set up a Linux VM or use WSL and practice command-line navigation",
    "ci/cd": "Set up GitHub Actions for one of your repos — it's free and well-documented",
    "data analysis": "Work through Kaggle's Pandas course and analyze a real dataset",
    "scikit-learn": "Build a classification pipeline on a Kaggle dataset",
    "pandas": "Work through Kaggle's Pandas micro-course",
    "numpy": "Practice with NumPy's quickstart tutorial and implement math from scratch",
    "figma": "Redesign an existing app's UI in Figma to build your design skills",
    "tailwind": "Rebuild a UI component library using Tailwind CSS utility classes",
    "rest api": "Build and document a REST API with Swagger/OpenAPI",
    "agile": "Study the Scrum Guide and participate in sprint-based projects",
    "blockchain": "Follow Ethereum's developer documentation and deploy a smart contract",
    "networking": "Study CompTIA Network+ material and set up a home lab",
    "penetration testing": "Try TryHackMe or HackTheBox platforms for hands-on cybersecurity practice",
    "encryption": "Study cryptography basics and implement common algorithms",
}


def _generate_gap_advice(skill: str, importance: str) -> str:
    """Generate human-readable advice for a single skill gap."""
    hint = SKILL_LEARNING_HINTS.get(skill.lower(), f"Look for online courses, tutorials, or projects involving {skill}")
    if importance == "critical":
        return f"This is a key skill explicitly mentioned in the job title. {hint}."
    return f"Having {skill} knowledge would strengthen your candidacy. {hint}."


def generate_match_advice(
    title: str,
    org: str,
    matched: List[str],
    missing: List[str],
    hybrid_score: float,
    skill_ratio: float,
) -> str:
    """Generate a natural-language explanation for a single match."""
    total = len(matched) + len(missing)
    pct = round(skill_ratio * 100)

    # Opening assessment
    if hybrid_score >= 0.65:
        opening = f"You're a strong match for the {title} role at {org}!"
    elif hybrid_score >= 0.45:
        opening = f"You're a reasonable match for the {title} role at {org}."
    else:
        opening = f"This role at {org} is a stretch, but could be a growth opportunity."

    # Skill summary
    parts = [opening]
    if matched:
        top_skills = matched[:5]
        parts.append(f"Your strengths in {', '.join(top_skills)} align well with what they're looking for ({pct}% skill overlap).")

    # Gap advice
    if not missing:
        parts.append("You meet all the listed skill requirements — focus on showcasing your projects and experience.")
    elif len(missing) <= 2:
        parts.append(f"To strengthen your application, consider picking up {' and '.join(missing)}.")
    else:
        top_gaps = missing[:3]
        parts.append(f"The main gaps are {', '.join(top_gaps)}. Prioritize learning these to become a competitive candidate.")

    return " ".join(parts)


def generate_overall_advice(
    extracted_skills: List[str],
    matches: List["MatchResult"],
    gaps: List[SkillGap],
) -> str:
    """Generate a summary of recommendations across all matches."""
    parts = []

    # Profile strength
    parts.append(f"Based on your resume, we identified {len(extracted_skills)} relevant skills.")

    # Best match context
    if matches:
        best = matches[0]
        pct = round(best.match_score * 100)
        parts.append(f"Your strongest match is the \"{best.title}\" role at {best.organisation} ({pct}% match).")

    # Gap summary
    critical_gaps = [g for g in gaps if g.importance == "critical"]
    preferred_gaps = [g for g in gaps if g.importance != "critical"]

    if not gaps:
        parts.append("Impressively, you have no skill gaps across your top matches — you're well-prepared!")
    else:
        if critical_gaps:
            crit_names = [g.skill for g in critical_gaps[:3]]
            parts.append(
                f"Your most important skill gaps are: {', '.join(crit_names)}. "
                f"These are explicitly mentioned in job titles, so learning them should be your top priority."
            )
        if preferred_gaps:
            pref_names = [g.skill for g in preferred_gaps[:3]]
            parts.append(
                f"Nice-to-have skills you could add: {', '.join(pref_names)}. "
                f"These would broaden your opportunities but aren't dealbreakers."
            )

    # Closing
    parts.append("Keep building projects that showcase these skills — hands-on experience is what recruiters look for most.")

    return " ".join(parts)


def generate_overall_advice_with_llm(
    extracted_skills: List[str],
    matches: List["MatchResult"],
    gaps: List[SkillGap],
) -> str:
    """Use a Transformer/LLM to generate a personalized, holistic learning roadmap."""
    # We allow running if the config was completed
    if not os.getenv("GEMINI_API_KEY", "AIzaSyBf_n_FOG0YV3JbPdItkJzy7dUTKxQ6Sv8"):
        # Fallback to rules-based advice if no LLM configured
        return generate_overall_advice(extracted_skills, matches, gaps)

    # 1. Build the context for the AI
    skills_context = ", ".join(extracted_skills) if extracted_skills else "None explicitly listed."
    
    jobs_context = ""
    for idx, match in enumerate(matches[:3]):
        jobs_context += (
            f"\n{idx+1}. {match.title} at {match.organisation} (Score: {int(match.match_score * 100)}%)\n"
            f"   - Matched Skills: {', '.join(match.matched_skills)}\n"
            f"   - Missing Skills: {', '.join(match.missing_skills)}\n"
        )
        
    gaps_context = ", ".join([g.skill for g in gaps[:5]]) if gaps else "None"

    prompt = f"""
    You are an expert technical mentor advising a student. Your job is to analyze their resume data against their top recommended roles, and provide a holistic, empowering "what's next?" roadmap to help them grow.
    
    STUDENT'S CURRENT SKILLS:
    {skills_context}
    
    TOP 3 RECOMMENDED ROLES & SCORES:
    {jobs_context}
    
    KEY SKILL GAPS TO BRIDGE:
    {gaps_context}
    
    INSTRUCTIONS:
    Write a clear, structured roadmap directly addressing the student (using "you").
    
    - Part 1 (Validation): Acknowledge their current skills and celebrate their best match. Make them feel recognized for their hard work.
    - Part 2 (The Real Talk): Act as an honest mentor. Point out the specific gaps that are holding them back from being the perfect candidate for these roles. Briefly explain *why* these skills matter in the industry.
    - Part 3 (The Action Plan): Give them a highly specific, actionable study plan formatted as a bulleted checklist. Tell them the exact next 3 things to learn and exactly where to focus their energy this week. Use clear task-list formatting (e.g. `[ ] Task 1`, `[ ] Task 2`).

    Keep the tone deeply inspiring, empathetic, action-oriented, and professional, exactly like a senior engineer who really wants them to succeed.
    """

    try:
        model = genai.GenerativeModel(model_name=LLM_MODEL_NAME)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=2048,
            )
        )
        return response.text.strip()
    except Exception as e:
        logger.error("LLM Generation failed: %s", e)
        return generate_overall_advice(extracted_skills, matches, gaps)


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

    # 3b. Store resume in session memory for hybrid search cross-reference
    session_id = str(uuid.uuid4())
    store_user_context(session_id, extracted_skills, resume_vector, resume_text)

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

    # 5. Build response with hybrid scoring + skill-gap analysis + advice
    matches: List[MatchResult] = []
    all_gaps: List[SkillGap] = []

    for hit in search_results:
        payload = hit.payload or {}
        opp_title = payload.get("title", "Unknown")
        opp_org = payload.get("organisation", "Unknown")
        opp_skills = normalize_skill_set(payload.get("required_skills", []))
        opp_url = payload.get("linkedin_url", "")

        # Normalize student skills for comparison
        student_normalized = normalize_skill_set(extracted_skills)

        # Compute discrete skill overlap
        matched, skill_ratio = compute_skill_overlap(student_normalized, opp_skills)

        # Compute hybrid score (semantic + skill overlap)
        semantic = round(hit.score, 4)
        hybrid = compute_hybrid_score(semantic, skill_ratio)

        # Compute weighted skill gaps with per-gap advice
        gaps = compute_skill_gaps(student_normalized, opp_skills, opp_title)

        # Generate natural language match advice
        advice = generate_match_advice(
            opp_title, opp_org, matched,
            [g.skill for g in gaps], hybrid, skill_ratio,
        )

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
                linkedin_url=opp_url,
                advice=advice,
            )
        )
        all_gaps.extend(gaps)

    # Re-sort matches by hybrid score (descending)
    matches.sort(key=lambda m: m.match_score, reverse=True)

    # Deduplicate skill gaps across all matches, keeping highest importance + advice
    gap_map: dict[str, SkillGap] = {}
    for gap in all_gaps:
        key = gap.skill.lower()
        if key not in gap_map:
            gap_map[key] = gap
        elif gap.importance == "critical" and gap_map[key].importance != "critical":
            gap_map[key] = gap  # upgrade importance

    unique_gaps = sorted(gap_map.values(), key=lambda g: (0 if g.importance == "critical" else 1, g.skill))

    # Generate AI-powered summary advice (falls back to hardcoded if no API key is set)
    overall = generate_overall_advice_with_llm(extracted_skills, matches, unique_gaps)

    return ResumeResponse(
        session_id=session_id,
        extracted_skills=extracted_skills,
        matches=matches,
        skill_gaps=unique_gaps,
        overall_advice=overall,
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
                linkedin_url=p.get("linkedin_url", ""),
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
