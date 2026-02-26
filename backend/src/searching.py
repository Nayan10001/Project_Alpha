"""
searching.py — Advanced Hybrid Search Engine with:
  1. Resume Memory Layer (session-scoped user context)
  2. Reciprocal Rank Fusion (RRF) scoring
  3. Geo-Spatial Weighting (country/city entity extraction + boost)
  4. Quality Assurance / Debug endpoint
"""

from __future__ import annotations

import re
import logging
import uuid
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("project_alpha.search")
router = APIRouter()

# =====================================================================
# 1. RESUME MEMORY LAYER — Session-scoped user context buffer
# =====================================================================
# In-memory store keyed by session_id.  In production replace with Redis.
_user_sessions: Dict[str, "UserContext"] = {}


class UserContext:
    """High-priority session buffer storing a parsed resume profile."""

    def __init__(
        self, session_id: str, skills: List[str],
        embedding: List[float], raw_text: str,
    ):
        self.session_id = session_id
        self.skills = skills            # canonical skill list
        self.embedding = embedding      # 384-D dense vector
        self.raw_text = raw_text        # original resume text


def store_user_context(
    session_id: str, skills: List[str],
    embedding: List[float], raw_text: str,
) -> None:
    """Called by main.py after a resume upload to persist user context."""
    _user_sessions[session_id] = UserContext(
        session_id=session_id,
        skills=skills,
        embedding=embedding,
        raw_text=raw_text,
    )
    logger.info("Stored user context for session %s (%d skills).", session_id, len(skills))


def get_user_context(session_id: str) -> Optional[UserContext]:
    return _user_sessions.get(session_id)


# =====================================================================
# 2. GEO-SPATIAL — Location entity extraction & boost
# =====================================================================

# Known countries & major US cities/states for entity extraction
_COUNTRY_ENTITIES: Dict[str, str] = {
    "united states": "United States", "usa": "United States", "us": "United States",
    "india": "India", "canada": "Canada", "uk": "United Kingdom",
    "united kingdom": "United Kingdom", "germany": "Germany",
    "australia": "Australia", "singapore": "Singapore", "japan": "Japan",
    "france": "France", "remote": "Remote",
}

_CITY_STATE_ENTITIES: Dict[str, str] = {
    # US
    "seattle": "Seattle, WA", "san francisco": "San Francisco, CA",
    "new york": "New York, NY", "nyc": "New York, NY",
    "mountain view": "Mountain View, CA", "palo alto": "Palo Alto, CA",
    "san jose": "San Jose, CA", "austin": "Austin, TX",
    "los angeles": "Los Angeles, CA", "la": "Los Angeles, CA",
    "atlanta": "Atlanta, GA", "boston": "Boston, MA",
    "california": "CA", "texas": "TX",
    # India (Northeast focus per problem statement)
    "guwahati": "Guwahati, Assam", "shillong": "Shillong, Meghalaya",
    "imphal": "Imphal, Manipur", "agartala": "Agartala, Tripura",
    "aizawl": "Aizawl, Mizoram", "itanagar": "Itanagar, Arunachal Pradesh",
    "kohima": "Kohima, Nagaland", "gangtok": "Gangtok, Sikkim",
    "silchar": "Silchar, Assam", "dibrugarh": "Dibrugarh, Assam",
    "assam": "Assam", "meghalaya": "Meghalaya",
}

GEO_BOOST_FACTOR = 1.5   # multiplicative boost for geo-matched results
GEO_PENALTY_FACTOR = 0.8 # penalty when query specifies a location and result doesn't match


def extract_geo_entities(text: str) -> Tuple[List[str], List[str]]:
    """
    Extract geographic entities (countries and cities/states) from text.
    Returns (countries, cities_or_states).
    """
    text_lower = text.lower()
    countries: List[str] = []
    cities: List[str] = []

    # Sort by length descending to match longer phrases first
    for key, canonical in sorted(_COUNTRY_ENTITIES.items(), key=lambda x: len(x[0]), reverse=True):
        if key in text_lower:
            if canonical not in countries:
                countries.append(canonical)

    for key, canonical in sorted(_CITY_STATE_ENTITIES.items(), key=lambda x: len(x[0]), reverse=True):
        # Avoid matching 2-char abbreviations unless surrounded by word boundaries
        if len(key) <= 2:
            if re.search(r'\b' + re.escape(key) + r'\b', text_lower):
                if canonical not in cities:
                    cities.append(canonical)
        else:
            if key in text_lower:
                if canonical not in cities:
                    cities.append(canonical)

    return countries, cities


def compute_geo_score(query_countries: List[str], query_cities: List[str], opp_location: str) -> float:
    """
    Returns a geo multiplier:
      - 1.5x if location matches query country or city
      - 0.8x if query has location intent but opp doesn't match
      - 1.0x if query has no location intent
    """
    if not query_countries and not query_cities:
        return 1.0  # No geo intent — neutral

    loc_lower = opp_location.lower()

    # Check city match first (more specific)
    for city in query_cities:
        if city.lower() in loc_lower or loc_lower in city.lower():
            return GEO_BOOST_FACTOR

    # Check country match
    for country in query_countries:
        if country.lower() in loc_lower:
            return GEO_BOOST_FACTOR

    # "Remote" matches everything if user asked for remote
    if "Remote" in query_countries and "remote" in loc_lower:
        return GEO_BOOST_FACTOR

    return GEO_PENALTY_FACTOR  # Query had geo intent, but this result doesn't match


# =====================================================================
# 3. RECIPROCAL RANK FUSION (RRF) — Multi-signal scoring
# =====================================================================

RRF_K = 60  # Standard RRF constant


def rrf_score(ranks: List[int]) -> float:
    """
    Compute RRF score from a list of ranks across different retrieval signals.
    RRF(d) = Σ  1 / (k + rank_i)
    """
    return sum(1.0 / (RRF_K + r) for r in ranks)


def _tokenize(text: str) -> Set[str]:
    """Tokenize for keyword overlap — strip common stopwords."""
    stopwords = {"for", "the", "and", "with", "role", "jobs", "i", "am",
                 "looking", "a", "an", "to", "in", "at", "of", "on", "is", "are", "that"}
    return {t for t in re.findall(r"[a-z0-9+#./]{2,}", text.lower()) if t not in stopwords}


# =====================================================================
# PYDANTIC SCHEMAS
# =====================================================================

class HybridSearchRequest(BaseModel):
    query: str
    session_id: str = ""    # Optional: link to a stored resume for cross-ref
    top_k: int = 10

class CompanySearchResult(BaseModel):
    organisation: str
    match_score: float
    description: str
    matched_roles: List[str]
    location: str
    geo_match: bool = False          # True if geo-boosted
    resume_skill_overlap: List[str] = []  # skills from resume that match

class DebugTraceStep(BaseModel):
    step: str
    detail: str

class DebugResult(BaseModel):
    organisation: str
    semantic_rank: int
    keyword_rank: int
    resume_rank: int
    geo_multiplier: float
    rrf_score: float
    final_score: float

class QATestCase(BaseModel):
    name: str
    query: str
    session_id: str = ""
    results: List[CompanySearchResult]
    debug_trace: List[DebugTraceStep]
    debug_results: List[DebugResult]


# =====================================================================
# MAIN HYBRID SEARCH ENDPOINT
# =====================================================================

@router.post("/hybrid_search", response_model=List[CompanySearchResult], tags=["Search"])
async def hybrid_company_search(payload: HybridSearchRequest):
    """
    Advanced Hybrid Search with:
      - Semantic vector retrieval from Qdrant
      - Keyword overlap re-ranking
      - Resume cross-reference (if session_id provided)
      - Geo-spatial entity boosting
      - Reciprocal Rank Fusion (RRF) for final scoring
    """
    from main import qdrant, model, COLLECTION_NAME, extract_skills, normalize_skill_set

    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if not model or not qdrant:
        raise HTTPException(status_code=503, detail="ML Models or Vector DB not loaded.")

    # ── Step 1: Extract geo entities from query ──
    query_countries, query_cities = extract_geo_entities(query)
    has_geo_intent = bool(query_countries or query_cities)

    # ── Step 2: Extract skills from the query text ──
    query_skills = normalize_skill_set(extract_skills(query))

    # ── Step 3: Get user context if session is provided ──
    user_ctx = get_user_context(payload.session_id) if payload.session_id else None

    # ── Step 4: Generate primary query embedding ──
    query_vector = model.encode(query, convert_to_tensor=False)
    if hasattr(query_vector, "tolist"):
        query_vector = query_vector.tolist()

    # ── Step 5: If we have a resume, blend the embeddings (cross-reference) ──
    if user_ctx:
        # Weighted blend: 60% query intent + 40% user profile
        blended = [0.6 * q + 0.4 * r for q, r in zip(query_vector, user_ctx.embedding)]
        # Re-normalize to unit vector for cosine similarity
        norm = sum(x * x for x in blended) ** 0.5
        if norm > 0:
            blended = [x / norm for x in blended]
        search_vector = blended
    else:
        search_vector = query_vector

    # ── Step 6: Qdrant semantic retrieval ──
    fetch_limit = max(payload.top_k * 5, 40)
    try:
        search_results = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=search_vector,
            limit=fetch_limit,
            with_payload=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Vector search failed.") from exc

    if not search_results:
        return []

    # ── Step 7: Build multi-signal rankings ──
    query_tokens = _tokenize(query)

    # Collect all candidate documents
    candidates = []
    for hit in search_results:
        p = hit.payload or {}
        opp_skills = [s.lower() for s in p.get("required_skills", [])]
        text_blob = f"{p.get('title', '')} {p.get('organisation', '')} {p.get('description', '')} {' '.join(opp_skills)}"

        candidates.append({
            "id": str(hit.id),
            "payload": p,
            "semantic_score": float(hit.score),
            "opp_skills": opp_skills,
            "text_blob": text_blob,
        })

    # Signal 1: Semantic — already ranked by Qdrant score
    candidates.sort(key=lambda c: c["semantic_score"], reverse=True)
    for rank, c in enumerate(candidates):
        c["semantic_rank"] = rank + 1

    # Signal 2: Keyword overlap with query
    for c in candidates:
        blob_tokens = _tokenize(c["text_blob"])
        if query_tokens:
            c["keyword_score"] = len(query_tokens & blob_tokens) / len(query_tokens)
        else:
            c["keyword_score"] = 0.0
    candidates.sort(key=lambda c: c["keyword_score"], reverse=True)
    for rank, c in enumerate(candidates):
        c["keyword_rank"] = rank + 1

    # Signal 3: Resume skill overlap (if user context exists)
    for c in candidates:
        if user_ctx:
            user_skills_set = {s.lower() for s in user_ctx.skills}
            opp_set = set(c["opp_skills"])
            matched = sorted(user_skills_set & opp_set)
            c["resume_overlap"] = matched
            c["resume_score"] = len(matched) / max(len(opp_set), 1)
        else:
            c["resume_overlap"] = []
            c["resume_score"] = 0.0
    candidates.sort(key=lambda c: c["resume_score"], reverse=True)
    for rank, c in enumerate(candidates):
        c["resume_rank"] = rank + 1

    # ── Step 8: RRF Fusion ──
    for c in candidates:
        ranks = [c["semantic_rank"], c["keyword_rank"]]
        if user_ctx:
            ranks.append(c["resume_rank"])
        c["rrf_score"] = rrf_score(ranks)

    # ── Step 9: Geo-spatial weighting ──
    for c in candidates:
        opp_location = str(c["payload"].get("location", ""))
        geo_mult = compute_geo_score(query_countries, query_cities, opp_location)
        c["geo_multiplier"] = geo_mult
        c["geo_match"] = (geo_mult > 1.0)
        c["final_score"] = c["rrf_score"] * geo_mult

    # ── Step 10: Group by company ──
    companies_map: Dict[str, dict] = {}
    for c in candidates:
        org = str(c["payload"].get("organisation", "Unknown"))
        title = str(c["payload"].get("title", ""))

        if org not in companies_map:
            companies_map[org] = {
                "organisation": org,
                "best_score": c["final_score"],
                "roles": [title],
                "locations": {c["payload"].get("location", "Unknown")},
                "geo_match": c["geo_match"],
                "resume_overlap": set(c["resume_overlap"]),
                "all_scores": [c["final_score"]],
            }
        else:
            entry = companies_map[org]
            entry["best_score"] = max(entry["best_score"], c["final_score"])
            entry["roles"].append(title)
            entry["locations"].add(c["payload"].get("location", "Unknown"))
            entry["geo_match"] = entry["geo_match"] or c["geo_match"]
            entry["resume_overlap"].update(c["resume_overlap"])
            entry["all_scores"].append(c["final_score"])

    # Aggregate company score: best_score + diminishing returns from other roles
    ranked = []
    for org, data in companies_map.items():
        scores = sorted(data["all_scores"], reverse=True)
        agg = scores[0]
        for s in scores[1:]:
            agg += s * 0.3   # diminishing contribution
        data["agg_score"] = agg

        unique_roles = list(dict.fromkeys(data["roles"]))  # deduplicate, preserve order
        desc = f"Offers roles: {', '.join(unique_roles[:3])}"
        if len(unique_roles) > 3:
            desc += f" and {len(unique_roles) - 3} more"

        overlap_list = sorted(data["resume_overlap"])

        ranked.append(CompanySearchResult(
            organisation=org,
            match_score=round(min(agg, 1.0), 4),
            description=desc,
            matched_roles=unique_roles,
            location=", ".join(sorted(data["locations"])),
            geo_match=data["geo_match"],
            resume_skill_overlap=overlap_list,
        ))

    ranked.sort(key=lambda x: x.match_score, reverse=True)
    return ranked[:payload.top_k]


# =====================================================================
# 4. QUALITY ASSURANCE — Debug / Test endpoint
# =====================================================================

@router.post("/qa_test", response_model=List[QATestCase], tags=["QA"])
async def run_qa_tests():
    """
    Run 3 simulated test cases with chain-of-thought debugging:
      1. General query
      2. Resume-dependent recommendation
      3. Country-specific search
    """
    from main import qdrant, model, COLLECTION_NAME, extract_skills, normalize_skill_set

    if not model or not qdrant:
        raise HTTPException(status_code=503, detail="ML Models or Vector DB not loaded.")

    test_cases_input = [
        {
            "name": "Case 1: General Query",
            "query": "I am looking for machine learning engineering roles at top tech companies",
            "session_id": "",
        },
        {
            "name": "Case 2: Resume-Dependent Recommendation",
            "query": "Find me roles that match my resume profile",
            "session_id": "_qa_mock_session",
        },
        {
            "name": "Case 3: Country-Specific Search",
            "query": "software engineering internships in Seattle United States",
            "session_id": "",
        },
    ]

    # Create a mock resume session for Case 2
    mock_skills = ["python", "pytorch", "machine learning", "deep learning", "nlp", "docker", "sql", "git"]
    mock_text = "Python developer with experience in PyTorch, machine learning, deep learning, NLP, Docker, SQL."
    mock_embedding = model.encode(mock_text, convert_to_tensor=False)
    if hasattr(mock_embedding, "tolist"):
        mock_embedding = mock_embedding.tolist()
    store_user_context("_qa_mock_session", mock_skills, mock_embedding, mock_text)

    results_all: List[QATestCase] = []

    for tc in test_cases_input:
        trace: List[DebugTraceStep] = []
        query = tc["query"]

        # Trace: Geo extraction
        countries, cities = extract_geo_entities(query)
        trace.append(DebugTraceStep(
            step="Geo Extraction",
            detail=f"Countries: {countries}, Cities: {cities}"
        ))

        # Trace: Skill extraction from query
        q_skills = normalize_skill_set(extract_skills(query))
        trace.append(DebugTraceStep(
            step="Query Skill Extraction",
            detail=f"Skills found: {q_skills}"
        ))

        # Trace: User context
        ctx = get_user_context(tc["session_id"]) if tc["session_id"] else None
        trace.append(DebugTraceStep(
            step="User Context",
            detail=f"Session active: {ctx is not None}, Skills: {ctx.skills if ctx else 'N/A'}"
        ))

        # Generate embedding
        qv = model.encode(query, convert_to_tensor=False)
        if hasattr(qv, "tolist"):
            qv = qv.tolist()

        if ctx:
            blended = [0.6 * q + 0.4 * r for q, r in zip(qv, ctx.embedding)]
            norm = sum(x * x for x in blended) ** 0.5
            if norm > 0:
                blended = [x / norm for x in blended]
            search_v = blended
            trace.append(DebugTraceStep(step="Embedding Blend", detail="60% query + 40% resume profile applied"))
        else:
            search_v = qv
            trace.append(DebugTraceStep(step="Embedding Blend", detail="No resume context — pure query embedding"))

        # Qdrant search
        hits = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=search_v,
            limit=30,
            with_payload=True,
        )

        trace.append(DebugTraceStep(step="Qdrant Retrieval", detail=f"Retrieved {len(hits)} candidates"))

        # Build signals
        query_tokens = _tokenize(query)
        cands = []
        for hit in hits:
            p = hit.payload or {}
            opp_skills = [s.lower() for s in p.get("required_skills", [])]
            blob = f"{p.get('title', '')} {p.get('organisation', '')} {' '.join(opp_skills)}"
            cands.append({
                "org": str(p.get("organisation", "Unknown")),
                "title": str(p.get("title", "")),
                "location": str(p.get("location", "")),
                "semantic": float(hit.score),
                "opp_skills": opp_skills,
                "blob": blob,
            })

        # Rank by semantic
        cands.sort(key=lambda c: c["semantic"], reverse=True)
        for i, c in enumerate(cands):
            c["sem_rank"] = i + 1

        # Rank by keyword
        for c in cands:
            bt = _tokenize(c["blob"])
            c["kw_score"] = len(query_tokens & bt) / max(len(query_tokens), 1)
        cands.sort(key=lambda c: c["kw_score"], reverse=True)
        for i, c in enumerate(cands):
            c["kw_rank"] = i + 1

        # Rank by resume
        for c in cands:
            if ctx:
                u_set = {s.lower() for s in ctx.skills}
                o_set = set(c["opp_skills"])
                overlap = u_set & o_set
                c["res_score"] = len(overlap) / max(len(o_set), 1)
                c["res_overlap"] = sorted(overlap)
            else:
                c["res_score"] = 0
                c["res_overlap"] = []
        cands.sort(key=lambda c: c["res_score"], reverse=True)
        for i, c in enumerate(cands):
            c["res_rank"] = i + 1

        # RRF + Geo
        debug_results: List[DebugResult] = []
        for c in cands[:10]:  # top 10 for debug
            ranks = [c["sem_rank"], c["kw_rank"]]
            if ctx:
                ranks.append(c["res_rank"])
            rrf = rrf_score(ranks)
            geo_mult = compute_geo_score(countries, cities, c["location"])
            final = rrf * geo_mult

            debug_results.append(DebugResult(
                organisation=c["org"],
                semantic_rank=c["sem_rank"],
                keyword_rank=c["kw_rank"],
                resume_rank=c["res_rank"],
                geo_multiplier=round(geo_mult, 2),
                rrf_score=round(rrf, 6),
                final_score=round(final, 6),
            ))

        debug_results.sort(key=lambda d: d.final_score, reverse=True)

        trace.append(DebugTraceStep(
            step="RRF + Geo Reranking",
            detail=f"Top result: {debug_results[0].organisation} (score={debug_results[0].final_score:.4f})" if debug_results else "No results",
        ))

        # Get actual search results via the main endpoint logic
        search_payload = HybridSearchRequest(query=query, session_id=tc["session_id"], top_k=5)
        actual_results = await hybrid_company_search(search_payload)

        results_all.append(QATestCase(
            name=tc["name"],
            query=query,
            session_id=tc["session_id"],
            results=actual_results,
            debug_trace=trace,
            debug_results=debug_results,
        ))

    return results_all
