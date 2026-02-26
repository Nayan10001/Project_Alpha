"""
seed_qdrant.py — Seed the Qdrant 'opportunities' collection with data from data.csv

Usage:
    python seed_qdrant.py
"""

import csv
import os
import uuid
import logging
from typing import List

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "opportunities")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = 384
CSV_PATH = os.path.join(os.path.dirname(__file__), "../data.csv")
BATCH_SIZE = 32

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger("seed_qdrant")

# ---------------------------------------------------------------------------
# Skill inference — maps category keywords & title keywords to skill lists.
# This gives the matching engine something concrete to compare against resumes.
# ---------------------------------------------------------------------------
CATEGORY_SKILLS = {
    "Software Engineering": [
        "python", "java", "javascript", "c++", "git", "data structures",
        "algorithms", "sql", "linux", "docker", "rest api",
    ],
    "AI/ML Engineering": [
        "python", "pytorch", "tensorflow", "machine learning", "deep learning",
        "nlp", "scikit-learn", "pandas", "numpy", "docker", "sql",
        "computer vision", "data analysis",
    ],
    "Data Science/ML": [
        "python", "sql", "pandas", "numpy", "scikit-learn", "machine learning",
        "data analysis", "statistics", "tableau", "r", "deep learning",
    ],
    "DevOps/Cloud": [
        "docker", "kubernetes", "aws", "gcp", "azure", "linux", "ci/cd",
        "terraform", "git", "python", "bash",
    ],
    "Frontend/Backend": [
        "javascript", "typescript", "react", "next.js", "node.js", "html",
        "css", "tailwind", "git", "rest api", "sql",
    ],
    "Product Management": [
        "product strategy", "agile", "data analysis", "sql", "user research",
        "roadmap planning", "a/b testing", "jira", "figma",
    ],
    "Cybersecurity": [
        "networking", "linux", "python", "firewalls", "siem",
        "vulnerability assessment", "penetration testing", "encryption",
        "security compliance", "incident response",
    ],
}

# Extra skills inferred from title keywords
TITLE_SKILL_MAP = {
    "android": ["android", "kotlin", "java"],
    "ios": ["ios", "swift"],
    "frontend": ["javascript", "react", "html", "css", "typescript"],
    "fullstack": ["javascript", "react", "node.js", "sql", "rest api"],
    "full-stack": ["javascript", "react", "node.js", "sql", "rest api"],
    "backend": ["python", "java", "sql", "docker", "rest api"],
    "deep learning": ["pytorch", "tensorflow", "deep learning"],
    "nlp": ["nlp", "python", "pytorch"],
    "llm": ["nlp", "python", "pytorch", "deep learning"],
    "cloud": ["aws", "gcp", "azure", "docker", "kubernetes"],
    "devops": ["docker", "kubernetes", "ci/cd", "linux", "terraform"],
    "data scientist": ["python", "sql", "pandas", "machine learning", "statistics"],
    "machine learning": ["python", "machine learning", "pytorch", "scikit-learn"],
    "ai": ["python", "machine learning", "deep learning"],
    "kernel": ["c++", "c", "linux", "python"],
    "payment": ["sql", "rest api", "python"],
    "security": ["networking", "linux", "python", "encryption"],
    "cyber": ["networking", "linux", "firewalls", "siem"],
}


def infer_skills(title: str, category: str) -> List[str]:
    """Combine category-level and title-level skill keywords."""
    skills = set(CATEGORY_SKILLS.get(category, []))
    title_lower = title.lower()
    for keyword, extra_skills in TITLE_SKILL_MAP.items():
        if keyword in title_lower:
            skills.update(extra_skills)
    return sorted(skills)


def infer_eligibility(job_type: str) -> str:
    if job_type == "Internship":
        return "Current students or recent graduates"
    return "Open to all qualified candidates"


def infer_timeline(job_type: str, date_scraped: str) -> str:
    if job_type == "Internship":
        return f"Summer 2026 (posted {date_scraped})"
    return f"Open / Rolling (posted {date_scraped})"


def build_embedding_text(row: dict, skills: List[str]) -> str:
    """Create a rich text blob for embedding."""
    parts = [
        row["Job Title"],
        row["Company"],
        row["Category"],
        row["Job Type"],
        row["Location"],
        "Skills: " + ", ".join(skills),
    ]
    return ". ".join(parts)


def main():
    # 1. Load CSV
    logger.info("Reading %s …", CSV_PATH)
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    logger.info("Loaded %d rows.", len(rows))

    # 2. Load model
    logger.info("Loading embedding model: %s …", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info("Model loaded.")

    # 3. Connect to Qdrant & ensure collection
    logger.info("Connecting to Qdrant at %s:%s …", QDRANT_HOST, QDRANT_PORT)
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        logger.info("Deleting existing collection '%s' to re-seed.", COLLECTION_NAME)
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    logger.info("Created fresh collection '%s'.", COLLECTION_NAME)

    # 4. Build points
    points: List[PointStruct] = []
    texts: List[str] = []

    for row in rows:
        skills = infer_skills(row["Job Title"], row["Category"])
        text = build_embedding_text(row, skills)
        texts.append(text)

        payload = {
            "title": row["Job Title"],
            "organisation": row["Company"],
            "type": row["Job Type"].lower(),          # internship | full-time
            "category": row["Category"],
            "location": row["Location"],
            "description": f'{row["Job Title"]} at {row["Company"]} ({row["Category"]}, {row["Job Type"]})',
            "required_skills": skills,
            "timeline": infer_timeline(row["Job Type"], row["Date Scraped"]),
            "eligibility": infer_eligibility(row["Job Type"]),
            "linkedin_url": row["LinkedIn URL"],
            "date_scraped": row["Date Scraped"],
        }
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                payload=payload,
                vector=[],  # placeholder, filled below
            )
        )

    # 5. Generate embeddings in batches
    logger.info("Generating embeddings for %d items …", len(texts))
    all_embeddings = model.encode(texts, show_progress_bar=True, batch_size=BATCH_SIZE)

    for point, emb in zip(points, all_embeddings):
        point.vector = emb.tolist()

    # 6. Upsert into Qdrant in batches
    logger.info("Upserting %d points into Qdrant …", len(points))
    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i : i + BATCH_SIZE]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        logger.info("  Upserted batch %d–%d", i + 1, min(i + BATCH_SIZE, len(points)))

    # 7. Verify
    info = client.get_collection(COLLECTION_NAME)
    logger.info(
        "✅ Done! Collection '%s' now has %d vectors.",
        COLLECTION_NAME,
        info.points_count,
    )


if __name__ == "__main__":
    main()
