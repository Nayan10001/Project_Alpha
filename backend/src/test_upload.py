"""
test_upload.py — Generate a sample PDF resume and upload it to POST /upload_resume.
Prints the full matching + gap analysis response.
"""

import json
import os
import requests
import fitz  # PyMuPDF — also works for PDF creation

API_URL = "http://localhost:8000/upload_resume"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PDF = os.path.join(BASE_DIR, "data", "sample_resume.pdf")


def create_sample_resume():
    """Create a simple 1-page PDF resume for testing."""
    doc = fitz.open()
    page = doc.new_page()

    resume_text = """
PORJAN BORDOLOI
Software Engineer | Guwahati, Assam, India
Email: porjan@example.com | GitHub: github.com/porjan

SUMMARY
Aspiring software engineer with experience in Python, JavaScript, and
machine learning. Passionate about building AI-driven products and
contributing to the Northeast India tech ecosystem.

EDUCATION
B.Tech in Computer Science & Engineering
Indian Institute of Technology Guwahati | 2022 – 2026

SKILLS
Programming: Python, JavaScript, TypeScript, C++, SQL
ML/AI: PyTorch, TensorFlow, Scikit-learn, Pandas, NumPy
Web: React, Next.js, Node.js, FastAPI, HTML, CSS, Tailwind
Tools: Git, Docker, Linux, AWS, VS Code
Other: Data Analysis, REST APIs, Agile

PROJECTS
1. Resume Matcher (Python, FastAPI, Qdrant, PyTorch)
   Built a semantic matching engine that connects students to opportunities
   using sentence-transformers and vector similarity search.

2. Northeast Startup Hub (React, Node.js, MongoDB)
   A platform showcasing startups and events in the Northeast region.
   Integrated with LinkedIn API for real-time job data.

3. Sentiment Analysis Tool (Python, NLP, Deep Learning)
   Trained a BERT-based classifier for regional language sentiment
   analysis on social media posts.

EXPERIENCE
ML Research Intern — IIT Guwahati AI Lab | Summer 2025
- Worked on NLP research for low-resource Assamese language processing
- Fine-tuned transformer models using PyTorch and Hugging Face

Software Development Intern — TechStartup Guwahati | Winter 2024
- Developed REST APIs using FastAPI and PostgreSQL
- Built a React dashboard for internal analytics
"""

    # Insert text into the PDF page
    text_rect = fitz.Rect(50, 50, 550, 800)
    page.insert_textbox(text_rect, resume_text, fontsize=10, fontname="helv")

    doc.save(SAMPLE_PDF)
    doc.close()
    print(f"✅ Sample resume created: {SAMPLE_PDF}")


def upload_and_test():
    """Upload the sample resume and display results."""
    print(f"\n📤 Uploading resume to {API_URL} …\n")

    with open(SAMPLE_PDF, "rb") as f:
        response = requests.post(
            API_URL,
            files={"file": ("sample_resume.pdf", f, "application/pdf")},
        )

    if response.status_code != 200:
        print(f"❌ Error {response.status_code}: {response.text}")
        return

    data = response.json()

    # --- Extracted Skills ---
    print("=" * 65)
    print("🧠 EXTRACTED SKILLS FROM RESUME")
    print("=" * 65)
    skills = data["extracted_skills"]
    print(f"  Found {len(skills)} skills: {', '.join(skills)}")

    # --- Top Matches ---
    print(f"\n{'=' * 65}")
    print(f"🎯 TOP {len(data['matches'])} MATCHING OPPORTUNITIES")
    print(f"{'=' * 65}")
    for i, match in enumerate(data["matches"], 1):
        print(f"\n  #{i}  {match['title']}")
        print(f"      Company:           {match['organisation']}")
        print(f"      Hybrid Score:      {match['match_score']:.4f}")
        print(f"      ├─ Semantic:       {match['semantic_score']:.4f}")
        print(f"      └─ Skill Overlap:  {match['skill_overlap_ratio']:.1%}")
        print(f"      Matched Skills:    {', '.join(match['matched_skills']) or 'None'}")
        print(f"      Missing Skills:    {', '.join(match['missing_skills']) or 'None ✨'}")

    # --- Skill Gaps ---
    print(f"\n{'=' * 65}")
    print(f"📊 SKILL GAP ANALYSIS (across all matches)")
    print(f"{'=' * 65}")
    critical = [g for g in data["skill_gaps"] if g["importance"] == "critical"]
    preferred = [g for g in data["skill_gaps"] if g["importance"] != "critical"]

    if critical:
        print(f"\n  🔴 CRITICAL (mentioned in job titles):")
        for g in critical:
            print(f"     • {g['skill']}")

    if preferred:
        print(f"\n  🟡 PREFERRED (general category skills):")
        for g in preferred:
            print(f"     • {g['skill']}")

    if not critical and not preferred:
        print("  ✨ No skill gaps — you're a perfect match!")

    print(f"\n{'=' * 65}")
    print("✅ Test complete!")
    print(f"{'=' * 65}")

    # Also dump raw JSON for inspection
    json_path = os.path.join(BASE_DIR, "data", "test_response.json")
    print(f"\n📋 Raw JSON response saved to: {json_path}")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    create_sample_resume()
    upload_and_test()
