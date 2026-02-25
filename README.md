# Project_Alpha

1. Project Objective
Build a working prototype that intelligently connects students in the Northeast region to relevant internships, startups, and skill-based opportunities. The system must allow a student to upload a resume and instantly surface roles that match their skills, while explicitly flagging skill gaps they need to fill.
+2

2. Core Features Required
Resume Parsing & Embedding: Accept PDF uploads, extract text, and convert student experience/skills into semantic vectors.


Semantic Matching Engine: Perform resume-to-role matching based on skills and experience using vector similarity.


Skill-Gap Analysis: Compare extracted student skills against job requirements to output specific missing skills.


Regional Discovery Hub: Display centralized internship, startup, event, and hackathon listings specifically aligned with the Northeast region (e.g., Guwahati, Assam).


Transparency UI: Display opportunities with clear indicators of type, timeline, and eligibility.

3. Strict Constraints & Rules
The final submission must be original work built entirely during the hackathon.
+1

Ensure total transparency and authenticity of all opportunity listings.

Do not include any paid placement, brokerage features, or outcome guarantees.

4. Technology Stack Definitions
Backend Framework: FastAPI (Python).

ML/AI Layer: PyTorch, sentence-transformers (e.g., all-MiniLM-L6-v2), and PyMuPDF (fitz) for document text extraction.

Database: Qdrant (Vector Database for storing job embeddings and executing similarity search).

Frontend: React or Next.js with Tailwind CSS for rapid UI development.

5. API Endpoints Required (FastAPI)
POST /upload_resume: Accepts a PDF file, extracts text, runs NER for skills, generates a vector embedding, and queries Qdrant. Returns JSON containing top match scores and identified skill gaps.

GET /opportunities: Returns a list of all regional internships and events stored in the system.

GET /health: Basic health check for the API and Qdrant connection.

6. Directory Structure Requirements
Generate the codebase strictly using the following flat and simplified structure. Create the folders exactly as named (empty initially) and place the FastAPI entry point at the root:

Plaintext
/backend          # (Empty folder) Leave empty for manual backend module sorting later
/frontend         # (Empty folder) Leave empty for React/Next.js initialization
/data             # (Empty folder) Leave empty for PDF sample storage
main.py           # The complete FastAPI application containing all routes, PyTorch embedding logic, and Qdrant connection code
requirements.txt  # Python dependencies including fastapi, uvicorn, qdrant-client, torch, sentence-transformers, pymupdf