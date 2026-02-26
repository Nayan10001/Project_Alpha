import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load env variables from backend/.env (and default .env if present)
load_dotenv("backend/.env")
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
	raise EnvironmentError("Missing GEMINI_API_KEY (or GOOGLE_API_KEY) in environment/.env")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-flash-latest")
response = model.generate_content("Write a 3-paragraph mentor roadmap to a student.", generation_config=genai.types.GenerationConfig(max_output_tokens=1200))

print("Text: ", response.text)
