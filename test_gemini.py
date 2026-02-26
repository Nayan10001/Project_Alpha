import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load env variables from backend/.env
load_dotenv("backend/.env")

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-flash-latest")
response = model.generate_content("Write a 3-paragraph mentor roadmap to a student.", generation_config=genai.types.GenerationConfig(max_output_tokens=1200))

print("Text: ", response.text)
