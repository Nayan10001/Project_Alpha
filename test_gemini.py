import os
import google.generativeai as genai

genai.configure(api_key="")
model = genai.GenerativeModel("gemini-flash-latest")
response = model.generate_content("Write a 3-paragraph mentor roadmap to a student.", generation_config=genai.types.GenerationConfig(max_output_tokens=1200))

print("Text: ", response.text)
