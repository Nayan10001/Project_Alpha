requirements.txt:
spacy
en_core_web_sm

backend/src/test_hello.py:
import spacy

def test_spacy_model_loads():
    nlp = spacy.load("en_core_web_sm")
    assert nlp is not None

