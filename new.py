from flask import Flask, request, jsonify
from pydantic import BaseModel
from pydantic_ai import Agent
from sentence_transformers import SentenceTransformer, util
import http.client
import json

app = Flask(__name__)

# Define models for RAG pipeline
class VerificationResult(BaseModel):
    is_true: bool
    confidence: float

class MatchResult(BaseModel):
    matches: bool
    confidence: float

class TechnicalTerms(BaseModel):
    terms: list

class QuestionGeneration(BaseModel):
    question: str

# Initialize agents
verification_agent = Agent('ollama:llama3.1', result_type=VerificationResult)
matching_agent = Agent('ollama:llama3.1', result_type=MatchResult)
agent = Agent('ollama:llama3.1', result_type=TechnicalTerms)
question_agent = Agent('ollama:llama3.1', result_type=QuestionGeneration)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Helper functions
def search_passages(query):
    try:
        conn = http.client.HTTPSConnection("google.serper.dev")
        payload = json.dumps({"q": query})
        headers = {
            'X-API-KEY': '1bb5c9f6f5e617dc66f1e6de22c48b837f95a159',
            'Content-Type': 'application/json'
        }
        conn.request("POST", "/search", payload, headers)
        res = conn.getresponse()
        data = res.read()
        return json.loads(data.decode("utf-8"))
    except Exception as e:
        return {"error": str(e)}

def verify_sentence_with_rag(sentence, passages):
    context = " ".join([passage.get('snippet', '') for passage in passages.get('organic', [])])
    result = verification_agent.run_sync(
        f"Verify if the sentence '{sentence}' is true based on the following context: {context}"
    )
    return result.data.is_true, result.data.confidence

def match_answer_to_question(answer, question):
    result = matching_agent.run_sync(
        f"Does the answer '{answer}' correctly address the question '{question}'?"
    )
    return result.data.matches, result.data.confidence

def extract_technical_terms(sentence):
    result = agent.run_sync(f'Extract technical terms from: {sentence}')
    return result.data.terms

def generate_question(passage, context, difficulty):
    difficulty_level = "simple" if difficulty == 1 else "intermediate" if difficulty == 2 else "complex"
    result = question_agent.run_sync(
        f'Generate a {difficulty_level} question from the passage: {passage} and the question should be related to the context: {context}'
    )
    return result.data.question

def verify_question_relevance(sentence, question):
    sentence_embedding = model.encode(sentence, convert_to_tensor=True)
    question_embedding = model.encode(question, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(sentence_embedding, question_embedding)
    return similarity.item() > 0.7

# Flask routes
@app.route('/extract_terms', methods=['POST'])
def extract_terms():
    data = request.json or {}
    sentence = data.get('sentence', '')
    terms = extract_technical_terms(sentence)
    return jsonify({"terms": terms})

@app.route('/search', methods=['POST'])
def search():
    data = request.json or {}
    query = data.get('query', '')
    passages = search_passages(query)
    return jsonify(passages)

@app.route('/verify_sentence', methods=['POST'])
def verify_sentence():
    data = request.json or {}
    sentence = data.get('sentence', '')
    passages = data.get('passages', {})
    is_true, confidence = verify_sentence_with_rag(sentence, passages)
    return jsonify({"is_true": is_true, "confidence": confidence})

@app.route('/match_answer', methods=['POST'])
def match_answer():
    data = request.json or {}
    answer = data.get('answer', '')
    question = data.get('question', '')
    matches, confidence = match_answer_to_question(answer, question)
    return jsonify({"matches": matches, "confidence": confidence})

@app.route('/generate_question', methods=['POST'])
def generate_question_endpoint():
    data = request.json or {}
    passage = data.get('passage', '')
    context = data.get('context', '')
    difficulty = data.get('difficulty', 1)
    question = generate_question(passage, context, difficulty)
    return jsonify({"question": question})

@app.route('/verify_question_relevance', methods=['POST'])
def verify_relevance():
    data = request.json or {}
    sentence = data.get('sentence', '')
    question = data.get('question', '')
    is_relevant = verify_question_relevance(sentence, question)
    return jsonify({"is_relevant": is_relevant})

@app.route('/process_sentence', methods=['POST'])
def process_sentence():
    data = request.json or {}
    sentence = data.get('sentence', '')
    terms = extract_technical_terms(sentence)
    passages = search_passages(' '.join(terms))
    difficulty = 1
    questions = []

    for i, passage in enumerate(passages.get('organic', [])):
        snippet = passage.get('snippet', '')
        if snippet:
            if i >= 5:
                difficulty = 3
            elif i >= 3:
                difficulty = 2
            question = generate_question(snippet, sentence, difficulty)
            questions.append(question)

    return jsonify({"questions": questions})

if __name__ == '__main__':
    app.run(debug=True)
