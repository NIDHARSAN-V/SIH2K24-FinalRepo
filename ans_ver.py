import http.client
import json
from pydantic import BaseModel
from pydantic_ai import Agent
from sentence_transformers import SentenceTransformer, util

# Step 1: Define models for the RAG pipeline
class VerificationResult(BaseModel):
    is_true: bool
    confidence: float

# Agent to verify if a statement is true
verification_agent = Agent('ollama:llama3.1', result_type=VerificationResult)

# Agent to verify if the answer matches the question
class MatchResult(BaseModel):
    matches: bool
    confidence: float

matching_agent = Agent('ollama:llama3.1', result_type=MatchResult)

def verify_sentence_with_rag(sentence: str, passages: list):
    """
    Use RAG (Retrieval-Augmented Generation) to verify if the sentence is true or false.
    """
    # Concatenate all retrieved passages
    context = " ".join([passage.get('snippet', '') for passage in passages.get('organic', [])])
    result = verification_agent.run_sync(
        f"Verify if the sentence '{sentence}' is true based on the following context: {context}"
    )
    return result.data.is_true, result.data.confidence

def match_answer_to_question(answer: str, question: str):
    """
    Use an agent to verify if the answer correctly matches the question.
    """
    result = matching_agent.run_sync(
        f"Does the answer '{answer}' correctly address the question '{question}'?"
    )
    return result.data.matches, result.data.confidence

# Step 2: Fetch passages from the web
def search_passages(query: str):
    conn = http.client.HTTPSConnection("google.serper.dev")
    payload = json.dumps({"q": query})
    headers = {
        'X-API-KEY': '1bb5c9f6f5e617dc66f1e6de22c48b837f95a159',  # Replace with your API key
        'Content-Type': 'application/json'
    }
    conn.request("POST", "/search", payload, headers)
    res = conn.getresponse()
    data = res.read()
    return json.loads(data.decode("utf-8"))

# Step 3: Process sentence for verification and matching
def process_question_and_answer(question: str, answer: str):
    
    passages = search_passages(answer)
    
    is_true, confidence_truth = verify_sentence_with_rag(answer, passages)
    
    matches, confidence_match = match_answer_to_question(answer, question)
    
    # Calculate combined score (e.g., average of both confidences)
    overall_score = (confidence_truth + confidence_match) / 2
    
    if not matches or not is_true:
        overall_score  =0
        

    # Display results
    print(f"Answer: {answer}")
    print(f"Verified Truth: {'TRUE' if is_true else 'FALSE'}, Confidence: {confidence_truth:.2f}")
    print(f"Matches Question: {'YES' if matches else 'NO'}, Confidence: {confidence_match:.2f}")
    print(f"Overall Score: {overall_score:.2f}")

# Example usage
def starter(question:str , answer:str):
   process_question_and_answer(question, answer)
