# import http.client
# import json
# from pydantic import BaseModel
# from pydantic_ai import Agent

# # Step 1: Define the output model for the agent (extracted technical terms)
# class TechnicalTerms(BaseModel):
#     terms: list

# # Step 2: Create the agent to extract technical terms from the sentence
# agent = Agent('ollama:llama3.1', result_type=TechnicalTerms)

# def extract_technical_terms(sentence: str):
#     result = agent.run_sync(f'Extract technical terms from: {sentence}')
#     return result.data.terms

# # Step 3: Web search using Serper API (via HTTP request)
# def search_passages(terms: list):
#     search_query = ' '.join(terms)
    
#     # Define the connection and request parameters for Serper API
#     conn = http.client.HTTPSConnection("google.serper.dev")
#     payload = json.dumps({"q": search_query})
#     headers = {
#         'X-API-KEY': '1bb5c9f6f5e617dc66f1e6de22c48b837f95a159',  # Replace with your API key
#         'Content-Type': 'application/json'
#     }
    
#     # Send the POST request to Serper API
#     conn.request("POST", "/search", payload, headers)
    
#     # Get the response
#     res = conn.getresponse()
#     data = res.read()
    
#     # Decode and return the response as a dictionary
#     return json.loads(data.decode("utf-8"))

# # Step 4: Create an agent for question generation
# class QuestionGeneration(BaseModel):
#     question: str

# question_agent = Agent('ollama:llama3.1', result_type=QuestionGeneration)

# def generate_question(passage: str):
#     result = question_agent.run_sync(f'Generate a question from the passage: {passage}')
#     return result.data.question

# # Putting everything together
# def process_sentence_for_question(sentence: str):
#     # Step 1: Extract technical terms from the sentence
#     terms = extract_technical_terms(sentence)
    
#     # Step 2: Search the web for relevant passages using the terms
#     passages = search_passages(terms)
    
#     # Step 3: Generate questions from the retrieved passages
#     for passage in passages.get('organic', []):
#         snippet = passage.get('snippet', '')
#         if snippet:  # Only generate a question if a snippet is present
#             question = generate_question(snippet)
#             print("Generated Question:", question)

# # Example usage
# sentence = "React Js is a Framework of javascript to build frontend"
# process_sentence_for_question(sentence)







import http.client
import json
from pydantic import BaseModel
from pydantic_ai import Agent
from sentence_transformers import SentenceTransformer, util
from ans_ver import starter  # Ensure this is imported correctly

# Step 1: Define the output model for the agent (extracted technical terms)
class TechnicalTerms(BaseModel):
    terms: list

# Step 2: Create the agent to extract technical terms from the sentence
agent = Agent('ollama:llama3.1', result_type=TechnicalTerms)

def extract_technical_terms(sentence: str):
    result = agent.run_sync(f'Extract technical terms from: {sentence}')
    return result.data.terms

# Step 3: Web search using Serper API (via HTTP request)
def search_passages(terms: list):
    search_query = ' '.join(terms)
    
    # Define the connection and request parameters for Serper API
    conn = http.client.HTTPSConnection("google.serper.dev")
    payload = json.dumps({"q": search_query})
    headers = {
        'X-API-KEY': '1bb5c9f6f5e617dc66f1e6de22c48b837f95a159',  # Replace with your API key
        'Content-Type': 'application/json'
    }
    
    # Send the POST request to Serper API
    conn.request("POST", "/search", payload, headers)
    
    # Get the response
    res = conn.getresponse()
    data = res.read()
    
    # Decode and return the response as a dictionary
    return json.loads(data.decode("utf-8"))

# Step 4: Create an agent for question generation
class QuestionGeneration(BaseModel):
    question: str

question_agent = Agent('ollama:llama3.1', result_type=QuestionGeneration)

def generate_question(passage: str, context: str, difficulty: int):
    # Increase question difficulty based on the 'difficulty' parameter
    difficulty_level = "simple" if difficulty == 1 else "intermediate" if difficulty == 2 else "complex"
    result = question_agent.run_sync(f'Generate a {difficulty_level} question from the passage: {passage} and the question should be related to the context: {context}')
    return result.data.question

# Step 5: Verify if the generated question is related to the input sentence using embeddings
def verify_question_relevance(sentence: str, question: str) -> bool:
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Using pre-trained sentence embeddings model
    sentence_embedding = model.encode(sentence, convert_to_tensor=True)
    question_embedding = model.encode(question, convert_to_tensor=True)
    
    # Compute cosine similarity
    similarity = util.pytorch_cos_sim(sentence_embedding, question_embedding)
    
    # Threshold for relevance (you can adjust this value)
    return similarity.item() > 0.7  # If similarity > 0.7, consider it relevant


def process_sentence_for_question(sentence: str):
    # Step 1: Extract technical terms from the sentence
    terms = extract_technical_terms(sentence)
    
    # Step 2: Search the web for relevant passages using the terms
    passages = search_passages(terms)
    
    # Step 3: Generate questions from the retrieved passages and verify relevance
    difficulty = 1  # Start with easy questions
    for i, passage in enumerate(passages.get('organic', [])):  # Check if 'organic' exists in the response
        snippet = passage.get('snippet', '')
        if snippet:  # Only generate a question if a snippet is present
            # Adjust difficulty level based on the passage index
            if i >= 5:
                difficulty = 3  # Hard
            elif i >= 3:
                difficulty = 2  # Medium
            question = generate_question(snippet, sentence, difficulty)
            print(f"Generated Question (Difficulty {difficulty}):", question)
            
            
            answer = input("Enter your answer: ").strip()  
            
            starter(question, answer)  
            # Validate answer input
            if answer == "":
                print("No answer was provided.")
            else:
                print(f"Answer provided: {answer}")
            
            
            is_relevant = verify_question_relevance(sentence, question)
            if is_relevant:
                print("The question is relevant to the input sentence.")
            else:
                print("The question is not relevant to the input sentence.")


sentence = "React is a free and open-source front-end JavaScript library that aims to make building user interfaces based on components more seamless"
process_sentence_for_question(sentence)
