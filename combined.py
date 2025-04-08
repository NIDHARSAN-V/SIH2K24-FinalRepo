import os
from dotenv import load_dotenv, find_dotenv
from flask import Flask, render_template, request, jsonify
from pymongo import MongoClient
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv(find_dotenv())

# Set the Groq API key
os.environ["GROQ_API_KEY"] = "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"

# Initialize LangChain model
llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)

# Create Flask app
app = Flask(__name__)

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["classy"]
domains_collection = db["Classify"]
print("DB connected")

def prompt_template(question, domain):
    return f"""
You are an expert in {domain} and skilled at difficulty classification. Your task is to classify the complexity of the following context into one of three categories: 
"easy", "medium", or "hard." Use the criteria below to guide your classification.

Context: {question}

### Classification Criteria:
1. **Easy**:
   - Refers to basic and introductory topics.
   - These topics are simple to explain and understand, requiring little prior knowledge.
   - The language is straightforward, with minimal technical terms.
   - Context will involve explanation for simple questions.

2. **Medium**:
   - Refers to moderately complex topics.
   - Requires some prior knowledge and involves concepts that take more effort to explain.
   - The language includes some technical terms requiring familiarity with the domain.

3. **Hard**:
   - Refers to technically advanced topics that require deep understanding.
   - Involves intricate details or abstract concepts.
   - The language may include many technical terms, and understanding may involve dealing with complex scenarios or multi-step processes.

### Instructions:
Carefully review the provided context and classify it based on the complexity required to understand or implement the concept. Provide your classification in a single word: easy, medium, hard.
"""

def classify_question(llm, question, domain):
    prompt = prompt_template(question, domain)
    response = llm.invoke(prompt)
    return response

def single_word(text):
    if "easy" in text.lower():
        return "easy"
    elif "medium" in text.lower():
        return "medium"
    elif "hard" in text.lower():
        return "hard"
    else:
        return "new"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        domain = request.form["domain"]
        question = request.form["question"]

        classification = classify_question(llm, question, domain)
        difficulty = single_word(classification.content)

        print("Difficulty : ", classification.content)

        return render_template("index.html", difficulty=difficulty, domain=domain, question=question)

    return render_template("index.html")

@app.route('/add-context', methods=['POST'])
def add_Context_db():
    data = request.json

    domain = data.get('domain', '').lower()
    difficulty = data.get('difficulty', '').lower()
    context = data.get('context', '')

    domain_data = domains_collection.find_one({"name": domain})

    if domain_data:
        if difficulty in domain_data['Difficulty']:
            domains_collection.update_one(
                {"name": domain},
                {"$push": {f"Difficulty.{difficulty}": context}}
            )
            return jsonify({"message": f"Inserted {difficulty} context."}), 200
        else:
            return jsonify({"error": f"Invalid difficulty level: {difficulty}."}), 400
    else:
        new_domain = {
            "name": domain,
            "Difficulty": {
                "easy": [],
                "medium": [],
                "hard": []
            }
        }
        new_domain['Difficulty'][difficulty].append(context)
        domains_collection.insert_one(new_domain)
        return jsonify({"message": "Newly inserted domain."}), 201

@app.route('/get-context/<domain>', methods=['GET'])
def get_context(domain):
    print(f"Received domain: {domain.lower()}")
    domain_data = domains_collection.find_one({"name": domain.strip().lower()})

    if domain_data:
        return jsonify({"name": domain_data['name'], "difficulty": domain_data['Difficulty']}), 200
    else:
        return jsonify({"error": "Domain not found."}), 404


if __name__ == "__main__":
    app.run(debug=True, port=6000)
