from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import os
import openai
from flask import send_file

# Load environment variables
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OpenAI API Key not found. Ensure it is set in the .env file or as an environment variable.")

# Initialize Flask app
app = Flask(__name__)

# Predefined set of questions and answers
qa_pairs = {
    "What is PHQ-9?": "The PHQ-9 is a self-report assessment tool used to screen for depression. It consists of 9 questions about the symptoms of depression.",
    "How is PHQ-9 scored?": "Each question is scored from 0 (Not at all) to 3 (Nearly every day). The total score ranges from 0 to 27 and is used to assess the severity of depression.",
    "What are the common symptoms of depression?": "Common symptoms include feeling down, little interest in activities, trouble sleeping, low energy, poor appetite, and feelings of hopelessness."
}

def get_gpt_response(user_input):
    """Get GPT response."""
    prompt = f"User: {user_input}\nHow can I help you further?"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are an empathetic mental health assistant."},
                  {"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response['choices'][0]['message']['content']

@app.route('/')
def index():
    """Render the HTML interface."""
    return render_template('index.html')

@app.route('/get-response', methods=['POST'])
def get_response():
    """Handle user input and provide a response."""
    user_input = request.json.get('message')

    # Check for a predefined answer in qa_pairs
    response_message = qa_pairs.get(user_input, "Sorry, I can only answer a few basic questions. Please ask about PHQ-9 or related topics.")

    # If not in predefined, use GPT response
    if response_message == "Sorry, I can only answer a few basic questions. Please ask about PHQ-9 or related topics.":
        response_message = get_gpt_response(user_input)

    return jsonify({"message": response_message})

@app.route('/download-results')
def download_results():
    """Serve a dummy results file for download."""
    results_file_path = "phq9_results.txt"
    if os.path.exists(results_file_path):
        return send_file(results_file_path, as_attachment=True)
    else:
        return "File not found.", 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
