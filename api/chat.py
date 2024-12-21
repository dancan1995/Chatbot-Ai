from flask import Flask, request, jsonify, render_template, send_file
from dotenv import load_dotenv
import os
import openai
from nltk.tokenize import sent_tokenize
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import json

# Load environment variables
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OpenAI API Key not found. Ensure it is set in the .env file or as an environment variable.")

# Initialize Flask app
app = Flask(__name__)

# Initialize sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# Initialize sentence similarity model
similarity_model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

# PHQ-9 rubric template
phq9_rubric = [
    "Little interest or pleasure in doing things.",
    "Feeling down, depressed, or hopeless.",
    "Feeling tired or having little energy.",
    "Poor appetite or overeating.",
    "Trouble falling or staying asleep, or sleeping too much.",
    "Feeling bad about yourself — or that you are a failure or have let yourself or your family down.",
    "Trouble concentrating on things, such as reading the newspaper or watching television.",
    "Moving or speaking so slowly that other people could have noticed? Or the opposite — being so fidgety or restless that you have been moving around a lot more than usual.",
    "Thoughts that you would be better off dead or of hurting yourself in some way."
]

# Scoring rubric in the background
phq9_scores = {question: None for question in phq9_rubric}

# File paths
feedback_file = 'feedback.json'
history_file = 'conversation_history.json'
self_evaluation_file = 'self_evaluation.json'

# Default similarity threshold
similarity_threshold = 0.58

# Initialize files if not present
for file in [feedback_file, history_file, self_evaluation_file]:
    if not os.path.exists(file):
        with open(file, 'w') as f:
            json.dump([], f)

# Helper: Convert audio to text
def audio_to_text(audio_file_path):
    try:
        with sr.AudioFile(audio_file_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)  # Use Google Web Speech API
            return text
    except Exception as e:
        print(f"Error in speech recognition: {e}")
        return "Sorry, I couldn't process your audio."


# Route: Handle audio input
@app.route('/process-audio', methods=['POST'])
def process_audio():
    """
    Accepts an audio file, converts it to text, and processes it as a message.
    """
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"error": "Empty file"}), 400

    # Save the audio file securely
    filename = secure_filename(audio_file.filename)
    audio_path = os.path.join('uploads', filename)
    audio_file.save(audio_path)

    # Convert audio to text
    user_input = audio_to_text(audio_path)

    # Process the text input
    if user_input.lower() == "results":
        results_message = format_phq9_results()
        return jsonify({"message": results_message})

    matched = update_phq9_scores_with_similarity(user_input)

    if not matched:
        log_unsuccessful_match(user_input)

    # Get GPT response
    gpt_response = get_gpt_response(user_input)

    # Save conversation history
    conversation_history.append({"role": "user", "content": user_input})
    conversation_history.append({"role": "assistant", "content": gpt_response})
    save_conversation_history(conversation_history)

    return jsonify({"message": gpt_response})
    

# Helper: Load and save feedback
def load_feedback():
    with open(feedback_file, 'r') as f:
        return json.load(f)


def save_feedback(feedback):
    feedback_data = load_feedback()
    feedback_data.append(feedback)
    with open(feedback_file, 'w') as f:
        json.dump(feedback_data, f)


# Helper: Adjust similarity threshold dynamically
def adjust_similarity_threshold():
    feedback_data = load_feedback()
    positive_feedback = sum(1 for item in feedback_data if item == "positive")
    total_feedback = len(feedback_data)

    global similarity_threshold
    if total_feedback > 0:
        similarity_threshold = 0.50 + 0.10 * (positive_feedback / total_feedback)


# Helper: Load and save conversation history
def load_conversation_history():
    with open(history_file, 'r') as f:
        return json.load(f)


def save_conversation_history(conversation_history):
    with open(history_file, 'w') as f:
        json.dump(conversation_history, f)


# Helper: Log unsuccessful matches for self-evaluation
def log_unsuccessful_match(user_input):
    with open(self_evaluation_file, 'r') as f:
        evaluation_data = json.load(f)
    evaluation_data.append({"input": user_input, "reason": "No match found"})
    with open(self_evaluation_file, 'w') as f:
        json.dump(evaluation_data, f)


# Helper: Analyze self-evaluation logs
def analyze_self_evaluation():
    with open(self_evaluation_file, 'r') as f:
        evaluation_data = json.load(f)
    # Example: Count frequency of unsuccessful inputs
    return {"total_unmatched": len(evaluation_data)}


conversation_history = load_conversation_history()


# Route: Home
@app.route('/')
def index():
    """Render the HTML interface."""
    return render_template('index.html')


# Route: Process User Input
@app.route('/get-response', methods=['POST'])
def get_response():
    """Handle user input and provide GPT responses while updating PHQ-9 rubric."""
    user_input = request.json.get('message')

    # Store the user input in conversation history
    conversation_history.append({"role": "user", "content": user_input})

    if user_input.lower() == "results":
        # Return the formatted results as a conversational message
        results_message = format_phq9_results()
        return jsonify({"message": results_message})

    # Update rubric scores
    matched = update_phq9_scores_with_similarity(user_input)

    # Log unsuccessful matches
    if not matched:
        log_unsuccessful_match(user_input)

    # Get the empathetic GPT response
    gpt_response = get_gpt_response(user_input)

    # Store the AI's response in conversation history
    conversation_history.append({"role": "assistant", "content": gpt_response})

    # Save updated conversation history back to the file
    save_conversation_history(conversation_history)

    # Only return the AI's conversational response
    return jsonify({"message": gpt_response})


# Route: Submit Feedback
@app.route('/submit-feedback', methods=['POST'])
def submit_feedback():
    """Save user feedback on PHQ-9 scoring accuracy and GPT responses."""
    # Feedback: 'positive' or 'negative'
    feedback = request.json.get('feedback')
    # Type: 'scoring' or 'response'
    feedback_type = request.json.get('type')

    save_feedback({"feedback": feedback, "type": feedback_type})
    adjust_similarity_threshold()  # Update threshold dynamically

    return jsonify({"message": "Feedback recorded. Thank you!"})


# Route: Download Results
@app.route('/download-results')
def download_results():
    """
    Serve the PHQ-9 results file for download.
    """
    results_file_path = "phq9_results.txt"
    if os.path.exists(results_file_path):
        return send_file(results_file_path, as_attachment=True)
    else:
        return "File not found.", 404


# Route: Self-Evaluation Analysis
@app.route('/analyze-evaluation')
def analyze_evaluation():
    """Analyze and return self-evaluation metrics."""
    analysis = analyze_self_evaluation()
    return jsonify(analysis)

# Helper: Assign dynamic PHQ-9 score using GPT
def assign_dynamic_score(sentence, question):
    """
    Use GPT to dynamically assign a PHQ-9 score based on user input.
    Scores:
    0: Not at all
    1: Several days
    2: More than half the days
    3: Nearly every day
    """
    try:
        prompt = (
            f"Given the following rubric question and user input, assign an appropriate PHQ-9 score:\n\n"
            f"Rubric Question: {question}\n"
            f"User Input: {sentence}\n\n"
            f"Use these scores:\n"
            f"0: Not at all\n"
            f"1: Several days\n"
            f"2: More than half the days\n"
            f"3: Nearly every day\n\n"
            f"Answer with only the score (0, 1, 2, or 3)."
        )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a scoring assistant trained to assign PHQ-9 scores."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        return int(response['choices'][0]['message']['content'].strip())
    except Exception as e:
        print(f"Error in GPT score assignment: {e}")
        return None  


# Update the "update_phq9_scores_with_similarity" function
def update_phq9_scores_with_similarity(user_input):
    """
    Update PHQ-9 scores based on user input using dynamic scoring.
    """
    sentences = user_input.split(".")
    matched_questions = {}
    matched = False

    for sentence in sentences:
        sentence_embedding = similarity_model.encode(sentence.strip(), convert_to_tensor=True)
        for question in phq9_rubric:
            if phq9_scores[question] is None:  # Only consider unanswered questions
                question_embedding = similarity_model.encode(question, convert_to_tensor=True)
                similarity = util.pytorch_cos_sim(sentence_embedding, question_embedding).item()
                if similarity >= similarity_threshold:
                    matched_questions[question] = sentence
                    matched = True

    for question, sentence in matched_questions.items():
        # Dynamically assign score using GPT
        score = assign_dynamic_score(sentence, question)
        if score is not None:
            phq9_scores[question] = score

    return matched


# Helper: Generate GPT Response
def get_gpt_response(user_input):
    """Generate GPT response with conversation history included."""
    max_history = 5
    context = conversation_history[-max_history:]  # Get recent messages

    context_prompt = ""
    for message in context:
        role = "User" if message["role"] == "user" else "Assistant"
        context_prompt += f"{role}: {message['content']}\n"

    full_prompt = context_prompt + f"User: {user_input}\nAssistant:"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are an empathetic mental health assistant."},
                  {"role": "user", "content": full_prompt}],
        temperature=0.7
    )
    return response['choices'][0]['message']['content']

# Helper: Format PHQ-9 Results with Interpretation
def format_phq9_results():
    """
    Prepare PHQ-9 results with total score, interpretation, and a downloadable link.
    """
    results_message = "Here are your PHQ-9 assessment results:\n\n"
    total_score = 0

    # Iterate through each question and its score
    for i, (question, score) in enumerate(phq9_scores.items(), 1):
        score_text = score if score is not None else "Not Filled"
        results_message += f"{i}. {question}\nScore: {score_text}\n\n"
        if score is not None:
            total_score += score

    # Add total score
    results_message += f"Total Score: {total_score}\n"

    # Interpret the total score
    if 1 <= total_score <= 4:
        interpretation = "Minimal depression."
    elif 5 <= total_score <= 9:
        interpretation = "Mild depression."
    elif 10 <= total_score <= 14:
        interpretation = "Moderate depression."
    elif 15 <= total_score <= 19:
        interpretation = "Moderately severe depression."
    elif 20 <= total_score <= 27:
        interpretation = "Severe depression."
    else:
        interpretation = "No significant depression detected."

    results_message += f"Interpretation: {interpretation}\n\n"

    # Save results to a file
    results_file_path = "phq9_results.txt"
    with open(results_file_path, "w") as f:
        f.write(results_message)

    # Add a download link to the results
    download_link = f'<a href="/download-results" target="_blank">Download your assessment results</a>'
    results_message += f"\n{download_link}"

    return results_message


if __name__ == '__main__':
    app.run(debug=True)
