from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import os
import openai
from nltk.tokenize import sent_tokenize
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from flask import send_file

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
    "Moving or speaking so slowly that other people could have noticed?  Or the opposite — being so fidgety or restless that you have been moving around a lot more than usual.",
    "Thoughts that you would be better off dead or of hurting yourself in some way."
]

# Scoring rubric in the background
phq9_scores = {question: None for question in phq9_rubric}

def detect_negation_with_gpt(user_input, rubric_question):
    """Detect if the user input negates the given rubric question using GPT-3.5-turbo."""
    try:
        prompt = (
            f"Analyze the user input in relation to the rubric question and determine whether it negates or aligns with it.\n\n"
            f"Rubric Question: {rubric_question}\n"
            f"User Input: {user_input}\n\n"
            f"Definitions:\n"
            f"- Negation: When the user input expresses a positive or opposite meaning to the rubric question. "
            f"Examples:\n"
            f"  - Rubric Question: 'Little interest or pleasure in doing things.'\n"
            f"    User Input: 'I have a lot of interest in doing things.' (Negation)\n"
            f"  - Rubric Question: 'Feeling down, depressed, or hopeless.'\n"
            f"    User Input: 'I feel very happy and hopeful.' (Negation)\n\n"
            f"- Alignment: When the user input supports the rubric question and conveys a similar or consistent meaning. "
            f"Examples:\n"
            f"  - Rubric Question: 'Little interest or pleasure in doing things.'\n"
            f"    User Input: 'I have little interest in doing things.' (Alignment)\n"
            f"  - Rubric Question: 'Feeling down, depressed, or hopeless.'\n"
            f"    User Input: 'I feel down and hopeless.' (Alignment)\n\n"
            f"Instructions:\n"
            f"Answer 'Yes' if the user input negates the rubric question (positive or contradictory).\n"
            f"Answer 'No' if the user input supports or aligns with the rubric question (negative or consistent)."
        )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are an assistant that accurately detects negation."},
                      {"role": "user", "content": prompt}],
            temperature=0.0,
        )
        answer = response['choices'][0]['message']['content'].strip().lower()
        return answer == "yes"
    except Exception as e:
        print(f"Error in GPT negation detection: {e}")
        return False

def update_phq9_scores_with_similarity(user_input):
    """
    Analyze user input using sentence similarity, segmentation, negation detection,
    and GPT scoring for multiple matched rubric questions.
    Prevent duplicate matches and aggregate scores dynamically.
    """
    sentences = sent_tokenize(user_input.lower().strip())

    threshold = 0.60  # Similarity threshold
    matched_questions = {}  # To store matched rubric questions and their scores

    # Step 1: Match each sentence to rubric questions
    for sentence in sentences:
        sentence_embedding = similarity_model.encode(sentence, convert_to_tensor=True)
        for question in phq9_rubric:
            if phq9_scores[question] is None:  # Only consider unanswered questions
                question_embedding = similarity_model.encode(question.lower().strip(), convert_to_tensor=True)
                similarity = util.pytorch_cos_sim(sentence_embedding, question_embedding).item()

                # Background processing only (not shown to the user)
                if similarity >= threshold:
                    if question not in matched_questions:
                        matched_questions[question] = {"similarity": similarity, "matched_sentence": sentence}
                    elif similarity > matched_questions[question]["similarity"]:
                        matched_questions[question] = {"similarity": similarity, "matched_sentence": sentence}

    # Step 2: Process each matched question
    for question, details in matched_questions.items():
        matched_sentence = details["matched_sentence"]

        # Detect negation for the matched question
        negation_detected = detect_negation_with_gpt(matched_sentence, question)

        # Validate with sentiment analysis if negation is unclear
        sentiment = sentiment_analyzer(matched_sentence)[0]
        sentiment_label = sentiment['label']
        sentiment_score = sentiment['score']

        # Override negation if sentiment suggests alignment
        if negation_detected and sentiment_label == "NEGATIVE" and sentiment_score > 0.5:
            negation_detected = False

        if negation_detected:
            phq9_scores[question] = 0
        else:
            # Assign score using GPT
            score = gpt_assign_score(matched_sentence, question)
            if score is not None:
                # Aggregate score (keep the maximum score if matched multiple times)
                if phq9_scores[question] is not None:
                    phq9_scores[question] = max(phq9_scores[question], score)
                else:
                    phq9_scores[question] = score

    # Return only a generic success message
    return ""

def gpt_assign_score(user_input, rubric_question):
    """Use GPT to assign a PHQ-9 score based on user input."""
    try:
        prompt = (
            f"Given the following rubric question and user input, assign an appropriate PHQ-9 score:\n\n"
            f"Rubric Question: {rubric_question}\n"
            f"User Input: {user_input}\n\n"
            f"Use these scores:\n"
            f"0: Not at all\n1: Several days\n2: More than half the days\n3: Nearly every day\n\n"
            f"Answer with only the score (0, 1, 2, or 3)."
        )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are an assistant that assigns PHQ-9 scores."},
                      {"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return int(response['choices'][0]['message']['content'].strip())
    except Exception as e:
        print(f"Error in GPT score assignment: {e}")
        return None

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
    """
    Handle user input and provide GPT responses while updating PHQ-9 rubric.
    Only show the user's input and AI's response on the chat interface.
    """
    user_input = request.json.get('message')

    if user_input.lower() == "results":
        # Return the formatted results as a conversational message
        results_message = format_phq9_results()
        return jsonify({"message": results_message})

    # Update rubric scores (hidden from the user)
    update_phq9_scores_with_similarity(user_input)

    # Get the empathetic GPT response
    gpt_response = get_gpt_response(user_input)

    # Only return the AI's conversational response
    return jsonify({"message": gpt_response})

@app.route('/download-results')
def download_results():
    """Serve the PHQ-9 results file for download."""
    results_file_path = "phq9_results.txt"
    if os.path.exists(results_file_path):
        return send_file(results_file_path, as_attachment=True)
    else:
        return "File not found.", 404

def format_phq9_results():
    """Prepare PHQ-9 results as a conversational message."""
    questions = list(phq9_scores.keys())
    results_message = "Here are your PHQ-9 assessment results:\n\n"
    total_score = 0

    for i, question in enumerate(questions):
        score = phq9_scores[question]
        if score is None:
            results_message += f"{i + 1}. {question} - Not answered yet.\n"
        else:
            total_score += score
            results_message += f"{i + 1}. {question} - Score: {score}\n"
    
    results_message += f"\nTotal Score: {total_score}\n\nYou can ask me more questions or type 'exit' to finish."
    return results_message

if __name__ == '__main__':
    app.run(debug=True)

