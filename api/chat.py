from flask import Flask, request, jsonify, render_template, send_file
import os
import openai
from dotenv import load_dotenv
from textblob import TextBlob

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

# Set OpenAI API Key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

def assess_sentiment(text):
    # Perform sentiment analysis using TextBlob
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity  # Range: -1 (negative) to 1 (positive)

    # Generate assessment based on sentiment
    if sentiment_score > 0.1:
        return "Positive", sentiment_score
    elif sentiment_score < -0.1:
        return "Negative", sentiment_score
    else:
        return "Neutral", sentiment_score

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        # Get sentiment assessment
        sentiment_label, sentiment_score = assess_sentiment(user_message)

        # Generate OpenAI response
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a compassionate and understanding AI therapist."},
                {"role": "user", "content": user_message},
            ],
        )
        reply = response['choices'][0]['message']['content']

        # Prepare the assessment text
        assessment_text = f"Assessment based on user input:\n\nMessage: {user_message}\n\nSentiment: {sentiment_label}\nSentiment Score: {sentiment_score}\n\nTherapist's Reply: {reply}"

        # Write the assessment to a .txt file
        with open("assessment.txt", "w") as file:
            file.write(assessment_text)

        # Return the reply and a link to download the assessment
        return jsonify({
            "reply": reply,
            "assessment_file": "/download/assessment"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/download/assessment")
def download_assessment():
    # Send the assessment.txt file to the user for download
    return send_file("assessment.txt", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
