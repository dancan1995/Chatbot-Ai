import openai
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def handler(request):
    user_input = request.json.get('message')  # Get the user input from the POST request
    if user_input:
        response = openai.Completion.create(
            engine="text-davinci-003",  # Using GPT-3.5 model
            prompt=user_input,
            max_tokens=150
        )
        return {
            "statusCode": 200,
            "body": {"message": response.choices[0].text.strip()}
        }
    else:
        return {
            "statusCode": 400,
            "body": {"message": "No message received."}
        }
