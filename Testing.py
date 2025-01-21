from openai import OpenAI
from azure.ai.openai import OpenAIClient
from azure.core.credentials import AzureKeyCredential
import os

# Set up your OpenAI API key
openai_api_key = "[Your key goes here]"

# Function to test OpenAI API
def test_openai_api():
    try:
        print("Initializing OpenAI Client...")
        openai_client = OpenAI(api_key=openai_api_key)
        
        print("Sending request to OpenAI API...")
        response = openai_client.Completion.create(
            model="text-davinci-003",
            prompt="Tell me a joke.",
            max_tokens=50
        )
        
        print("OpenAI Response:", response.choices[0].text.strip())
    except Exception as e:
        print("Error with OpenAI API:", str(e))

# Azure configuration
azure_endpoint = "[Your key goes here]"
azure_api_key = "[Your key goes here]" 

# Function to test Azure OpenAI API
def test_azure_openai():
    try:
        print("Initializing Azure OpenAI Client...")
        azure_client = OpenAIClient(
            endpoint=azure_endpoint,
            credential=AzureKeyCredential(azure_api_key)
        )
        
        print("Sending request to Azure OpenAI API...")
        response = azure_client.get_chat_completions(
            deployment_id="gpt-4",  # Replace with your deployment name
            messages=[{"role": "user", "content": "Tell me a joke."}],
            max_tokens=50
        )
        
        print("Azure OpenAI Response:", response.choices[0].message.content)
    except Exception as e:
        print("Error with Azure OpenAI API:", str(e))

if __name__ == "__main__":
    print("Testing OpenAI API...")
    test_openai_api()

    print("\nTesting Azure OpenAI API...")
    test_azure_openai()


