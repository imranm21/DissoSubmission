import timeit
from openai import AzureOpenAI

def api_response_time():
    # Initialize AzureOpenAI client with provided API key and endpoint
    client = AzureOpenAI(
        azure_endpoint="[Your key goes here]",
        api_key="[Your key goes here]",
        api_version="2024-04-01-preview"
    )

    # Simulate a conversation history
    history = [
        {"role": "system", "content": "Your name is Rory..."},
        {"role": "user", "content": "What can you do for me today?"}
    ]

    # Measure how long it takes to get a response
    start_time = timeit.default_timer()

    completion = client.chat.completions.create(
        model="gpt4o",
        messages=history,
        temperature=0.7,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )

    end_time = timeit.default_timer()

    # Calculate response time
    response_time = end_time - start_time
    print(f"API call response time: {response_time:.4f} seconds")

# Call the function to test
api_response_time()
