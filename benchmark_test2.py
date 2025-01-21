import pytest
from openai import AzureOpenAI

@pytest.mark.benchmark
def test_gpt_api_response(benchmark):
    # Initialize AzureOpenAI client
    client = AzureOpenAI(
        azure_endpoint="<https://emtechopenaitrial.openai.azure.com/>",
        api_key="163444d32fae45d198a15e9d37e549f6",
        api_version="2024-04-01-preview"
    )

    history = [
        {"role": "system", "content": "Your name is Rory..."},
        {"role": "user", "content": "What can you do for me today?"}
    ]

    # Benchmark the GPT API call
    result = benchmark(lambda: client.chat.completions.create(
        model="gpt4o",
        messages=history,
        temperature=0.7,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    ))

    assert result is not None
