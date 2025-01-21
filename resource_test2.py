import time
import psutil  # To measure system resource usage
from openai import AzureOpenAI

def monitor_system_usage():
    # Get CPU usage percentage
    cpu_usage = psutil.cpu_percent(interval=1)
    
    # Get memory usage percentage
    memory_info = psutil.virtual_memory()
    memory_usage = memory_info.percent
    
    # Print the CPU and memory usage
    print(f"CPU Usage: {cpu_usage}%")
    print(f"Memory Usage: {memory_usage}%")

def test_inference_with_monitoring():
    client = AzureOpenAI(
        azure_endpoint="https://emtechopenaitrial.openai.azure.com/",
        api_key="163444d32fae45d198a15e9d37e549f6",
        api_version="2024-04-01-preview"
    )

    history = [
        {"role": "system", "content": "Your name is Rory, and you are a friendly assistant designed to help and converse with older adults."},
        {"role": "user", "content": "What can you do for me today?"}
    ]

    # Measure system usage before inference
    print("\n[BEFORE INFERENCE]")
    monitor_system_usage()

    # Start time for inference
    start_time = time.time()

    # Perform the inference (GPT API call)
    print("\nRunning inference...")
    completion = client.chat.completions.create(
        model="gpt4o",
        messages=history,
        temperature=0.7,
        max_tokens=800
    )

    # End time for inference
    end_time = time.time()

    # Measure system usage after inference
    print("\n[AFTER INFERENCE]")
    monitor_system_usage()

    # Calculate and print inference time
    inference_time = end_time - start_time
    print(f"\nInference Time: {inference_time:.4f} seconds")

    # Print the generated response
    print("\nGenerated Response:")
    print(completion.choices[0].message.content)

if __name__ == "__main__":
    test_inference_with_monitoring()
