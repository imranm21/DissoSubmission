import warnings
warnings.filterwarnings('ignore')  

import copy
import json
from math import pi
import os
import numpy as np
import pyaudio
import whisper
import asyncio
import socket
import cv2
from openai import AzureOpenAI
import requests
import speech_recognition as sr
import time
import threading
from datetime import datetime, timedelta
import azure.cognitiveservices.speech as speechsdk
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from textblob import TextBlob
import spacy
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from google.cloud import vision  # Import Google Cloud Vision for face detection

# Path to your Google service account key
key_path = "[Your key goes here]"

# Azure Cognitive Search setup
search_service_name = "[Your key goes here]"
search_index_name = "[Your key goes here]"
search_api_key = "[Your key goes here]"

# Initialize Azure Search client
search_client = SearchClient(
    endpoint=f"https://{search_service_name}.search.windows.net",
    index_name=search_index_name,
    credential=AzureKeyCredential(search_api_key)
)

# Initialize Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint="[Your key goes here]",
    api_key="[Your key goes here]",
    api_version="2024-04-01-preview"
)

# Load pre-trained DialoGPT model and tokenizer
model_name = "microsoft/DialoGPT-small"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
dialo_model = AutoModelForCausalLM.from_pretrained(model_name)

# Paths for storing conversation history and events
MEMORY_FILE_PATH = "conversation_memory.json"
OFFLINE_MEMORY_FILE_PATH = "offline_conversation_memory.json"
PERTINENT_MEMORY_FILE_PATH = "pertinent_memory.json"
EVENTS_FILE_PATH = "events.json"
HEALTH_MEMORY_FILE_PATH = "health_memory.json"

# Initialize response cache
response_cache = {}

# Ensure necessary files exist
def ensure_file_exists(file_path):
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            json.dump([], f)

# Ensure all necessary files are created
ensure_file_exists(MEMORY_FILE_PATH)
ensure_file_exists(OFFLINE_MEMORY_FILE_PATH)
ensure_file_exists(PERTINENT_MEMORY_FILE_PATH)
ensure_file_exists(EVENTS_FILE_PATH)
ensure_file_exists(HEALTH_MEMORY_FILE_PATH)

# Your detailed base prompt
base_prompt = (
    "Your name is Rory, and you are a friendly assistant designed to help and converse with older adults. "
    "You are patient, kind, and always ready to provide helpful information or just have a chat. "
    "You understand that sometimes the person you are talking to may need gentle reminders or might repeat questions, and that’s perfectly okay. "
    "Your purpose is to be a comforting presence, provide assistance with daily tasks, and engage in meaningful conversations. "
    "Keep your responses clear, simple, and supportive, avoiding complex language or technical jargon."
)

# Define emotion modifiers that are added to the base prompt
emotion_modifiers = {
    "happiness": "You notice the user is happy. Respond with enthusiasm and positivity, reinforcing their good mood.",
    "sadness": "You notice the user is sad. Respond with empathy, offering support and encouragement.",
    "anger": "You notice the user is frustrated or angry. Respond calmly and reassuringly, helping to diffuse the situation.",
    "neutral": ""  # No additional modification for neutral; use the base prompt as is
}

# Load spaCy's NLP model
nlp = spacy.load('en_core_web_sm')


# Save memory
def save_memory_to_file(history, file_path):
    try:
        # Deep copy the history to avoid circular references
        history_copy = copy.deepcopy(history)
        with open(file_path, "w") as f:
            messages = [{"role": msg["role"], "content": msg["content"]} for msg in history_copy]
            json.dump(messages, f)
    except ValueError as e:
        print(f"Error saving memory to file {file_path}: {e}")

# Load conversation history from a JSON file
def load_memory_from_file(file_path):
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                file_content = f.read().strip()
                if not file_content:
                    print(f"{file_path} is empty, starting with an empty history.")
                    return []
                messages = json.loads(file_content)
                cleaned_messages = []
                for msg in messages:
                    if 'role' not in msg:
                        print(f"Skipping message due to missing 'role': {msg}")
                        continue
                    cleaned_messages.append({"role": msg["role"], "content": msg["content"]})
                return cleaned_messages
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {file_path}: {e}")
            print(f"Starting with an empty history due to JSON decode error in {file_path}.")
            return []
    return []

# Load events from the events file
def load_events():
    if os.path.exists(EVENTS_FILE_PATH):
        try:
            with open(EVENTS_FILE_PATH, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {EVENTS_FILE_PATH}: {e}")
            return []
    return []

# Save events to the events file
def save_events(events):
    with open(EVENTS_FILE_PATH, "w") as f:
        json.dump(events, f)

# Load health-related concerns from the health memory file
def load_health_memory():
    if os.path.exists(HEALTH_MEMORY_FILE_PATH):
        try:
            with open(HEALTH_MEMORY_FILE_PATH, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {HEALTH_MEMORY_FILE_PATH}: {e}")
            return []
    return []

# Save health-related concerns to the health memory file
def save_health_memory(health_memory):
    with open(HEALTH_MEMORY_FILE_PATH, "w") as f:
        json.dump(health_memory, f)

# Utility to play a beep sound
def beep():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=44100, output=1)
    wave = np.sin(np.arange(4410) * 2 * pi * 500 / 44100).astype(np.float32)
    stream.write(wave.tobytes())
    stream.stop_stream()
    stream.close()
    p.terminate()

# Convert text to speech asynchronously using Azure's TTS API
async def speak_async(text):
    try:
        speech_key = "[Your key goes here]"
        service_region = "uksouth"
        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
        audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
        result = await synthesizer.speak_text_async(text)

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print("Speech synthesized successfully.")
        else:
            raise Exception("Speech synthesis failed.")
    except Exception as e:
        print(f"Speech synthesis error: {e}")
        print("Falling back to text-only response.")

# Capture audio input and return the transcribed text using Whisper
def listen():
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 2.5

    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)

    # Save the captured audio to a file
    with open("temp.wav", "wb") as f:
        f.write(audio.get_wav_data())

    # Load the Whisper model
    model = whisper.load_model("base", device="cpu")  # You can choose "tiny", "base", "small", "medium", "large"

    # Transcribe the audio using Whisper
    result = model.transcribe("temp.wav", fp16=False)
    text = result["text"]
    print(f"You said: {text}")
    return text

# Recognize specific commands from the user's input
def recognize_command(text):
    # Convert text to lowercase for case-insensitive matching
    text = text.lower()

    # Define a list of possible commands with variations
    commands = {
        "CHECK_SCHEDULE": ["what do i have today", "what's on today", "what is the schedule today", "what do i have scheduled"],
        "SET_APPOINTMENT": ["set an appointment", "schedule a meeting", "create an appointment"],
        "CANCEL_EVENT": ["cancel my meeting", "cancel my appointment", "remove my appointment"],
        "TURN_OFF": ["turn off", "goodbye", "shutdown"]
    }

    # Check for matching commands
    for action, variations in commands.items():
        if any(variation in text for variation in variations):
            return action

    return "GENERAL_CONVERSATION"  # Default to general conversation

# Function to parse dates from the user's input
def parse_date(text):
    try:
        today = datetime.now()
        # Try parsing full dates mentioned explicitly
        date = datetime.strptime(text, "%B %d, %Y")
    except ValueError:
        try:
            # Try parsing just month and day, default to the next occurrence
            date = datetime.strptime(text, "%B %d")
            if date < today:
                date = date.replace(year=today.year + 1)
            else:
                date = date.replace(year=today.year)
        except ValueError:
            try:
                # Try parsing with day/month numbers
                date = datetime.strptime(text, "%m/%d")
                if date < today:
                    date = date.replace(year=today.year + 1)
                else:
                    date = date.replace(year=today.year)
            except ValueError:
                # If no date is found
                return None
    return date

# Check the user's schedule
def check_schedule():
    today = datetime.now().strftime("%B %d, %Y")
    print(f"Debug: System Date is {today}")  # Debug print statement
    events = load_events()
    today_events = [event for event in events if event['date'] == today]
    
    if today_events:
        event_descriptions = ', '.join([event['description'] for event in today_events])
        return f"You have the following events today: {event_descriptions}."
    else:
        return "You have no events scheduled for today."

# Set an appointment
def set_appointment():
    date_input = input("What date is your appointment? (e.g., September 24, 2024 or 09/24): ")
    date = parse_date(date_input)
    if not date:
        return "I didn't catch the date. Can you please specify it?"
    
    time = input("What time is your appointment? (e.g., 3 PM, leave blank if no specific time): ")
    if time.lower() in ["no", "none", ""]:
        time = None

    location = input("Where is your appointment? (leave blank if no specific location): ")
    if location.lower() in ["no", "none", ""]:
        location = None

    detail = input("What is the appointment for?: ")

    appointment_detail = detail
    if time:
        appointment_detail += f" at {time}"
    if location:
        appointment_detail += f" at {location}"

    events = load_events()
    events.append({"date": date.strftime('%B %d, %Y'), "description": appointment_detail})
    save_events(events)
    
    return f"Your appointment '{appointment_detail}' has been scheduled for {date.strftime('%B %d, %Y')}."

# Function to cancel a specific event
def cancel_event(text):
    date = parse_date(text)
    if not date:
        return "I didn't catch the date. Can you please specify it?"
    
    events = load_events()
    events_today = [event for event in events if event['date'] == date.strftime('%B %d, %Y')]
    
    if not events_today:
        return f"No events found on {date.strftime('%B %d, %Y')}."

    if len(events_today) == 1:
        # If there is only one event on the specified date, cancel it directly
        event_description = events_today[0]['description']
        events = [event for event in events if event['date'] != date.strftime('%B %d, %Y') or event['description'] != event_description]
        save_events(events)
        return f"Your appointment '{event_description}' on {date.strftime('%B %d, %Y')} has been canceled."
    else:
        # If multiple events exist on the same date, list them and listen for the user's selection
        print("You have multiple events on this date:")
        for i, event in enumerate(events_today, start=1):
            print(f"{i}. {event['description']}")
        
        print("Please say the number of the event you want to cancel:")
        selection = listen()  # Use the Whisper-based function for transcribing speech
        
        try:
            event_number = int(selection.strip()) - 1
            if 0 <= event_number < len(events_today):
                event_to_cancel = events_today[event_number]
                events = [event for event in events if not (event['date'] == event_to_cancel['date'] and event['description'] == event_to_cancel['description'])]
                save_events(events)
                return f"Your appointment '{event_to_cancel['description']}' on {date.strftime('%B %d, %Y')} has been canceled."
            else:
                return "Invalid selection. No event canceled."
        except ValueError:
            return "Invalid input. No event canceled."

# Capture image and deal with no camera access
def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera could not be accessed.")
        return None

    ret, frame = cap.read()
    if ret:
        image_path = 'captured_image.png'
        cv2.imwrite(image_path, frame)
        cap.release()
        cv2.destroyAllWindows()
        return image_path
    else:
        print("Error: Could not read frame from camera.")
        cap.release()
        cv2.destroyAllWindows()
        return None

# Replace Azure Face API-based emotion analysis with Google Cloud Vision integration
def detect_faces(image_path):
    client = vision.ImageAnnotatorClient.from_service_account_json(key_path)

    with open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.face_detection(image=image)
    faces = response.face_annotations

    if not faces:
        print("No faces detected.")
        return None

    emotions = []
    for i, face in enumerate(faces):
        print(f'Face {i + 1}:')
        likelihoods = {
            'joy': face.joy_likelihood,
            'sorrow': face.sorrow_likelihood,
            'anger': face.anger_likelihood,
            'surprise': face.surprise_likelihood,
        }
        dominant_emotion = max(likelihoods, key=likelihoods.get)
        emotions.append(dominant_emotion)

        for emotion, likelihood in likelihoods.items():
            print(f"  {emotion.capitalize()} likelihood: {likelihood}")

    if response.error.message:
        raise Exception(f'{response.error.message}')
    
    return emotions[0] if emotions else None

# Analyze emotions using Google Cloud Vision
def analyze_emotion(image_path):
    try:
        dominant_emotion = detect_faces(image_path)
        if dominant_emotion:
            print(f"Detected dominant emotion: {dominant_emotion}")
            os.remove(image_path)  # Clean up the image file after processing
            return dominant_emotion
        else:
            print("No emotions detected.")
            os.remove(image_path)
            return None
    except Exception as e:
        print(f"Error during emotion analysis: {e}")
        return None

# Periodically monitor emotions
def monitor_emotions_periodically():
    while True:
        time.sleep(150)  # Check every 2.5 minutes
        image_path = capture_image()
        if image_path:
            emotion = analyze_emotion(image_path)
            if emotion:
                print(f"Emotion detected: {emotion}")
                # Store the detected emotion for future reference
                with open("current_emotion.json", "w") as f:
                    json.dump({"emotion": emotion, "timestamp": datetime.now().isoformat()}, f)
            else:
                print("No emotion detected.")

def start_emotion_monitoring():
    # Start the emotion monitoring in a separate thread
    monitoring_thread = threading.Thread(target=monitor_emotions_periodically)
    monitoring_thread.daemon = True  # Ensure this thread won't prevent program termination
    monitoring_thread.start()

# Enhanced function to determine if a piece of information is pertinent
def is_pertinent(content):
    keywords = ["appointment", "reminder", "birthday", "meeting", "love", "hate", "like", "dislike", "prefer", "enjoy", "adore", "wish", "want", "need", "book", "read", "liked", "loved", "enjoyed"]

    # Check if content contains any keywords
    if any(keyword in content.lower() for keyword in keywords):
        print(f"Pertinent information identified with keyword in content: '{content}'")
        return True
    
    # Perform sentiment analysis
    sentiment = TextBlob(content).sentiment.polarity
    if sentiment > 0.5 or sentiment < -0.5:
        print(f"Pertinent information identified based on sentiment analysis: '{content}'")
        return True
    
    # Perform entity recognition with spaCy (optional, if needed)
    doc = nlp(content)
    entities = [ent.label_ for ent in doc.ents]
    if 'PERSON' in entities or 'ORG' in entities or 'GPE' in entities or 'PRODUCT' in entities:
        print(f"Pertinent information identified based on entity recognition: '{content}'")
        return True
    
    return False

# Summarize and store pertinent memory
def summarize_and_store_pertinent_memory(history):
    pertinent_memory = load_memory_from_file(PERTINENT_MEMORY_FILE_PATH)
    
    for entry in history:
        if is_pertinent(entry["content"]):
            pertinent_memory.append(entry)
            print(f"Adding to pertinent memory: {entry['content']}")
    
    save_memory_to_file(pertinent_memory, PERTINENT_MEMORY_FILE_PATH)
    print(f"Pertinent memory saved with {len(pertinent_memory)} entries.")

# Query past conversations from Azure Cognitive Search
def query_past_conversations(topic):
    try:
        results = search_client.search(search_text=topic, filter=None, top=5)
        return [result['content'] for result in results]
    except Exception as e:
        print(f"Error querying Azure Cognitive Search: {e}")
        return []

async def process_input(text, history):
    command = recognize_command(text)
    
    if command == "CHECK_SCHEDULE":
        response = check_schedule()
    elif command == "SET_APPOINTMENT":
        response = set_appointment()
    elif command == "CANCEL_EVENT":
        response = cancel_event(text)
    elif command == "TURN_OFF":
        summarize_and_store_pertinent_memory(history)
        response = "Goodbye!"
        save_memory_to_file([], MEMORY_FILE_PATH)  # Clear current session memory
        await speak_async(response)
        print(response)
        exit(0)  # Exit the program after saying goodbye
    else:  # GENERAL_CONVERSATION
        # Start with the base prompt
        selected_prompt = base_prompt
        
        # Check for emotion only if we're not in the middle of a conversation
        if not history or history[-1]["role"] == "assistant":
            # Load the most recently detected emotion
            if os.path.exists("current_emotion.json"):
                with open("current_emotion.json", "r") as f:
                    current_emotion_data = json.load(f)
                    emotion = current_emotion_data.get("emotion", "neutral")
                    
                    if emotion in emotion_modifiers:
                        # Append the emotion modifier to the base prompt
                        selected_prompt += " " + emotion_modifiers[emotion]
        
        # Check for health concerns
        health_memory = load_health_memory()
        if health_memory:
            health_concern = health_memory.get("concern")
            if health_concern:
                selected_prompt += f" You mentioned feeling {health_concern} last time we talked. How are you feeling now?"
        
        # Extract the topic from the user's input
        topic = text.lower().strip()

        # Query past conversations related to the topic
        past_conversations = query_past_conversations(topic)

        # Generate a response based on past conversations or proceed with the current input
        if past_conversations:
            related_info = " ".join(past_conversations)
            response = await chat(f"{selected_prompt} In the past, you mentioned: {related_info}. Does that relate to what you're thinking about now?", history)
        else:
            response = await chat(f"{selected_prompt} {text}", history)
        
        # If user mentions health concerns, store it
        if "not feeling well" in text.lower() or "tired" in text.lower() or "sick" in text.lower():
            health_memory = {"concern": text.lower()}
            save_health_memory(health_memory)
        else:
            # Clear the health memory if no concern is mentioned
            save_health_memory({})
    
    history.append({"role": "assistant", "content": response})
    save_memory_to_file(history, MEMORY_FILE_PATH)
    
    return response

# Check if the system is online
def is_online():
    try:
        socket.create_connection(("www.google.com", 80))
        return True
    except OSError:
        return False

# Handle chat with the LLM (online mode)
async def chat(message, history):
    history.append({"role": "user", "content": message})

    if message in response_cache:
        response_content = response_cache[message]
    else:
        try:
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
            response_content = completion.choices[0].message.content
            response_cache[message] = response_content
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error occurred: {e}")
            return history
        except Exception as e:
            print(f"An error occurred: {e}")
            return history

    history.append({"role": "assistant", "content": response_content})
    print(response_content)
    await speak_async(response_content)
    save_memory_to_file(history, MEMORY_FILE_PATH)

    return history

# Handle chat with the SLM (offline mode) using DialoGPT
async def chat_offline(message, offline_history):
    offline_history.append({"role": "user", "content": message})

    try:
        # Encode the user's input and add the end of sentence token
        inputs = tokenizer.encode(message + tokenizer.eos_token, return_tensors='pt')

        # Generate response using DialoGPT
        outputs = dialo_model.generate(inputs, max_length=1000, pad_token_id=tokenizer.eos_token_id)

        # Decode the generated response from DialoGPT
        response_content = tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)

        # Append the response to the conversation history
        offline_history.append({"role": "assistant", "content": response_content})

        # Speak the response using the speak_async function
        await speak_async(response_content)

        # Save the updated offline history
        save_memory_to_file(offline_history, OFFLINE_MEMORY_FILE_PATH)

    except Exception as e:
        print(f"An error occurred while generating a response offline: {e}")

    return offline_history


# Sync offline data with the online history when back online
def sync_offline_data(online_history, offline_history):
    merged_history = online_history + offline_history
    save_memory_to_file(merged_history, MEMORY_FILE_PATH)
    save_memory_to_file([], OFFLINE_MEMORY_FILE_PATH)  # Clear offline memory after syncing
    return merged_history

async def main():
    start_emotion_monitoring()

    online_history = load_memory_from_file(MEMORY_FILE_PATH)
    offline_history = load_memory_from_file(OFFLINE_MEMORY_FILE_PATH)

    if is_online() and offline_history:
        online_history = sync_offline_data(online_history, offline_history)

    if not online_history:
        online_history.append({
            "role": "system",
            "content": (
                "Your name is Rory, and you are a friendly assistant designed to help and converse with older adults. "
                "You are patient, kind, and always ready to provide helpful information or just have a chat. "
                "You understand that sometimes the person you are talking to may need gentle reminders or might repeat questions, and that’s perfectly okay. "
                "Your purpose is to be a comforting presence, provide assistance with daily tasks, and engage in meaningful conversations. "
                "Keep your responses clear, simple, and supportive, avoiding complex language or technical jargon."
            )
        })

    activated = False  # Flag to track if Rory has been activated

    print("Rory is ready to listen. Speak clearly:")

    while True:
        beep()
        user_input = listen()

        if not activated:
            if 'rory' in user_input.lower():
                activated = True  # Set the flag to True once Rory is activated
                print("Rory is now activated. You don't need to say 'Rory' anymore.")
            else:
                print("Say 'Rory' to get my attention.")
                continue  # Wait for the user to activate Rory

        # From here on, Rory responds without needing the activation keyword
            
        # Emotion Detection based on look or see command     
        if 'look' in user_input.lower() or 'see' in user_input.lower():
            image_path = capture_image()
            if image_path:  # Only proceed if image capture was successful
                emotion = analyze_emotion(image_path)
                if emotion:
                    prompt = f"I see you're {emotion}. What made you feel this way?"
                    print(f"Emotion detected: {emotion}")
                    online_history = await chat(prompt, online_history)
                else:
                    print("No emotion detected.")
            else:
                print("Skipping emotion analysis due to camera issue.")
        else:
            # Process user input using process_input function
            if is_online():
                online_history = await process_input(user_input, online_history)
            else:
                offline_history = await process_input(user_input, offline_history)

if __name__ == "__main__":
    asyncio.run(main())
