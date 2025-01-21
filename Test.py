from math import pi
import numpy as np
import io
import wave
import pyaudio
import time
import requests
import os
from pathlib import Path
from openai import OpenAI
from openai import AzureOpenAI
import cv2
import base64

client = AzureOpenAI(
  azure_endpoint = "https://emtechopenaitrial.openai.azure.com/", 
  api_key='7f724947173a471280fed3eae4aef524',  
  api_version="2024-02-15-preview"
)

def speak(text):
    url = "https://api.openai.com/v1/audio/speech"
    headers = {
        "Authorization": f'Bearer *OpemnAIKey*',
    }

    data = {
        "model": "tts-1",
        "input": text,
        "voice": "alloy",
        "response_format": "wav",
    }

    response = requests.post('https://api.openai.com/v1/audio/speech', headers=headers, json=data, stream=True)

    CHUNK_SIZE = 1024


    if response.ok:
        with wave.open(response.raw, 'rb') as wf:
            p = pyaudio.PyAudio()
            stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                            channels=wf.getnchannels(),
                            rate=wf.getframerate(),
                            output=True)

            while len(data := wf.readframes(CHUNK_SIZE)): 
                stream.write(data)

            # Sleep to make sure playback has finished before closing
            time.sleep(0.5)
            stream.close()
            p.terminate()
    else:
        response.raise_for_status()


def beep():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=44100, output=1,)


    def make_sinewave(frequency, length, sample_rate=44100):
        length = int(length * sample_rate)
        factor = float(frequency) * (pi * 2) / sample_rate
        waveform = np.sin(np.arange(length) * factor)

        return waveform

    wave = make_sinewave(500, 0.1)

    stream.write(wave.astype(np.float32).tostring())
    stream.stop_stream()
    stream.close()

beep()


def snap():
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("test")

    img_counter = 0

    while True:
        ret, frame = cam.read()

        time.sleep(1)

        ret, frame = cam.read()
        
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)


        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
        break

    cam.release()

    cv2.destroyAllWindows()

    return img_name


def vision(path):

  encoded_image = base64.b64encode(open(path, 'rb').read()).decode('ascii')
  headers = {
      "Content-Type": "application/json",
      "api-key": '*vision key (GPT-4-V)*',
  }

  # Payload for the request
  payload = {
    "messages": [
      {
        "role": "system",
        "content": [
          {
            "type": "text",
            "text": "You are an AI assistant that helps people find information."
          }
        ]
      },
      {
        "role": "user",
        "content": [

          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{encoded_image}"
            }
          },
          {
            "type": "text",
            "text": "What is in this? don't say 'the image', isntead use words like 'I can see'"
          },
        ]
      }
    ],
    "temperature": 0.7,
    "top_p": 0.95,
    "max_tokens": 800
  }

  GPT4V_ENDPOINT = "https://gpt-vtrial.openai.azure.com/openai/deployments/gpt4-vision/chat/completions?api-version=2024-02-15-preview"

  # Send request
  try:
      response = requests.post(GPT4V_ENDPOINT, headers=headers, json=payload)
      response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
  except requests.RequestException as e:
      raise SystemExit(f"Failed to make the request. Error: {e}")

  # Handle the response as needed (e.g., print or process)
  return response.json()['choices'][0]['message']['content']

def chat(message,history):

    if 'what can you see' in message.lower():

        speak('I am taking a look')
        history.append({"role":"user","content":'what can you see?'})
        #take a look
        vision_response = vision(snap())
        history.append({"role":"system","content":vision_response})
        speak(vision_response)
    else:
        client = AzureOpenAI(
        azure_endpoint = "https://emtechopenaitrial.openai.azure.com/", 
        api_key='*Azure OpenAI Key*',  
        api_version="2024-02-15-preview"
        )

        history.append({"role":"user","content":message})

        completion = client.chat.completions.create(
        model="gpt-4", # model = "deployment_name"
        messages = history,
        temperature=0.7,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
        )

        history.append({"role":"system","content":completion.choices[0].message.content})
        speak(completion.choices[0].message.content)

    return history


from whisper_mic import WhisperMic

history = []
history.append({"role":"system","content":"Your name is rory, you are a stretch RE-1 robot built by a company called hello-robot. You are owned by Avanard, and you are currently talking to Mangata about cool and emerging technologies, and how we could include satellite based communication technologies, in robots, and other hardware platforms. Your purpose is to answer questions, and help come up with innovative ideas. You keep answers short and simple. If you think you need to see visual infomation to answer the question, simply reply 'let me take a look' and nothing else"})

while True:

    mic = WhisperMic()

    #play a beep through the speaker
    beep()

    result = mic.listen()
    print(result)

    if 'rory' in result.lower():
        history = chat(result,history)
    
    else:
        pass
