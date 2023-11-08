# -*- coding: utf-8 -*-

import os
import openai
import time
from numpy import True_
import gradio as gr
import soundfile as sf
from pydub import AudioSegment

from openai import OpenAI

# Load API key from an environment variable
OPENAI_SECRET_KEY = os.environ.get("OPENAI_SECRET_KEY")
client = OpenAI(api_key = OPENAI_SECRET_KEY)


# Load API key from an environment variable
## openai.api_key = os.environ.get('OPENAI_SECRET_KEY')


note_transcript = ""


def transcribe(audio, history_type):
  global note_transcript    
  history_type_map = {
      "Weldon": "Weldon_Full_Visit_Format.txt",
      "Impression/Plan": "Weldon_Impression_Note_Format.txt",
      "Handover": "Weldon_Handover_Note_Format.txt",
      "Meds Only": "Medications.txt",
      "EMS": "EMS_Handover_Note_Format.txt",
      "Triage": "Triage_Note_Format.txt",
      "Ortlieb": "Ortlieb_Note_Format.txt",
      "Leinweber": "Leinweber_Note_Format.txt",
      "Cooper": "Cooper_Sports_Consult_History.txt"
   }
    
  file_name = history_type_map.get(history_type, "Weldon_History_Physical_Format.txt")
  with open(f"Format_Library/{file_name}", "r") as f:
      role = f.read()
    
  messages = [{"role": "system", "content": role}]

  ###### Create Dialogue Transcript from Audio Recording and Append(via Whisper)
  # Load the audio file (from filepath)
  audio_data, samplerate = sf.read(audio)

  #### Massage .wav and save as .mp3
  #audio_data = audio_data.astype("float32")
  #audio_data = (audio_data * 32767).astype("int16")
  #audio_data = audio_data.mean(axis=1)
  sf.write("Audio_Files/test.wav", audio_data, samplerate, subtype='PCM_16')
  sound = AudioSegment.from_wav("Audio_Files/test.wav")
  sound.export("Audio_Files/test.mp3", format="mp3")


  #Send file to Whisper for Trans"cription
  audio_file = open("Audio_Files/test.mp3", "rb")
  
    
  max_attempts = 3
  attempt = 0
  while attempt < max_attempts:
      try:
          #audio_transcript = openai.Audio.transcribe("whisper-1", audio_file)
          audio_transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
          break
      except openai.error.APIConnectionError as e:
          print(f"Attempt {attempt + 1} failed with error: {e}")
          attempt += 1
          time.sleep(3) # wait for x seconds before retrying
  else:
      print("Failed to transcribe audio after multiple attempts")  
    
  print(audio_transcript)
  #messages.append({"role": "user", "content": audio_transcript["text"]})
  messages.append({"role": "user", "content": audio_transcript["text"]})
  
  #Create Sample Dialogue Transcript from File (for debugging)
  #with open('Audio_Files/Test_Elbow.txt', 'r') as file:
  #  audio_transcript = file.read()
  #messages.append({"role": "user", "content": audio_transcript})
  

  ### Word and MB Count
  file_size = os.path.getsize("Audio_Files/test.mp3")
  mp3_megabytes = file_size / (1024 * 1024)
  mp3_megabytes = round(mp3_megabytes, 2)

  audio_transcript_words = audio_transcript["text"].split() # Use when using mic input
  #audio_transcript_words = audio_transcript.split() #Use when using file

  num_words = len(audio_transcript_words)


  #Ask OpenAI to create note transcript
  response = openai.ChatCompletion.create(model="gpt-3.5-turbo-1106", temperature=0, messages=messages)
  note_transcript = (response["choices"][0]["message"]["content"])
  print(note_transcript)
  return [note_transcript, num_words,mp3_megabytes]

#Define Gradio Interface
my_inputs = [
    gr.Audio(source="microphone", type="filepath"),
    gr.Radio(["Cooper","Weldon","Ortlieb","Leinweber","Impression/Plan","Handover","Triage","Meds Only"], show_label=False),
]

ui = gr.Interface(fn=transcribe, 
                  inputs=my_inputs, 
                  outputs=[gr.Textbox(label="Your Note").style(show_copy_button=True),
                           gr.Number(label="Audio Word Count"),
                           gr.Number(label=".mp3 MB")])


ui.launch(share=False, debug=True)