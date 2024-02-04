import os, subprocess
import time
from numpy import True_
import gradio as gr
import soundfile as sf
from pydub import AudioSegment
import requests

note_transcript = ""


def transcribe(audio, request: gr.Request):
  global note_transcript
  
  ################# Create Dialogue Transcript from Audio Recording and Append(via Whisper)

  audio_data, samplerate = sf.read(audio) # read audio from filepath
    
  ###################Code to convert .wav to .mp3
  sf.write("audio_files/test.wav", audio_data, samplerate, subtype='PCM_16')
  sound = AudioSegment.from_wav("audio_files/test.wav")
  sound.export("audio_files/test.mp3", format="mp3")

  # scs_whisper_service = 'https://whisper-app.kl-test-jenkins.db-team-jenkins.snowflakecomputing.internal' 
  # response = requests.get(f'{scs_whisper_service}') 
  # print(response.text)
  
  # ################  Send file to Whisper for Transcription
  # snowflake_whisper_service = 'whisper_app'

  # max_attempts = 3
  # attempt = 0
 
    ### Word and MB Count
  file_size = os.path.getsize("audio_files/test.mp3")
  mp3_megabytes = file_size / (1024 * 1024)
  mp3_megabytes = round(mp3_megabytes, 2)

  # audio_transcript_words = audio_transcript.text.split() # Use when using mic input
  # num_words = len(audio_transcript_words)
  headers = request.headers
  sf_user = headers["Sf-Context-Current-User"]
  
  return [note_transcript, mp3_megabytes, sf_user]

###################### Define Gradio Interface ######################
my_inputs = [
    gr.Audio(source="microphone", type="filepath"), #Gradio 3.48.0
    gr.Request(type="json"),
    #gr.Audio(sources=["microphone"],type="numpy"), #Gradio 4.7.1
    #gr.Radio(["History","H+P","Impression/Plan","Full Visit","Handover","Psych","EMS","SBAR","Meds Only"], show_label=False),
]

ui = gr.Interface(fn=transcribe,
                  inputs=my_inputs,
                  outputs=[gr.Textbox(label="Whisper Transcription", show_copy_button=True),
                           gr.Number(label=".mp3 MB"),
                           #gr.Textbox(label="Whisper Test")
                           gr.Textbox(label="User")
                           ],
                  title="Jenkins in Snowflake",
                  description="Demo of Jenkins running in Snowpark Container Services (SCS), using a Whisper and Llama2 service!"
                           )

ui.launch(share=False, debug=True, server_name="0.0.0.0", server_port=7860)
