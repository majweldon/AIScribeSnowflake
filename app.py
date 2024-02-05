import os, subprocess, sys
import time
from numpy import True_
import gradio as gr
import soundfile as sf
from pydub import AudioSegment
import requests
import logging

note_transcript = ""

def get_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter(
            '%(name)s [%(asctime)s] [%(levelname)s] %(message)s'))
    logger.addHandler(handler)
    return logger

logger = get_logger('service-to-service')

def transcribe(audio, request: gr.Request):
  global note_transcript
  
  ################# Create Dialogue Transcript from Audio Recording and Append(via Whisper)

  audio_data, samplerate = sf.read(audio) # read audio from filepath
    
  ###################Code to convert .wav to .mp3
  sf.write("audio_files/test.wav", audio_data, samplerate, subtype='PCM_16')
  sound = AudioSegment.from_wav("audio_files/test.wav")
  sound.export("audio_files/test.mp3", format="mp3")

  ###################Call Whister Service in SCS
 
  service_url = 'http://whisper-app.kl-test-jenkins.db-team-jenkins.snowflakecomputing.internal:9000/transcripe_stage_audio' 
  logger.info(f'Calling {service_url}')
  datasend = {
        "audio_file_path" :"/audio_files/test.mp3"
    }
  response = requests.post(url=service_url,
                           json=datasend,
                           headers={"Content-Type": "application/json"})
  
  whisper_response = response.json()
  if whisper_response is None:
    logger.error('Received empty response from service ' + service_url)

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
  
  return [note_transcript, mp3_megabytes, sf_user, whisper_response]

###################### Define Gradio Interface ######################
my_inputs = [
    gr.Audio(source="microphone", type="filepath"), #Gradio 3.48.0
    #gr.Audio(sources=["microphone"],type="numpy"), #Gradio 4.7.1
    #gr.Radio(["History","H+P","Impression/Plan","Full Visit","Handover","Psych","EMS","SBAR","Meds Only"], show_label=False),
]

ui = gr.Interface(fn=transcribe,
                  inputs=my_inputs,
                  outputs=[gr.Textbox(label="Whisper Transcription", show_copy_button=True),
                           gr.Number(label=".mp3 MB"),
                           #gr.Textbox(label="Whisper Test")
                           gr.Textbox(label="Snowflake User"),
                           gr.JSON(label="Whisper Service Response")
                           ],
                  title="Jenkins in Snowflake",
                  description="Demo of Jenkins running in Snowpark Container Services (SCS), using a Whisper and Llama2 service!"
                           )

ui.launch(share=False, debug=True, server_name="0.0.0.0", server_port=7860)
