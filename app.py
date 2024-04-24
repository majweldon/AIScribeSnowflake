import os, subprocess, sys
import time
from numpy import True_
import gradio as gr
import soundfile as sf
from pydub import AudioSegment
import requests
import logging
from openai import OpenAI

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

def transcribe(audio, use_test_audio, model_name, history_type, request: gr.Request):
  global note_transcript
  history_type_map = {
      "History": "Weldon_History_Format.txt",
      "Physical": "Weldon_PE_Note_Format.txt",
      "H+P": "Weldon_History_Physical_Format.txt",
      "Impression/Plan": "Weldon_Impression_Note_Format.txt",
      "Handover": "Weldon_Handover_Note_Format.txt",
      "Meds Only": "Medications.txt",
      "EMS": "EMS_Handover_Note_Format.txt",
      "Triage": "Triage_Note_Format.txt",
      "Full Visit": "Weldon_Full_Visit_Format.txt",
      "Psych": "Weldon_Psych_Format.txt",
      "SBAR": "SBAR.txt"
   }
  
  file_name = history_type_map.get(history_type, "Weldon_Full_Visit_Format.txt")
  with open(f"Format_Library/{file_name}", "r") as f:
      role = f.read()
  
  ################# Create Dialogue Transcript from Audio Recording and Append(via Whisper)
  
  if not use_test_audio:
    audio_data, samplerate = sf.read(audio) # read audio from filepath
    
    ###################Code to convert .wav to .mp3
    sf.write("audio_files/test.wav", audio_data, samplerate, subtype='PCM_16')
    sound = AudioSegment.from_wav("audio_files/test.wav")
    sound.export("audio_files/test.mp3", format="mp3")
  else:
    # copy /Test_Audio_Files/Test_Elbow.mp3 to /audio_files/test.mp3
    subprocess.run(["cp", "Test_Audio_Files/Test_Elbow.mp3", "audio_files/test.mp3"])
    logger.info("Using test audio file")

  ###################Call Whisper Service in SCS
  service_url = os.getenv('WHISPER_API', 'http://mw-whisper-app.mw-test-jenkins.db-team-jenkins.snowflakecomputing.internal:9000/transcripe_stage_audio') 
  logger.info(f'Calling {service_url}')
  datasend = {"audio_file_path" :"/audio_files/test.mp3"}
  
  whisper_response = requests.post(url=service_url, json=datasend)
  # I just want the text of this response
  whisper_response = whisper_response.text

  if whisper_response is None:
    logger.error('Received empty response from service' + service_url)

  # Setup prompt message using mistral format
  
  prompt_txt = "<sr>[INST]" + role + whisper_response + "[/INST]" 
  logger.info(f'Prompt Text: {prompt_txt}')

  ###################Call LLM Service in SCS
  #openai_api_base = os.getenv('OPENAI_API', 'http://mw-mistral-vllm.mw-test-jenkins.db-team-jenkins.snowflakecomputing.internal:8000/v1') 
  openai_api_base = os.getenv('OPENAI_API', 'http://mw-llama3-8b-vllm.mw-test-jenkins.db-team-jenkins.snowflakecomputing.internal:8000/v1')
  logger.info(f'Calling {openai_api_base}')
  api_headers = {'Content-Type': 'application/json'}
  
  openai_api_key = "EMPTY"
  client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
    default_headers=api_headers
)
  try:
    response = client.completions.create(
      #model="/models/",  #For my Mistral Model
      model = /models/LLAMA3-8B-INSTRUCT-HF
      prompt=prompt_txt,
      max_tokens=500,
      stream=False,
      n=1,
      temperature=0,
      extra_body={"stop_token_ids":[128009]}
      )

    note_transcript = response.choices[0].text.strip() 
    logger.info(f'Note Transcript: {note_transcript}')
  except Exception as e:
    logger.error(f"Error occurred during LLM completion: {str(e)}")
    note_transcript = "Error occurred during LLM completion"

  ### Word and MB Count
  file_size = os.path.getsize("audio_files/test.mp3")
  mp3_megabytes = file_size / (1024 * 1024)
  mp3_megabytes = round(mp3_megabytes, 2)

  headers = request.headers
  sf_user = headers["Sf-Context-Current-User"]
  
  return [whisper_response, note_transcript, mp3_megabytes, sf_user]

###################### Define Gradio Interface ######################

my_inputs = [
    gr.Audio(source="microphone", type="filepath"), #Gradio 3.48.0
    gr.Checkbox(label="Use Test_Elbow.mp3 (overrides mic input)"),
    # gr.File(file_types=['.mp3', '.wav'], label="Upload Audio File (overrides microphone input - for TESTING)"),
    gr.Dropdown(['mistralai/Mistral-7B-Instruct-v0.2'], value= 'mistralai/Mistral-7B-Instruct-v0.2', label="Summarization LLM"),
    gr.Radio(["History","H+P","Impression/Plan","Full Visit","Handover","Psych","EMS","SBAR","Meds Only"], show_label=False)
    #gr.Audio(sources=["microphone"],type="numpy"), #Gradio 4.7.1
]

#get contents of description.html into a python variable
with open("description.html", "r") as file:
    description = file.read()

ui = gr.Interface(fn=transcribe,
                  inputs=my_inputs,
                  outputs=[gr.Textbox(label="Transcription)"),
                           gr.Textbox(label="Summary"),
                           gr.Number(label=".mp3 MB"), 
                           gr.Textbox(label="Snowflake User")
                           ],
                  
                  title="Jenkins in Snowflake",
                  description=description
                           )

ui.launch(share=False, debug=True, server_name="0.0.0.0", server_port=7860)

#TODO: Replace whisper with whisperX
#TODO: Add Transcription ID to save a unique identifier for each use
#TODO: Add option to upload file instead of using microphone (for testing)
#TODO: Add fields to capture LLM povenance (name, version, etc) to save back to Snowflake. Ideally this would be handled by Snowflake model registry
#TODO: Fix handling of audio files.
