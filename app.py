import os, subprocess
import openai
import time
from numpy import True_
import gradio as gr
import soundfile as sf
from pydub import AudioSegment
from openai import OpenAI
#from mistralai.client import MistralClient
#from mistralai.models.chat_completion import ChatMessage
import replicate

# Load API keys from an environment variables
OPENAI_SECRET_KEY = os.environ.get("OPENAI_SECRET_KEY")
REPLICATE_API_KEY = os.environ.get("REPLICATE_API_TOKEN")
subprocess.run("export REPLICATE_API_TOKEN=REPLICATE_API_KEY")

#api_key = os.environ["MISTRAL_API_KEY"]
#mistral_model = "mistral-tiny"

client = OpenAI(api_key = OPENAI_SECRET_KEY)
#mistral_client = MistralClient(api_key=api_key)


note_transcript = ""
mistral_note_transcript = ""
llama_note_transcript = ""

def transcribe(audio, history_type):
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

  messages = [{"role": "system", "content": role}]
  #mistral_messages = [ChatMessage(role="system", content=role)]
  
  ################# Create Dialogue Transcript from Audio Recording and Append(via Whisper)

  audio_data, samplerate = sf.read(audio) # read audio from filepath
  #samplerate, audio_data = audio  # read audio from numpy array


  ########## Cast as float 32, normalize
  #audio_data = audio_data.astype("float32")
  #audio_data = (audio_data * 32767).astype("int16")
  #audio_data = audio_data.mean(axis=1)

  #sf.write("Audio_Files/test.mp3", audio_data, samplerate)

  ###################Code to convert .wav to .mp3
  sf.write("Audio_Files/test.wav", audio_data, samplerate, subtype='PCM_16')
  sound = AudioSegment.from_wav("Audio_Files/test.wav")
  sound.export("Audio_Files/test.mp3", format="mp3")


  ################  Send file to Whisper for Transcription
  audio_file = open("Audio_Files/test.mp3", "rb")


  max_attempts = 3
  attempt = 0
  while attempt < max_attempts:
      try:
          audio_transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
          break
      except openai.error.APIConnectionError as e:
          print(f"Attempt {attempt + 1} failed with error: {e}")
          attempt += 1
          time.sleep(3) # wait for x seconds before retrying
  else:
      print("Failed to transcribe audio after multiple attempts")

  print(audio_transcript.text)
  messages.append({"role": "user", "content": audio_transcript.text})

  #mistral_messages.append(ChatMessage(role="user", content=audio_transcript.text))



  ### Word and MB Count
  file_size = os.path.getsize("Audio_Files/test.mp3")
  mp3_megabytes = file_size / (1024 * 1024)
  mp3_megabytes = round(mp3_megabytes, 2)

  audio_transcript_words = audio_transcript.text.split() # Use when using mic input
  num_words = len(audio_transcript_words)


  #Ask OpenAI to create note transcript
  ## 1.1.1
  response = client.chat.completions.create(model="gpt-4-1106-preview", temperature=0, messages=messages)
  note_transcript = response.choices[0].message.content

  print("\n\n" + note_transcript + "\n\n")


  #Ask Mistral to create note transcript
  ## 0.0.1
  #mistral_response = mistral_client.chat(model=mistral_model, temperature=0, messages=mistral_messages)
  #mistral_note_transcript = mistral_response.choices[0].message.content
  #print (mistral_note_transcript)

  #Ask LLaMA2 (via Replicate) to create note transcript
  llama_output = replicate.run(
    #"meta/llama-2-7b-chat:f1d50bb24186c52daae319ca8366e53debdaa9e0ae7ff976e918df752732ccc4",
    "meta/llama-2-70b-chat:2d19859030ff705a87c746f7e96eea03aefb71f166725aee39692f1476566d48",
    input={
        "debug": False,
        "top_p": 1,
        "prompt": audio_transcript.text,
        "temperature": 0.01,
        "system_prompt": role,
        "max_new_tokens": 500,
        "min_new_tokens": -1
    }
  )
  llama_note_transcript = ""
  for item in llama_output:
    llama_note_transcript += item
  print(llama_note_transcript)


  return [note_transcript, llama_note_transcript, mp3_megabytes]




###################### Define Gradio Interface ######################
my_inputs = [
    gr.Audio(source="microphone", type="filepath"), #Gradio 3.48.0
    #gr.Audio(sources=["microphone"],type="numpy"), #Gradio 4.7.1
    gr.Radio(["History","H+P","Impression/Plan","Full Visit","Handover","Psych","EMS","SBAR","Meds Only"], show_label=False),
]

ui = gr.Interface(fn=transcribe,
                  inputs=my_inputs,
                  outputs=[gr.Textbox(label="OpenAI Note", show_copy_button=True),
                           gr.Textbox(label="LLaMa2 AI Note", show_copy_button=True),
                           gr.Number(label=".mp3 MB")])


ui.launch(share=False, debug=True)