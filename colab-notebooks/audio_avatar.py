from elevenlabs import clone, generate, play, set_api_key, voices, save
import openai
import streamlit as st
from subprocess import run
import os, datetime, time


# from speech_recognition import audio_transcribe
# from voice_samples import audio_sampling


openai.api_key  = "" # Paste your OpenAI API key here.
set_api_key('') # Paste your ElevenLabs API key here.



def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

def generate_audio(prompt):
    parent = '/content/RAD-NeRF/data'
    os.makedirs(parent, exist_ok=True)

    st.markdown("Generating Response from ChatGPT...")
    prompt = prompt + ' Use at most 20 words'
    chatgpt_resp = get_completion(prompt)
    # chatgpt_resp = chatgpt_resp + ' Use at most 20 words'
    print(chatgpt_resp)

    st.markdown("Convert ChatGPT response to your voice...")
    audio_response = generate(text=chatgpt_resp, model="eleven_monolingual_v1", voice='Fritz')
    # saving audio file in wav format.
    now = datetime.datetime.now()
    time_ = now.time()
    time_var = time_.isoformat().split('.')[0]
    time_var = time_var.replace(':', '_')
    audio_name = parent + '/chatgpt_response_' + time_var + '.wav'
    save(audio_response, audio_name)

    return audio_name