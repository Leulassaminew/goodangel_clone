import whisperx
import torch
import time
import requests
import openai
import json
from pydub import AudioSegment
from concurrent.futures import ProcessPoolExecutor

openai.api_key  = 
eleven_labels_api_key = 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
language = 'en'
try:
    model_x = whisperx.load_model("tiny.en", device, compute_type='float16',language = language)
except:
    model_x = whisperx.load_model("tiny.en", device, compute_type='int8',language = language)

def generate_transcript(audio):
    """
    the first time it runs takes a few seconds to run, then after that takes faction of seconds.
    it could be a good improvement point, when it comes to build the optimizing the code on the the backend of the webapp
    """
    audio = whisperx.load_audio(audio)
    try:
        return model_x.transcribe(audio, batch_size=512)["segments"][0]['text']
    except:
        return model_x.transcribe(audio, batch_size=64)["segments"][0]['text']

def mp3_to_wav(name_file):
    audio = AudioSegment.from_mp3(f'{name_file}.mp3')
    answer = 'data/answer.wav'
    audio.export(answer, format="wav")

def generate_audio(text,voice_id,model_id ='eleven_monolingual_v1'):
    CHUNK_SIZE = 400
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": eleven_labels_api_key
        }
    data = {
        "text": text,
        "model_id": model_id,
        'optimize_streaming_latency': 4,
        "voice_settings": {
            "stability": 0,
            "similarity_boost": 0
            }
        }
    response = requests.post(url, json=data, headers=headers)
    with open('output_new.mp3', 'wb') as f:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                        f.write(chunk)
    with ProcessPoolExecutor() as executor:
        executor.submit(mp3_to_wav, 'output_new')

def get_voice_id(voice_name):
    url = "https://api.elevenlabs.io/v1/voices"
    headers = {
        "Accept": "application/json",
        "xi-api-key": eleven_labels_api_key
    }
    response = requests.get(url, headers=headers)
    return [voice['voice_id'] for voice in json.loads(response.text)['voices'] if voice['name'] == voice_name][0]

def transcript_2_audio_response(transcript,voice_name):
    voice_id = get_voice_id(voice_name)
    transcript = f'{transcript} in 20 words at max '
    messages = [{"role": "user", "content": transcript}]
    model = 'gpt-3.5-turbo'
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
        )
    text = response['choices'][0]['message']['content']
    generate_audio(text = text,voice_id = voice_id)


def question2answer(audio,voice_name):
    """
    returns a generated voice answering the generated response of the audio question.
    audio: is the direction where the audio file containing the question is located.
    voice_name: the name of the AI generated voice.
    """
    transcript = generate_transcript(audio)
    transcript_2_audio_response(transcript,voice_name)
