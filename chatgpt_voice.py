from elevenlabs import clone, generate, play, set_api_key, voices, save
import openai
import whisper
import os, datetime, time, shutil

from voice_samples import audio_sampling

openai.api_key  = "" # Paste OpenAI API key here
set_api_key('') # Paste ElevenLabs API


# creating ChatGPT function for voice conversation
def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


def voice_chatgpt(name, audio):

    parent = './' + name
    os.makedirs(parent, exist_ok=True)
    # if type(audio) is not list():

    voice_name = 'Cloned_' + name # 'Cloned_Fahad'
    cloning_name = ['Alex', 'Bella', 'Cloned_Fahad', 'Fahad']

    # converting voice input into text for chatgpt prompt.
    model = whisper.load_model("base")
    response_ = model.transcribe(audio)['text']
    prompt = get_completion(response_)

    if voice_name in cloning_name:
        audio_response = generate(text=prompt, model="eleven_monolingual_v1", voice=voice_name)
        # saving audio file in wav format.
        now = datetime.datetime.now()
        time_ = now.time()
        time_var = time_.isoformat().split('.')[0]
        time_var = time_var.replace(':', '_')
        audio_name = parent + '/chatgpt_response_' + time_var + '.wav'
        save(audio_response, audio_name)
        return audio_name


    elif name in cloning_name:
        audio_response = generate(text=prompt, model="eleven_monolingual_v1", voice=name)
        # saving audio file in wav format.
        now = datetime.datetime.now()
        time_ = now.time()
        time_var = time_.isoformat().split('.')[0]
        time_var = time_var.replace(':', '_')
        audio_name = parent + '/chatgpt_response_' + time_var + '.wav'
        save(audio_response, audio_name)

        return audio_name

    else:
        voice_samples_3 = audio_sampling(name, audio)
        cloning_name.append(voice_name)
        # return voice_samples_3, cloning_name
        voice = clone(name=voice_name, files=voice_samples_3)
        audio_response = generate(text=prompt, model="eleven_monolingual_v1", voice=voice_name)
        # saving audio file in wav format.
        now = datetime.datetime.now()
        time_ = now.time()
        time_var = time_.isoformat().split('.')[0]
        time_var = time_var.replace(':', '_')
        audio_name = parent + '/chatgpt_response_' + time_var + '.wav'
        save(audio_response, audio_name)
        
        cloning_name.append(voice_name) ## appending your voice

        return audio_name
