from elevenlabs import clone, generate, play, set_api_key, voices, save
import os, datetime, time, shutil

set_api_key('') # Paste ElevenLabs API

# using pre-trained voices
name = 'Bella'
prompt = input('Enter what you want to speak.... ')
audio_response = generate(text=prompt, model="eleven_monolingual_v1", voice=name)

# saving audio file in wav format.

now = datetime.datetime.now()
time_ = now.time()
time_var = time_.isoformat().split('.')[0]
time_var = time_var.replace(':', '_')

audio_name = parent + '/chatgpt_response_' + time_var + '.wav'
save(audio_response, audio_name)

# do voice cloning

voice = clone(
    name="Alex",
    description="An old American male voice with a slight hoarseness in his throat. Perfect for news", # Optional
    files=["./sample_0.mp3", "./sample_1.mp3", "./sample_2.mp3"],
)

audio = generate(text="Hi! I'm a cloned voice!", voice=voice)

play(audio, notebook=True)
