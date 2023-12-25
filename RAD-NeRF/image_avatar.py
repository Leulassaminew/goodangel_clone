import torch, cv2
import streamlit as st
import time, os, shutil, datetime
import imageio
from moviepy.editor import *
from PIL import Image
import imageio_ffmpeg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
import warnings
import os, subprocess
from TPSMM.demo import make_animation
from skimage import img_as_ubyte
from rad_nerf import avatar_generation

warnings.filterwarnings("ignore")

from TPSMM.demo import load_checkpoints
from audio_avatar import generate_audio
import streamlit as st
import openai, whisper
from elevenlabs import generate, play, save, set_api_key

openai_key = input('Enter your OpenAPI key... ')
eleven_key = input('Enter your ElevenLabs API key... ')
set_api_key(eleven_key) # Paste your ElevenLabs API key here.
openai.api_key = openai_key # Paste your OpenAPI Key here.

from audiorecorder import audiorecorder

subprocess.run(['gdown', '--id', '1-CKOjv_y_TzNe-dwQsjjeVxJUuyBAb5X', '--output', 'TPSMM/vox.pth.tar'])

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,  # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


def save_uploadedfile(uploadedfile):
    os.makedirs('tempDir', exist_ok=True)
    video_file_name = os.path.join("tempDir", uploadedfile.name)
    with open(video_file_name, "wb") as f:
        f.write(uploadedfile.getbuffer())

    return video_file_name

def generate_animation(voice_file, image_file):
    # edit the config
    device = torch.device('cuda:0')
    dataset_name = 'vox'  # ['vox', 'taichi', 'ted', 'mgif']
    # source_image_path = image
    # driving_video_path = driving_video_1

    try:
        voice_file_name = save_uploadedfile(voice_file)
    except:
        voice_file_name = voice_file
        
    model = whisper.load_model("base")
    result = model.transcribe(voice_file_name)
    voice_prompt = result["text"]

    chatgpt_response = get_completion(voice_prompt)
    audio_response = generate_audio(chatgpt_response)
    
    parent = './'
    now = datetime.datetime.now()
    time_ = now.time()
    time_var = time_.isoformat().split('.')[0]
    time_var = time_var.replace(':', '_')

    output_video_path = parent + '/' + time_var + '_animation' + '.mp4'

    config_path = 'TPSMM/config/vox-256.yaml'
    checkpoint_path = 'TPSMM/vox.pth.tar'

    predict_mode = 'relative'  # ['standard', 'relative', 'avd']
    find_best_frame = False  # when use the relative mode to animate a face, use 'find_best_frame=True' can get better quality result

    pixel = 256  # for vox, taichi and mgif, the resolution is 256*256
    if (dataset_name == 'ted'):  # for ted, the resolution is 384*384
        pixel = 384

    save_image = Image.open(image_file).filename
    # source_image = cv2.imread(save_image)
    source_image = imageio.imread(image_file)
    
    # print('Rendering video animation...', '\n')
    ### driving video rendering

    os.makedirs('tempDir', exist_ok=True)
    # video_input = save_uploadedfile(driving_video_1)
    talking_person = avatar_generation(audio_response)
    reader = imageio.get_reader(talking_person)

    source_image = resize(source_image, (pixel, pixel))[..., :3]

    fps = reader.get_meta_data()['fps']
    render_video = []
    try:
        for im in reader:
            render_video.append(im)
    except RuntimeError:
        pass
    reader.close()

    driving_video = [resize(frame, (pixel, pixel))[..., :3] for frame in render_video]

    inpainting, kp_detector, dense_motion_network, avd_network = load_checkpoints(config_path=config_path,
                                                                                  checkpoint_path=checkpoint_path,
                                                                                  device=device)

    if predict_mode == 'relative' and find_best_frame:
        from TPSMM.demo import find_best_frame as _find
        i = _find(source_image, driving_video, device.type == 'cpu')
        print("Best frame: " + str(i))
        driving_forward = driving_video[i:]
        driving_backward = driving_video[:(i + 1)][::-1]
        predictions_forward = make_animation(source_image, driving_forward, inpainting, kp_detector,
                                             dense_motion_network, avd_network, device=device, mode=predict_mode)
        predictions_backward = make_animation(source_image, driving_backward, inpainting, kp_detector,
                                              dense_motion_network, avd_network, device=device, mode=predict_mode)
        predictions = predictions_backward[::-1] + predictions_forward[1:]
    else:

        predictions = make_animation(source_image, driving_video, inpainting, kp_detector, dense_motion_network,
                                         avd_network, device=device, mode=predict_mode)

    # save resulting video

    imageio.mimsave(output_video_path, [img_as_ubyte(frame) for frame in predictions], fps=fps)

    # audio_file = text_to_speech(chatgpt_response)

    videoclip = VideoFileClip(output_video_path)
    audioclip = AudioFileClip(audio_response)

    new_audioclip = CompositeAudioClip([audioclip])
    videoclip.audio = new_audioclip

    audio_video_file = 'audio_video.mp4'
    videoclip.write_videofile(audio_video_file)

    return audio_video_file
