import os
import streamlit as st
from subprocess import run
from audio_avatar import generate_audio

def avatar_generation(user_audio):
    Person = 'engm' #@param ['obama', 'marco', 'engm', 'chris']
    Audio = 'custom' #@param ['intro', 'nvp', 'custom']
    Background = 'default' #@param ['default', 'custom']
    Pose_start = 0 #@param {type: 'integer'}
    Pose_end = 100 #@param {type: 'integer'}

    # chat_Aud = generate_audio(user_audio)
    # if Audio == 'custom':
    chat_Aud = user_audio
    # st.markdown('Getting AI features...')
    run(['python', 'nerf/asr.py', '--wav', chat_Aud, '--save_feats'])

    BG = 'bg.jpg'
    audio_npy = 'data/' + chat_Aud.split('data/')[1][:-4] + '_eo.npy'
    # audio_npy = '/content/RAD-NeRF/data/chatgpt_response_20_25_59_eo.npy'
    bg_img = 'data/' + BG
    # st.markdown('Generating Avatar')

    os.system(f"python test.py -O --torso --pose data/pose.json --data_range {Pose_start} {Pose_end} --ckpt pretrained/model.pth --aud {audio_npy} --bg_img {bg_img} --workspace trail")
    for fil in os.scandir('trail/results'):
        if fil.name.endswith('.mp4'):
            Video = 'trail/results/' + fil.name

    Video_aud = Video.replace('.mp4', '_aud.mp4')

    # concat audio
    os.system(f"ffmpeg -y -i {Video} -i {chat_Aud} -c:v copy -c:a aac {Video_aud}")
    return Video_aud