
# 🗣️🔄💬 Communicate with Cloneality Voice Cloner Generate your Talking Avatar in your voice and Get Response From ChatGPT
Welcome to Cloneality Voice Cloner, The VoiceClone Avatar Communicator is an innovative and immersive product that redefines the way users engage with technology. By seamlessly integrating cutting-edge technologies, including Thin-Plate Spline Motion Model, ElevenLabs for voice cloning, ChatGPT for question answering, and RAD-NeRF for lifelike avatar expressions, facial movements, and lip synchronization, this project offers an unparalleled level of personalization and interaction, offering users the ability to communicate naturally, see their emotions come to life through avatars, and retain their distinct voices in the digital realm, we pave the way for a future where technology truly understands and reflects human identity.

```
!git clone https://github.com/goodangel1012/goodangel.git
## Visit the colab-notebooks directory

%cd goodangel/RAD-NeRF
%mkdir -p pretrained
%mkdir -p data
```
Add your `API keys in chatgpt_voice.py` file for OpenAI, and ElevenLabs API for voice cloning.

### Install all the dependencies
```
# install (slow...)
!apt install libasound2-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg
!pip install -r requirements.txt
!pip install ffmpeg-python
!bash scripts/install_ext.sh
```
## Repo is under development, and not fully ready for testing...
