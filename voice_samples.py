import time, os, shutil
import torchaudio, datetime
from pydub import AudioSegment
from pydub.silence import split_on_silence

def audio_sampling(name, audio):

    voice_samples = []
    parent = './' + name + '_samples'

    os.makedirs(parent, exist_ok=True)

    # saving audio file in wav format.
    now = datetime.datetime.now()
    time_ = now.time()
    time_var = time_.isoformat().split('.')[0]
    time_var = time_var.replace(':', '_')
    audio_name = parent + '/' + time_var + '.wav'


    audio_read, sr = torchaudio.load(audio)
    torchaudio.save(audio_name, audio_read, sample_rate=sr)

    # reading audio file
    sound = AudioSegment.from_wav(audio_name)

    # spliting audio file on silence for making audio samples
    audio_chunks = split_on_silence(sound, min_silence_len=700, silence_thresh=-50)

    #loop is used to iterate over the output list
    for i, chunk in enumerate(audio_chunks):
        save_dir = parent + "audio_chunks"
        os.makedirs(save_dir, exist_ok=True)

        output_file = save_dir + "/chunk{0}.wav".format(i)
        print("Exporting file", output_file)
        chunk.export(output_file, format="wav")

    for aud_chunk in os.scandir(save_dir):
        voice_samples.append(save_dir + '/' + aud_chunk.name)

    if len(voice_samples) > 3:
        voice_samples_3 = voice_samples[:3]

    else:
        voice_samples_3 = voice_samples

    return voice_samples_3
