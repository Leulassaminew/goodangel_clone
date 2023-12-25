from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse, audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch, face_detection
from models import Wav2Lip
import platform
import time
import subprocess
import shlex

def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes

def face_detect(images,face_det_batch_size):
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
    flip_input=False)
    #TODO change the name of the variable batch_size
    batch_size = face_det_batch_size
    while 1:
        predictions = []
        for i in tqdm(range(0, len(images), batch_size)):
            predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
        break       
    results = []
    pady1, pady2, padx1, padx2 = [0, -1, 0, -1]
    for rect, image in zip(predictions, images):
        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)
        
        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    del detector
    return results 

def datagen(frames, mels,wav2lip_batch_size,face_det_batch_size):
    img_size = 96
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    face_det_results = face_detect(frames,face_det_batch_size) # BGR2RGB for CNN face detection
    for i, m in enumerate(mels):
        idx = i%len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (img_size, img_size))
            
        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, img_size//2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, img_size//2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        yield img_batch, mel_batch, frame_batch, coords_batch

mel_step_size = 16

def load_model(checkpoint_path = 'checkpoints/wav2lip_gan.pth'):
    model = Wav2Lip()
    checkpoint = torch.load(checkpoint_path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    model = model.to('cuda:0')
    return model.eval()

model  = load_model()
torch.backends.cudnn.benchmark = True

def run_inference(face,audio_source,model = model,wav2lip_batch_size=1,face_det_batch_size=1,resize_factor=1,stream = False):
    """
    generates the avatar if stream is set to truth the avatar gets streamed and if false gets saved on the folder results.
    face: is a video or a picture of the person that will be used for the avatar (if it's a video the face of the person must be shown at all times)
    audio_source: is the voice of the avatar, it must be a wav file.
    face_det_batch_size = batch size of the face detection algorithm it's recommended to be set at 1, no matter the hardware.
    wav2lip_batch_size = batch size of the wav2lip algorithm it's recommended to be set at 1, no matter the hardware.
    resize_factor = downsampling factor. 1 equals to keeping the original resolution, increasing the number reduces the quality of the avatar
    but improves the speed of the function
    """
    face_name = face.split('/')[1].split('.')[0]
    output_name = f'{face_name}_{720 // resize_factor}p'
    if not face.split('.')[-1] in ['jpg', 'png', 'jpeg']:
        print('video')
        video_stream = cv2.VideoCapture(face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            if resize_factor > 1:
                frame = cv2.resize(frame, (frame.shape[1]//resize_factor, frame.shape[0]//resize_factor))

            y1, y2, x1, x2 = [0, -1, 0, -1]
            if x2 == -1: x2 = frame.shape[1]
            if y2 == -1: y2 = frame.shape[0]

            frame = frame[y1:y2, x1:x2]

            full_frames.append(frame)
    else:
        full_frames = [cv2.imread(face)]
        fps = 25

    wav = audio.load_wav(audio_source, 16000)
    mel = audio.melspectrogram(wav)

    
    mel_chunks = []
    mel_idx_multiplier = 80./fps 
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1

    print("Length of mel chunks: {}".format(len(mel_chunks)))

    full_frames = full_frames[:len(mel_chunks)]

    batch_size = wav2lip_batch_size
    gen = datagen(full_frames.copy(), mel_chunks,face_det_batch_size = face_det_batch_size,wav2lip_batch_size = wav2lip_batch_size)

    for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, 
                                            total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
        if i == 0:
            frame_h, frame_w = full_frames[0].shape[:-1]
            out = cv2.VideoWriter('temp/result.avi', 
                                    cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

        img_batch = torch.cuda.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2)))
        mel_batch = torch.cuda.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2)))

        with torch.no_grad():
            pred = model(mel_batch, img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
        
        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

            f[y1:y2, x1:x2] = p
            out.write(f)

    out.release()
    audio_source_quoted = shlex.quote(audio_source)
    command = f'ffmpeg -y -i {audio_source_quoted} -i temp/result.avi -c:v h264_nvenc -preset fast -c:a aac -strict -2 -qp 18 -threads 4 results/{output_name}.mp4'
    if not stream:


      # Execute the FFmpeg command
      try:
          subprocess.run(command, shell=True, check=True)
          print("FFmpeg operation completed successfully.")
      except subprocess.CalledProcessError as e:
          print(f"Error: {e}")
    else:
      # FFmpeg command with multithreading, CUDA hardware acceleration, and qp option, streaming the output

      # Execute the FFmpeg command and capture the output
      try:
          process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

          # Read and process the output stream (stdout and stderr) if needed
          output, error = process.communicate()

          # Your processing logic for the output stream goes here

          print("FFmpeg operation completed successfully.")
      except subprocess.CalledProcessError as e:
          print(f"Error: {e}")
