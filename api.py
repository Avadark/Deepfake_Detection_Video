from flask import Flask, render_template, request, redirect, url_for
import gradio as gr
import cv2
import numpy as np
import tensorflow as tf
from facenet_pytorch import MTCNN
import moviepy.editor as mp
from PIL import Image
import os
import zipfile
import json
import base64
from tensorflow_addons.optimizers import RectifiedAdam  # Import the RectifiedAdam optimizer
from keras.utils import get_custom_objects  # Use tensorflow.keras.utils instead
get_custom_objects().update({"RectifiedAdam": RectifiedAdam})

app = Flask(__name__)

# Load face detector
mtcnn = MTCNN(margin=14, keep_all=True, factor=0.7, device='cpu')

# DetectionPipeline class
class DetectionPipeline:
    def __init__(self, detector, n_frames=None, batch_size=60, resize=None):
        self.detector = detector
        self.n_frames = n_frames
        self.batch_size = batch_size
        self.resize = resize

    def __call__(self, filename):
        v_cap = cv2.VideoCapture(filename)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.n_frames is None:
            sample = np.arange(0, v_len)
        else:
            sample = np.linspace(0, v_len - 1, self.n_frames).astype(int)

        faces = []
        frames = []

        for j in range(v_len):
            success = v_cap.grab()
            if j in sample:
                success, frame = v_cap.retrieve()
                if not success:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if self.resize is not None:
                    frame = frame.resize([int(d * self.resize) for d in frame.size])

                frames.append(frame)

                if len(frames) % self.batch_size == 0 or j == sample[-1]:
                    boxes, probs = self.detector.detect(frames)

                    for i in range(len(frames)):
                        if boxes[i] is None:
                            faces.append(face2)
                            continue

                        box = boxes[i][0].astype(int) 
                        frame = frames[i]
                        face = frame[box[1]:box[3], box[0]:box[2]]

                        if not face.any():
                            faces.append(face2)
                            continue

                        face2 = cv2.resize(face, (224, 224))
                        faces.append(face2)

                    frames = []

        v_cap.release()

        return faces

detection_pipeline = DetectionPipeline(detector=mtcnn, n_frames=20, batch_size=60)

model = tf.keras.models.load_model("./Detecto-DeepFake_Video_Detector/p1")

def deepfakespredict(input_video):
    faces = detection_pipeline(input_video)
    total = 0
    real = 0
    fake = 0

    for face in faces:
        face2 = face / 255
        pred = model.predict(np.expand_dims(face2, axis=0))[0]
        total += 1
        pred2 = pred[1]

        if pred2 > 0.5:
            fake += 1
        else:
            real += 1

    fake_ratio = fake / total
    text = ""
    text2 = f"Deepfakes Confidence: {fake_ratio * 100:.2f}%"

    if fake_ratio >= 0.5:
        text = "The video is FAKE."
    else:
        text = "The video is REAL."

    face_frames = []
    
    for face in faces:
        face_frame = Image.fromarray(face.astype('uint8'), 'RGB')
        face_frames.append(face_frame)
        
    face_frames[0].save('results.gif', save_all=True, append_images=face_frames[1:], duration=250, loop=100)
    clip = mp.VideoFileClip("results.gif")
    clip.write_videofile("video.mp4")

    return text, text2, "video.mp4"

iface = gr.Interface(
    fn=deepfakespredict,
    inputs=gr.inputs.Video(type="mp4"),
    outputs=[
        gr.outputs.Text(label="Detection Result"),
        gr.outputs.Text(label="Confidence"),
        gr.outputs.File(label="Result Video")
    ],
    live=True,
    title="EfficientNetV2 Deepfakes Video Detector",
    description="This is a demo implementation of EfficientNetV2 Deepfakes Image Detector ",
    examples=[
        [open('./Detecto-DeepFake_Video_Detector/Video1-fake-1-ff.mp4', 'rb')],
        [open('./Detecto-DeepFake_Video_Detector/Video6-real-1-ff.mp4', 'rb')],
        [open('./Detecto-DeepFake_Video_Detector/Video3-fake-3-ff.mp4', 'rb')],
        [open('./Detecto-DeepFake_Video_Detector/Video8-real-3-ff.mp4', 'rb')],
        [open('./Detecto-DeepFake_Video_Detector/real-1.mp4', 'rb')],
        [open('./Detecto-DeepFake_Video_Detector/fake-1.mp4', 'rb')]
    ]
)

@app.route('/')
def index():
    iface.launch(share=True)
    return iface.ui()

if __name__ == '__main__':
    app.run(debug=True)
