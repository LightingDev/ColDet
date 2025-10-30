import gradio as gr
import numpy as np
import cv2
from keras.models import load_model

# Load model and labels
model = load_model("coldet.h5", compile=False)
class_names = open("labels.txt").read().splitlines()

def predict(img):
    # Convert to 224x224 and normalize
    img_resized = cv2.resize(img, (224, 224))
    img_array = (np.array(img_resized, dtype=np.float32).reshape(1, 224, 224, 3) / 127.5) - 1
    
    # Predict
    pred = model.predict(img_array)
    idx = np.argmax(pred)
    return f"{class_names[idx]} ({pred[0][idx]*100:.2f}%)"

# Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload or Webcam"),
    outputs="text"
)

iface.launch()
