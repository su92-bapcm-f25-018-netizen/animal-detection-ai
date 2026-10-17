import os
import json
import zipfile
import gradio as gr
import numpy as np
import tensorflow as tf

from PIL import Image, ImageOps

# Extract the downloaded model ZIP
os.makedirs("animal_model", exist_ok=True)

with zipfile.ZipFile("animal_model.zip", "r") as z:
    z.extractall("animal_model")

model = tf.keras.models.load_model(
    "animal_model/animal_model.keras"
)

with open("animal_model/class_names.json") as f:
    classes = json.load(f)

def predict(image):
    if image is None:
        return {"Please upload a photo": 1.0}

    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize((224, 224))

    x = np.array(image, dtype=np.float32)
    x = np.expand_dims(x, axis=0)

    predictions = model.predict(x, verbose=0)[0]

    return {
        classes[i]: float(predictions[i])
        for i in range(len(classes))
    }

app = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Animal Photo"),
    outputs=gr.Label(num_top_classes=5),
    title="Animal Detection AI",
    description="Upload a cat, dog, lion, tiger or elephant photo."
)

app.launch()
