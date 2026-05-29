import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="PPEGuard AI", layout="centered")

st.title("🦺 PPEGuard AI Scanner")
st.write("Take a photo to check PPE compliance.")

model = tf.keras.models.load_model("keras_model.h5")

labels = [
"Complete PPE",
"Incomplete PPE",
"Not wearing PPE"
]

image_file = st.camera_input("Take a PPE photo")

if image_file is not None:
image = Image.open(image_file).convert("RGB")

```
st.image(image, caption="Captured Image")

image = image.resize((224, 224))

image_array = np.asarray(image)

image_array = image_array.astype(np.float32) / 255.0

image_array = np.expand_dims(image_array, axis=0)

prediction = model.predict(image_array)

index = np.argmax(prediction)

confidence = float(prediction[0][index]) * 100

result = labels[index]

st.subheader(f"Prediction: {result}")
st.write(f"Confidence: {confidence:.2f}%")

if result == "Complete PPE" and confidence >= 80:
    st.success("✅ ACCESS GRANTED")
else:
    st.error("❌ ACCESS DENIED")
```
