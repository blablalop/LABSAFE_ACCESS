import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(page_title="LABSAFE ACCESS", layout="centered")

st.title("🧪 LABSAFE ACCESS")
st.subheader("Live PPE Scanner")

model = tf.keras.models.load_model("keras_model.h5")

labels = [
    "Complete PPE",
    "Incomplete PPE",
    "Not wearing PPE"
]

class PPEProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224))
        image_array = resized.astype(np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        prediction = model.predict(image_array, verbose=0)
        index = np.argmax(prediction)
        confidence = float(prediction[0][index]) * 100
        result = labels[index]

        if result == "Complete PPE" and confidence >= 80:
            status = "ACCESS GRANTED"
            color = (0, 255, 0)
        else:
            status = "ACCESS DENIED"
            color = (0, 0, 255)

        cv2.putText(img, status, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        cv2.putText(img, f"{result}: {confidence:.2f}%", (30, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="ppe-live-scanner",
    video_processor_factory=PPEProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    }
)
