
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import mediapipe as mp

# Set title
st.title("🖐️ Real-time Hand Gesture Detection Demo")

# MediaPipe Hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class HandGestureDetector(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert frame to numpy array
        image = frame.to_ndarray(format="bgr24")

        # Flip for natural view
        image = cv2.flip(image, 1)

        # Convert to RGB for MediaPipe
        results = self.hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(image, format="bgr24")

# Start the webcam streamer
webrtc_streamer(key="hand-gesture-demo", video_processor_factory=HandGestureDetector)
