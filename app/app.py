import streamlit as st
import os
import time
from flask import Flask, Response, send_from_directory
import threading
import cv2

UPLOAD_FOLDER = "videos"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Flask app to serve streaming endpoint
flask_app = Flask(__name__)

@flask_app.route("/stream/<video_name>")
def stream_video(video_name):
    video_path = os.path.join(UPLOAD_FOLDER, video_name)
    if not os.path.exists(video_path):
        return "Video not found", 404

    def generate():
        while True:  # infinite loop
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            frame_interval = 1.0 / min(fps, 20)
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                frame = cv2.resize(frame, (640, 360))
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
                _, buffer = cv2.imencode(".jpg", frame, encode_param)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                time.sleep(frame_interval)
            cap.release()
            time.sleep(1)

    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def run_flask():
    flask_app.run(host="0.0.0.0", port=8000, debug=False, use_reloader=False)

# Start Flask server in background thread
threading.Thread(target=run_flask, daemon=True).start()

# Streamlit UI
st.title("🎥 MJPEG Video Streamer")

uploaded_file = st.file_uploader("Upload MJPEG Video", type=["mjpeg", "avi", "mp4"])
if uploaded_file is not None:
    save_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success(f"Uploaded: {uploaded_file.name}")
    stream_url = f"http://<app-name>.<app-namespace>.svc.cluster.local:8000/stream/{uploaded_file.name}"
    st.markdown(f"**Stream URL:** [{stream_url}]({stream_url})")
