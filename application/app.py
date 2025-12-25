from flask import Flask, render_template, Response, jsonify, send_from_directory
import cv2
import numpy as np
import os
import sys
import time
import threading
import smtplib
import torch
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from ultralytics import YOLO
from dotenv import load_dotenv
import os
load_dotenv() 
# --- PROJECT A IMPORTS ---
from analytics_engine import CrowdAnalytics 

# Add parent directory to path to find config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import RAW_VIDEOS_DIR

app = Flask(__name__)

# ==========================================
#   CONFIGURATION
# ==========================================
PROCESSING_SCALE = 640
SENDER_EMAIL = "harshabathala12@gmail.com"
SENDER_PASSWORD = os.getenv("EMAIL_PASS")
RECEIVER_EMAIL = "harshabathala12@gmail.com"

# --- GLOBAL STATE ---
dashboard_data = {
    "status": "INITIALIZING",
    "score": "0.00",
    "occupancy": 0,
    "recommendation": "System monitoring...",
    "alert_triggered": False,
    "thresh_warn": 30.0,
    "thresh_danger": 50.0,
    "receiver_email": RECEIVER_EMAIL
}

video_control = { "state": "play" }

# --- HARDWARE ACCELERATION ---
if torch.cuda.is_available():
    DEVICE = 0
    print(f"\n🚀 HARDWARE ACCELERATION: ENABLED ({torch.cuda.get_device_name(0)})")
else:
    DEVICE = 'cpu'
    print("\n⚠️ HARDWARE ACCELERATION: DISABLED")

# --- LOAD MODULES ---
DETECTOR_MODEL = "../yolov8n.pt" 
try:
    detector = YOLO(DETECTOR_MODEL)
    detector.to(DEVICE)
except Exception as e:
    print(f"FATAL Error: {e}")
    exit()

analytics = CrowdAnalytics()
# DEFAULT VIDEO SOURCE (Change logic later if needed)
VIDEO_SOURCE = os.path.join(RAW_VIDEOS_DIR, "sample1.mp4")

# Global Variables
prev_gray = None
frozen_frame = None
last_yielded_frame = None

# --- EMAIL ALERT SYSTEM ---
def send_email_alert(status, score, recommendation):
    def _send():
        print(f"\n📧 Sending Alert to {RECEIVER_EMAIL}...")
        try:
            clean_rec = recommendation.replace('<br>', '\n').replace('<b>', '').replace('</b>', '').split('<span')[0]
            msg = MIMEMultipart()
            msg['From'] = SENDER_EMAIL
            msg['To'] = RECEIVER_EMAIL
            msg['Subject'] = f"🚨 CROWD MONITOR ALERT: {status}"
            body = f"Status: {status}\nIndex: {score}\nAction: {clean_rec}\nTime: {time.strftime('%H:%M:%S')}"
            msg.attach(MIMEText(body, 'plain'))
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            text = msg.as_string()
            server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, text)
            server.quit()
            print("✅ Email sent successfully!\n")
        except Exception as e:
            print(f"❌ Email Failed: {e}")
    thread = threading.Thread(target=_send)
    thread.start()

# --- HELPER: FORMAT RECOMMENDATION STRING (UPDATED FOR STREET/GENERAL) ---
def format_recommendation(status, hotspot):
    if status == "NORMAL": return "Flow nominal. Monitoring active."
    
    if status == "WARNING":
        return (f"Congestion building in <b style='color:#ea580c'>{hotspot}</b>.<br>"
                f"Deploy stewards / Monitor traffic signals.")
    
    if status == "CRITICAL":
        return (f"HAZARD: BLOCKAGE in <b style='color:#dc2626'>{hotspot}</b>.<br>"
                f"<b>HALT INFLOW / REDIRECT TRAFFIC IMMEDIATELY.</b>")
    
    return "Analyzing..."

def generate_frames():
    global prev_gray, frozen_frame, dashboard_data, video_control, last_yielded_frame
    last_email_time = 0
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    ret, frame = cap.read()
    if not ret: return
    h_orig, w_orig = frame.shape[:2]
    scale = PROCESSING_SCALE / w_orig
    new_w, new_h = int(w_orig * scale), int(h_orig * scale)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Performance & Flicker Variables
    frame_counter = 0
    SKIP_FRAMES = 2
    active_boxes = [] 
    last_status_color = (0, 255, 0) 

    while True:
        if video_control["state"] == "pause" or video_control["state"] == "ended":
            if last_yielded_frame: yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + last_yielded_frame + b'\r\n')
            time.sleep(0.1); continue
        
        # --- RESET LOGIC ---
        if video_control["state"] == "replay":
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            prev_gray = None 
            active_boxes = []
            dashboard_data["alert_triggered"] = False
            dashboard_data["status"] = "INITIALIZING"
            dashboard_data["score"] = "0.00"
            video_control["state"] = "play"
            continue

        success, frame = cap.read()
        if not success:
            video_control["state"] = "ended"
            continue
        
        frame_counter += 1
        frame_small = cv2.resize(frame, (new_w, new_h))
        clean_video_feed = frame_small.copy()

        # --- LOGIC BRANCH ---
        if frame_counter % (SKIP_FRAMES + 1) == 0:
            
            # YOLO Inference
            results = detector.track(frame_small, persist=True, classes=[0], verbose=False, device=DEVICE)
            
            # Update Persistent Boxes
            active_boxes = []
            if results[0].boxes.xyxy is not None:
                active_boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

            clean_video_feed = results[0].plot(labels=False, conf=False)
            
            detections_norm = []
            if results[0].boxes.xyxyn is not None:
                 detections_norm = results[0].boxes.xyxyn.cpu().numpy().tolist()

            # Optical Flow
            gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
            flow_magnitude = 0.0
            
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                flow_magnitude = np.mean(np.abs(flow)) * 5.0
            prev_gray = gray

            # Analytics Logic
            metrics = analytics.analyze_frame(detections_norm, flow_magnitude)
            status = metrics['status']
            idx = metrics['congestion_index']
            rec_text = format_recommendation(status, metrics['hotspot'])

            # Determine Box Color
            if status == "CRITICAL": last_status_color = (0, 0, 255)   
            elif status == "WARNING": last_status_color = (0, 165, 255) 
            else: last_status_color = (0, 255, 0) 

            # Alert Cooldown Logic
            current_time = time.time()
            if status == "CRITICAL":
                dashboard_data["alert_triggered"] = False
                if (current_time - last_email_time) > 60:
                    send_email_alert(status, f"{idx}", rec_text)
                    last_email_time = current_time

            # Update Data
            dashboard_data["score"] = f"{idx}"
            dashboard_data["occupancy"] = metrics['count']
            dashboard_data["status"] = status
            dashboard_data["recommendation"] = rec_text

        # Skipped Frame Drawing
        else:
            for box in active_boxes:
                x1, y1, x2, y2 = box
                cv2.rectangle(clean_video_feed, (x1, y1), (x2, y2), last_status_color, 2)

        # Stream
        ret, buffer = cv2.imencode('.jpg', clean_video_feed)
        if ret:
            last_yielded_frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + last_yielded_frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html', benchmark=None)

@app.route('/video_feed')
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_status')
def get_status(): return jsonify(dashboard_data)

@app.route('/control/<action>')
def control_video(action):
    global video_control, dashboard_data
    if action in ['play', 'pause', 'replay']: video_control['state'] = action
    if action == 'continue': dashboard_data["alert_triggered"] = False
    return jsonify(success=True)

if __name__ == '__main__':
    print("System Starting: Crowd Density Monitoring System")
    print(f"Monitoring Source: {VIDEO_SOURCE}")
    app.run(host='0.0.0.0', port=5000, debug=False)