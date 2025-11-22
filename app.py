# app.py
# Updated: Removed Mediapipe; ArUco-only; fitscore computed from DB sizes with weights:
# eye: 0.3285, bridge: 0.3613, temple: 0.3102
# Added: /fit_test route to auto-open camera, capture exactly 3 images, reuse FitScore
#        for same QR, FitScore=0 if no QR, and save best-fit image to static/best_fit.jpg

# Standard library
import os
import csv
import pickle
import secrets
import traceback
import urllib.request
import calendar
from threading import Timer
from datetime import datetime, date, timedelta, time
import shutil

import bcrypt
# Third-party
import cv2
import numpy as np
import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, render_template, request, json, jsonify, Response, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash
from dotenv import load_dotenv
from supabase import create_client
from user_agents import parse as parse_user_agent
import sib_api_v3_sdk
from sib_api_v3_sdk.rest import ApiException

# Local
from ml_utils import extract_features

# Load environment
load_dotenv()
app = Flask('name')

# -------------------------------------------------------------------------------------------
# Database & Supabase connections
# -------------------------------------------------------------------------------------------
# Connect to Supabase PostgreSQL (Postgres via psycopg2) - using env variables from .env
conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    port=os.getenv("DB_PORT")
)
cursor = conn.cursor(cursor_factory=RealDictCursor)

# Supabase client
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Flask secret
app.secret_key = os.environ.get('SECRET_KEY', 'fallback_dev_key')

# -------------------------------------------------------------------------------------------
# Camera / Picamera support
# -------------------------------------------------------------------------------------------
try:
    from picamera2 import Picamera2
    pi_camera_supported = True
except Exception:
    pi_camera_supported = False

camera = None
camera_timer = None
camera_active = False
using_pi_camera = False

def start_camera():
    global camera, camera_active, using_pi_camera

    if camera_active:
        return  # already running

    print("[CAMERA STATUS] Starting camera...")

    if pi_camera_supported:
        try:
            picam2 = Picamera2()
            config = picam2.create_preview_configuration()
            picam2.configure(config)
            picam2.start()
            camera = picam2
            using_pi_camera = True
            camera_active = True
            print("[CAMERA STATUS] ✅ Raspberry Pi Camera is ON")
            return
        except Exception as e:
            print(f"[❌] Pi Camera failed: {e}")

    # Fallback to USB camera
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        camera = cap
        using_pi_camera = False
        camera_active = True
        print("[CAMERA STATUS] ✅ USB camera is ON")
    else:
        camera = None
        print("[CAMERA STATUS] ❌ No camera available")

def stop_camera():
    global camera, camera_active, using_pi_camera
    if not camera_active:
        return

    if using_pi_camera:
        camera.stop()
    else:
        camera.release()

    camera = None
    camera_active = False
    print("[CAMERA STATUS] 🔌 Camera is OFF")

def reset_camera_timer():
    global camera_timer
    if camera_timer:
        camera_timer.cancel()
    camera_timer = Timer(2.0, stop_camera)  # Auto-stop after 2s idle
    camera_timer.start()


# Load face shape classifier (pickle)
with open("face_shape_model.pkl", "rb") as f:
    model = pickle.load(f)

# -------------------------------------------------------------------------------------------
# ArUco setup (used for QR / marker detection)
# -------------------------------------------------------------------------------------------
# Use a 4x4 dictionary with 100 entries as in original code
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# -------------------------------------------------------------------------------------------
# Helper: Detect ArUco markers in an image
# -------------------------------------------------------------------------------------------
def detect_aruco(image_path, min_marker_area=50):
    """
    Read image_path and detect aruco IDs. Returns list of detected marker ints.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use detector (OpenCV ArUco)
    corners, ids, _ = detector.detectMarkers(gray)

    detected = []
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            pts = corners[i][0].astype(int)
            area = cv2.contourArea(pts)
            if area > min_marker_area:
                detected.append(int(marker_id))

    print(f"[INFO] Detected ArUco IDs in {image_path}: {detected}")
    return detected

# -------------------------------------------------------------------------------------------
# DB helpers: map marker ID -> eyeglass_frame_id and fetch frame details
# -------------------------------------------------------------------------------------------
def get_eyeglass_frame_id(marker_id: int):
    """Fetch eyeglass_frame_id from aruco_mapping table given a numeric marker_id."""
    try:
        cursor.execute(
            "SELECT eyeglass_frame_id FROM aruco_mapping WHERE marker_id = %s",
            (marker_id,)
        )
        result = cursor.fetchone()
        if result:
            return result['eyeglass_frame_id']
        else:
            print(f"[❌] No eyeglass_frame_id found for marker_id {marker_id}")
            return None
    except Exception as e:
        print("[ERROR] Query failed:", e)
        return None

def fetch_frame_details(eyeglass_frame_id: str):
    """Fetch frame metadata (shape and brand) from eyeglassframeinventory."""
    try:
        cursor.execute(
            """
            SELECT eyeglass_frame_id, frame_brand, model_number,
                   frame_color, frame_shape, price, image_url
            FROM eyeglassframeinventory
            WHERE eyeglass_frame_id = %s
            """,
            (eyeglass_frame_id,)
        )
        result = cursor.fetchone()
        if result:
            return result
        else:
            print(f"[❌] No frame found for eyeglass_frame_id {eyeglass_frame_id}")
            return None
    except Exception as e:
        print("[ERROR] Query failed:", e)
        return None

# -------------------------------------------------------------------------------------------
# New: Fetch sizes (eye, bridge, temple) from Supabase or Postgres for a given eyeglass_frame_id.
# If the DB columns use different names, adjust the keys used below.
# -------------------------------------------------------------------------------------------
def fetch_sizes_from_supabase(eyeglass_frame_id):
    """
    Try to fetch 'eye', 'bridge', 'temple' fields from Supabase for a given eyeglass_frame_id.
    Returns dict with keys 'eye','bridge','temple' or None if not found.
    """
    try:
        q = supabase.table("eyeglassframeinventory").select("*").eq("eyeglass_frame_id", eyeglass_frame_id).execute()
        data = getattr(q, "data", None)
        if data and len(data) > 0:
            rec = data[0]
            eye = rec.get("eye") or rec.get("eye_size") or rec.get("eye_width") or rec.get("eye_w")
            bridge = rec.get("bridge") or rec.get("bridge_size") or rec.get("bridge_width")
            temple = rec.get("temple") or rec.get("temple_size") or rec.get("temple_length")
            if eye is None and bridge is None and temple is None:
                return None
            return {"eye": float(eye or 0), "bridge": float(bridge or 0), "temple": float(temple or 0)}
        return None
    except Exception as e:
        print("[ERROR] fetch_sizes_from_supabase:", e)
        return None

def fetch_sizes_from_postgres(eyeglass_frame_id):
    """
    Fallback Postgres query for sizes. Modify column names if necessary.
    """
    try:
        cursor.execute(
            """
            SELECT eye, bridge, temple
            FROM eyeglassframeinventory
            WHERE eyeglass_frame_id = %s
            """, (eyeglass_frame_id,)
        )
        row = cursor.fetchone()
        if row:
            eye = row.get("eye") or row.get("eye_size") or row.get("eye_width")
            bridge = row.get("bridge") or row.get("bridge_size") or row.get("bridge_width")
            temple = row.get("temple") or row.get("temple_size") or row.get("temple_length")
            return {"eye": float(eye or 0), "bridge": float(bridge or 0), "temple": float(temple or 0)}
        return None
    except Exception as e:
        print("[ERROR] fetch_sizes_from_postgres:", e)
        return None

# -------------------------------------------------------------------------------------------
# FITSCORE COMPUTATION
# -------------------------------------------------------------------------------------------
def compute_fitscore(eye, bridge, temple):
    """
    Compute weighted FitScore using given weights:
      eye: 0.3285, bridge: 0.3613, temple: 0.3102
    """
    try:
        return (float(eye) * 0.3285) + (float(bridge) * 0.3613) + (float(temple) * 0.3102)
    except Exception as e:
        print("[ERROR] compute_fitscore:", e)
        return 0.0



# -------------------------------------------------------------------------------------------
# Crop eyeglass region using Haar cascade (unchanged)
# -------------------------------------------------------------------------------------------
def crop_eyeglass_region(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        eyeglass_y = y + int(h * 0.25)
        eyeglass_h = int(h * 0.3)
        eyeglass_x = x
        eyeglass_w = w

        cropped = img[eyeglass_y:eyeglass_y + eyeglass_h,
                      eyeglass_x:eyeglass_x + eyeglass_w]

        cropped_dir = os.path.join("static", "Cropped")
        os.makedirs(cropped_dir, exist_ok=True)

        base = os.path.basename(image_path)
        name, ext = os.path.splitext(base)
        save_path = os.path.join(cropped_dir, f"cropped_{name}.jpg")
        cv2.imwrite(save_path, cropped)

        print(f"[✅] Cropped eyeglass region saved to {save_path}")
        return save_path   # return the new file path

    return None

# -------------------------------------------------------------------------------------------
# gen_frames: streaming endpoint for camera view (unchanged)
# -------------------------------------------------------------------------------------------
def gen_frames():
    global camera, using_pi_camera

    # --- Ensure camera is started ---
    start_camera()

    if camera is None:
        print("[CAMERA STATUS] ❌ No camera detected.")
        return

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    while True:
        reset_camera_timer()

        try:
            if using_pi_camera:
                # Raspberry Pi camera
                frame = camera.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                # USB webcam
                ret, frame = camera.read()
                if not ret:
                    print("[ERROR] Failed to read frame from USB camera")
                    break

            frame = cv2.flip(frame, 1)  # Mirror image

        except Exception as e:
            print(f"[ERROR] Failed to capture frame: {e}")
            break

        # --- Frame dimensions & oval parameters ---
        frame_height, frame_width = frame.shape[:2]
        center_x, center_y = frame_width // 2, frame_height // 2
        axis_x, axis_y = 110, 140  # Ellipse horizontal/vertical radius

        # --- Convert to grayscale for face detection ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        alignment_ok = False

        # --- Detect face and check alignment ---
        if len(faces) > 0:
            # Pick the largest detected face
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
            (x, y, w, h) = faces[0]
            face_cx = x + w // 2
            face_cy = y + h // 2

            # Check if face center is inside the oval
            ellipse_eq = ((face_cx - center_x) ** 2) / (axis_x ** 2) + ((face_cy - center_y) ** 2) / (axis_y ** 2)
            if ellipse_eq <= 1.0:
                alignment_ok = True

            # Draw face rectangle and center dot
            cv2.rectangle(frame, (x, y), (x + w, y + h), (150, 150, 150), 1)
            cv2.circle(frame, (face_cx, face_cy), 3, (255, 0, 0), -1)

        # --- Draw oval alignment guide ---
        oval_color = (0, 255, 0) if alignment_ok else (0, 0, 255)
        cv2.ellipse(frame, (center_x, center_y), (axis_x, axis_y), 0, 0, 360, oval_color, 2)

        # --- Display alignment message ---
        message = "✅ Aligned - You may take a photo" if alignment_ok else "🔴 Please align your face in the oval"
        text_color = (0, 255, 0) if alignment_ok else (0, 0, 255)
        cv2.putText(frame, message, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

        # --- Encode frame to JPEG for streaming ---
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        # --- Yield frame stream ---
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# -------------------------------------------------------------------------------------------
# Routes (kept intact; small edits so they call run_fit_test defined above)
# -------------------------------------------------------------------------------------------

# INDEX — Load patient dropdown
@app.route('/')
def index():
    cursor.execute("SELECT patient_id, patient_fname, patient_minitial, patient_lname FROM patient;")
    rows = cursor.fetchall()

    # Convert results to a simple list for dropdown
    patients = [
        {
            "patient_id": r["patient_id"],
            "full_name": f"{r['patient_fname']} {r['patient_minitial']}. {r['patient_lname']}"
        }
        for r in rows
    ]

    page = request.args.get("page", default=1, type=int)

    return render_template("choose_frame.html", patients=patients)


# -------------------------------------------------------------------------------------------
# AJAX route — check patient status from Supabase (kept intact)
# -------------------------------------------------------------------------------------------
@app.route('/check_patient_status', methods=['POST'])
def check_patient_status():
    data = request.get_json()
    patient_id = data.get('patient_id')

    if not patient_id:
        return jsonify({'error': 'Missing patient_id'}), 400

    try:
        if str(patient_id).isdigit():
            patient_id = int(patient_id)

        # --- Fetch patient info ---
        patient_query = supabase.table("patient").select("age").eq("patient_id", patient_id).execute()
        patient_data = getattr(patient_query, "data", []) or getattr(patient_query, "json", {}).get("data", [])

        if not patient_data:
            return jsonify({'error': 'Patient not found'}), 404

        age = patient_data[0].get('age')
        if age is None:
            return jsonify({'error': 'Age not found in patient record'}), 404

        # --- Default result ---
        result = {
            "is_child": age < 18,
            "is_acidic": False,
            "is_high_prescription": False,
            "distance_sph_od": None,
            "distance_sph_os": None,
            "recommended_material": None,
            "recommended_frame": None,
            "recommendation_reason": None,
            "recommended_frames": []
        }

        # --- Fetch prescription details ---
        pres_query = supabase.table("prescription").select(
            "condition, distance_sph_od, distance_sph_os"
        ).eq("patient_id", patient_id).execute()
        pres_data = getattr(pres_query, "data", []) or getattr(pres_query, "json", {}).get("data", [])

        condition_found = None

        for pres in pres_data:
            condition = (pres.get("condition") or "").strip().lower()

            # 🧴 Acidic Skin
            if condition == "acidic":
                result["is_acidic"] = True
                condition_found = "acidic"
                result["recommended_material"] = "Titanium or Cellulose Acetate"
                result["recommended_frame"] = "Full Rim, Half Rim, or Rimless"
                result["recommendation_reason"] = (
                    "Titanium and Cellulose Acetate are hypoallergenic and corrosion-resistant, "
                    "ideal for clients with acidic skin. Rimless titanium frames may also be used for a lighter fit."
                )

            # 👓 High Prescription (Thick Lenses)
            elif condition in ["high prescription", "thick lenses"]:
                result["is_high_prescription"] = True
                condition_found = "high prescription"
                result["distance_sph_od"] = pres.get("distance_sph_od")
                result["distance_sph_os"] = pres.get("distance_sph_os")
                result["recommended_material"] = "Acetate, thermoplastic, Nylon, or Polyamide (Rigid Plastic Frames)"
                result["recommended_frame"] = "Full Rim"
                result["recommendation_reason"] = (
                    "Only rigid full-rim plastic frames such as Acetate, thermoplastic, Nylon, or Polyamide are recommended. "
                    "They provide structural support and conceal thick lens edges effectively."
                )

        # 🧒 Active Lifestyle / Children
        if result["is_child"] and not condition_found:
            condition_found = "active"
            result["recommended_material"] = "Polycarbonate or Trivex"
            result["recommended_frame"] = "Full Rim"
            result["recommendation_reason"] = (
                "Polycarbonate and Trivex materials are highly impact-resistant, flexible, and lightweight — "
                "ideal for children or active individuals."
            )

        # --- Define material and frame filters ---
        material_filter = []
        frame_filter = []

        if condition_found == "acidic":
            # ✅ Rimless allowed for acidic skin
            material_filter = ["titanium", "acetate"]
            frame_filter = ["full rim", "half rim", "rim less"]

        elif condition_found == "high prescription":
            # ✅ Only rigid plastics for thick lenses
            material_filter = ["thermoplastic", "acetate", "nylon", "polyamide"]
            frame_filter = ["full rim"]

        elif condition_found == "active":
            # ✅ Impact-resistant for active users
            material_filter = ["polycarbonate", "trivex"]
            frame_filter = ["full rim"]

        else:
            # ✅ Normal condition — all types allowed
            material_filter = []
            frame_filter = ["full rim", "half rim", "rim less"]

        # --- Fetch eyeglass frame data ---
        frame_query = supabase.table("eyeglassframeinventory").select(
            "eyeglass_frame_id, frame_brand, model_number, frame_color, "
            "frame_shape, price, image_url, material, frame_type"
        ).execute()

        frame_data = getattr(frame_query, "data", []) or getattr(frame_query, "json", {}).get("data", [])
        selected_frames = []

        for frame_record in frame_data:
            material = (frame_record.get("material") or "").strip().lower()
            ftype = (frame_record.get("frame_type") or "").strip().lower()

            # Apply filters
            if (not material_filter or material in material_filter) and \
               (not frame_filter or ftype in frame_filter):
                selected_frames.append({
                    "eyeglass_frame_id": frame_record.get("eyeglass_frame_id"),
                    "brand": frame_record.get("frame_brand", "Unknown"),
                    "model_number": frame_record.get("model_number", "N/A"),
                    "color": frame_record.get("frame_color", "N/A"),
                    "shape": frame_record.get("frame_shape", "N/A"),
                    "price": frame_record.get("price", 0.0),
                    "image_url": frame_record.get("image_url", ""),
                    "material": material,
                    "frame_type": ftype
                })

        result["recommended_frames"] = selected_frames

        # --- Default (Normal Condition) ---
        if not condition_found:
            result["recommended_material"] = "Any Material"
            result["recommended_frame"] = "Any Frame Type"
            result["recommendation_reason"] = (
                "No special condition detected — any frame style and material may be recommended."
            )

        return jsonify(result)

    except Exception as e:
        print(f"[ERROR] check_patient_status: {e}")
        return jsonify({'error': 'Server error while checking patient status'}), 500

# -------------------------------------------------------------------------------------------
# Analyze route (major logic kept but run_fit_test replaced with internal function)
# -------------------------------------------------------------------------------------------
@app.route('/analyze', methods=['POST'])
def analyze():
    import pickle, numpy as np, traceback, os, requests, tempfile
    from ml_utils import extract_features
    from flask import session, render_template

    selected_frames = []
    temp_files_map = {}

    try:
        # --- Load classifier & scaler ---
        with open("face_shape_model.pkl", "rb") as f:
            shape_model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        # --- Get selected patient_id from session ---
        selected_patient_id = session.get('selected_patient_id')
        if not selected_patient_id:
            return "No patient selected for analysis.", 400

        # --- Fetch latest date ---
        cursor.execute("""
            SELECT MAX(date) AS latest_date
            FROM capturedimages
            WHERE patient_id = %s
        """, (selected_patient_id,))
        latest_date_row = cursor.fetchone()
        latest_date = latest_date_row["latest_date"] if latest_date_row else None
        if not latest_date:
            return f"No captured images found for patient {selected_patient_id}.", 404

        # --- Fetch all images for that date ---
        cursor.execute("""
            SELECT img_id, crop_img_id, image_url_img, image_url_crop_img, patient_id, date
            FROM capturedimages
            WHERE patient_id = %s AND date = %s
            ORDER BY img_id ASC
        """, (selected_patient_id, latest_date))
        db_images = cursor.fetchall()

        feature_list, valid_images, frame_predictions = [], [], []

        # --- Download images & extract features ---
        for img_record in db_images:
            try:
                img_url = img_record['image_url_img']
                crop_url = img_record['image_url_crop_img']

                temp_dir = tempfile.gettempdir()
                temp_img = os.path.join(temp_dir, f"temp_{os.path.basename(img_url.split('?')[0])}")
                temp_crop = os.path.join(temp_dir, f"temp_{os.path.basename(crop_url.split('?')[0])}")

                for url, path in [(img_url, temp_img), (crop_url, temp_crop)]:
                    if not os.path.exists(path):
                        r = requests.get(url)
                        with open(path, "wb") as f:
                            f.write(r.content)

                features = extract_features(temp_img)
                if not features:
                    continue
                feature_list.append(features)
                valid_images.append(img_url)
                temp_files_map[img_url] = (temp_img, temp_crop)

                # --- ArUco detection ---
                aruco_ids = detect_aruco(temp_crop, min_marker_area=50)
                frame_shape, frame_brand = "Unknown", "Unknown"

                if aruco_ids:
                    marker_id = aruco_ids[0]
                    eyeglass_frame_id = get_eyeglass_frame_id(marker_id)
                    if eyeglass_frame_id:
                        frame_record = fetch_frame_details(eyeglass_frame_id)
                        if frame_record:
                            frame_shape = frame_record['frame_shape']
                            frame_brand = frame_record['frame_brand']

                            selected_frames.append({
                                "eyeglass_frame_id": frame_record.get("eyeglass_frame_id"),
                                "brand": frame_record.get("frame_brand", "Unknown"),
                                "model_number": frame_record.get("model_number", "N/A"),
                                "color": frame_record.get("frame_color", "N/A"),
                                "shape": frame_record.get("frame_shape", "N/A"),
                                "price": frame_record.get("price", 0.0),
                                "image_url": frame_record.get("image_url", ""),
                                "captured_photo": img_url
                            })

                frame_predictions.append((img_url, crop_url, frame_shape, frame_brand))

            except Exception as e:
                print(f"[ERROR] Processing image {img_record['img_id']}: {e}")
                traceback.print_exc()
                continue

        if not feature_list:
            return "Failed to extract features", 400

        # --- Face shape prediction ---
        features_scaled = scaler.transform(feature_list)
        class_probs = shape_model.predict_proba(features_scaled)
        avg_probs = np.mean(class_probs, axis=0)
        class_labels = shape_model.classes_
        average_prob_dict = {label: round(prob * 100, 2) for label, prob in zip(class_labels, avg_probs)}
        main_predicted_shape = class_labels[np.argmax(avg_probs)]
        main_shape_percent = round(np.max(avg_probs) * 100, 2)
        face_shape_key = main_predicted_shape.lower()

        # --- Fit scores, compatibility, best match ---
        # --- Fit scores ---
        fit_scores = {}
        marker_cache = {}
        for img_url in valid_images:
            try:
                temp_img, temp_crop = temp_files_map[img_url]
                aruco_ids = detect_aruco(temp_crop, min_marker_area=50)
                if not aruco_ids:
                    fit_scores[img_url] = 0.0
                    continue

                marker_id = aruco_ids[0]
                if marker_id in marker_cache:
                    fit_scores[img_url] = marker_cache[marker_id]
                    continue

                eyeglass_frame_id = get_eyeglass_frame_id(marker_id)
                if not eyeglass_frame_id:
                    fit_scores[img_url] = 0.0
                    marker_cache[marker_id] = 0.0
                    continue

                sizes = fetch_sizes_from_supabase(eyeglass_frame_id) or fetch_sizes_from_postgres(eyeglass_frame_id)
                if not sizes:
                    fit_scores[img_url] = 0.0
                    marker_cache[marker_id] = 0.0
                    continue

                eye = float(sizes.get("eye", 0))
                bridge = float(sizes.get("bridge", 0))
                temple = float(sizes.get("temple", 0))
                score = round(float(compute_fitscore(eye, bridge, temple)), 4)

                marker_cache[marker_id] = score
                fit_scores[img_url] = score

                print(
                    f"[FIT SCORE] Image: {img_url} | Eye: {eye}, Bridge: {bridge}, Temple: {temple} => FitScore: {score}")

            except Exception as e:
                print(f"[ERROR] run_fit_test for {img_url}: {e}")
                traceback.print_exc()
                fit_scores[img_url] = 0.0

        # --- Compatibility rules ---
        compatibility_rules = {
            "round": {"Rectangle": 3, "Geometric": 2, "Cat eye": 2, "Round": 1, "Oval": 2, "Circle": 2},
            "square": {"Round": 3, "Oval": 2, "Cat eye": 2, "Rectangle": 1, "Geometric": 2, "Circle": 2},
            "oval": {"Rectangle": 2, "Round": 2, "Oval": 3, "Geometric": 3, "Cat eye": 2, "Circle": 3},
            "heart": {"Cat eye": 3, "Circle": 3, "Oval": 2, "Round": 1, "Rectangle": 2, "Geometric": 2},
            "oblong": {"Oval": 3, "Cat eye": 2, "Rectangle": 2, "Round": 2, "Geometric": 2, "Circle": 2}
        }

        # --- Compute total score (50/50 Fit + Compatibility%) ---
        max_comp_score = max(compatibility_rules.get(face_shape_key, {}).values() or [1])  # avoid div by 0

        compatibility_scores = {}
        for frame in selected_frames:
            frame_shape = frame["shape"]
            fit_score = fit_scores.get(frame["captured_photo"], 0)

            comp_raw = compatibility_rules.get(face_shape_key, {}).get(frame_shape, 0)
            comp_percent = (comp_raw / max_comp_score) * 100  # normalize to 0-100%

            total_score = round(0.5 * comp_percent + 0.5 * fit_score, 4)

            compatibility_scores[frame["captured_photo"]] = {
                "total": round(total_score, 2),
                "comp_percent": round(comp_percent, 2),
                "fit_score": fit_score
            }

            # --- DEBUG summary ---
            print(
                f"[TOTAL SCORE] Image: {frame['captured_photo']} | Comp%: {comp_percent:.2f}, FitScore: {fit_score:.2f} => TotalScore: {total_score:.2f}")

        # --- Frame summary table ---
        print("\n[FRAME SCORES SUMMARY]")
        print(
            f"{'Image':<50} | {'Shape':<10} | {'Brand':<10} | {'FitScore':<10} | {'CompScore%':<12} | {'WeightedComp':<12} | {'WeightedFit':<12} | {'TotalScore':<10}")
        print("-" * 140)
        for frame in selected_frames:
            img = frame['captured_photo'][:45] + "..." if len(frame['captured_photo']) > 45 else frame['captured_photo']
            frame_shape = frame['shape']
            frame_brand = frame['brand']
            fit_score = fit_scores.get(frame['captured_photo'], 0)
            comp_raw = compatibility_rules.get(face_shape_key, {}).get(frame_shape, 0)
            comp_percent = (comp_raw / max_comp_score) * 100
            weighted_comp = 0.5 * comp_percent
            weighted_fit = 0.5 * fit_score
            total_score = round(weighted_comp + weighted_fit, 4)
            print(
                f"{img:<50} | {frame_shape:<10} | {frame_brand:<10} | {fit_score:<10.4f} | {comp_percent:<12.2f} | {weighted_comp:<12.4f} | {weighted_fit:<12.4f} | {total_score:<10.4f}")
        print("-" * 140)

        # --- Best match selection ---
        best_image = max(
            compatibility_scores.items(),
            key=lambda x: x[1]["total"]
        )[0] if compatibility_scores else None
        best_match, best_frame = None, None

        if best_image:
            best_match = next(
                ((img, crop, shape, brand)
                 for img, crop, shape, brand in frame_predictions
                 if img == best_image),
                None
            )

        if not best_match and frame_predictions:
            best_match = frame_predictions[0]

        frame_info = {
            "Rectangle": "Rectangle frames complement round faces by adding angles.",
            "Round": "Round frames soften square or angular face shapes.",
            "Oval": "Oval frames suit most face shapes and offer a balanced look.",
            "Geometric": "Geometric frames work well with oval and round face shapes.",
            "Cat eye": "Cat eye frames are ideal for heart-shaped faces and add a stylish lift.",
            "Circle": "Circle frames add balance to square and oblong faces.",
            "Unknown": "The frame could not be recognized. Please ensure a clear photo."
        }

        frame_description = (
            frame_info.get(best_match[2], "No description available.")
            if best_match else ""
        )

        if best_match:
            for frame in selected_frames:
                if frame["shape"] == best_match[2] and frame["brand"] == best_match[3]:
                    best_frame = frame
                    break

        # --- Load patient info ---
        cursor.execute("SELECT patient_id, patient_fname, patient_minitial, patient_lname FROM patient;")
        patients = [
            {
                "patient_id": r["patient_id"],
                "patient_fname": r["patient_fname"],
                "patient_minitial": r["patient_minitial"],
                "patient_lname": r["patient_lname"]
            }
            for r in cursor.fetchall()
        ]

        # INSERT FIX HERE
        if best_frame:
            selected_frames = [
                f for f in selected_frames
                if not (
                        f.get("model_number") == best_frame.get("model_number") and
                        f.get("brand") == best_frame.get("brand")
                )
            ]

        # --- Store selected frames in session ---
        session['selected_frames'] = selected_frames

        return render_template(
            "results.html",
            uploaded_images=[url for url in valid_images],
            average_features=[round(f, 4) for f in np.mean(feature_list, axis=0)],
            predicted_shape=main_predicted_shape,
            main_shape_percent=main_shape_percent,
            average_face_shape_probs=average_prob_dict,
            frame_predictions=frame_predictions,
            best_match=best_match,
            best_frame=best_frame,
            frame_description=frame_description,
            fit_scores=fit_scores,
            compatibility_scores=compatibility_scores,
            selected_frames=selected_frames,
            best_frame_id=best_frame["model_number"] if best_frame else None,
            patients=patients
        )

    finally:
        # --- Clean up temp files ---
        for temp_img, temp_crop in temp_files_map.values():
            for path in (temp_img, temp_crop):
                try:
                    if os.path.exists(path):
                        os.remove(path)
                        print(f"[CLEANUP] Deleted {path}")
                except Exception as e:
                    print(f"[CLEANUP ERROR] Could not delete {path}: {e}")



def get_recommended_frame_ids(patient_id):
    """Returns a set of eyeglass_frame_id that are allowed for this patient."""
    try:
        # Fetch patient age
        pres_query = supabase.table("patient").select("age").eq("patient_id", patient_id).execute()
        patient_data = getattr(pres_query, "data", []) or getattr(pres_query, "json", {}).get("data", [])
        if not patient_data:
            return set()

        age = patient_data[0].get("age", 0)

        # Fetch prescription conditions
        pres_query = supabase.table("prescription").select("condition").eq("patient_id", patient_id).execute()
        pres_data = getattr(pres_query, "data", []) or getattr(pres_query, "json", {}).get("data", [])

        condition_found = None

        for pres in pres_data:
            condition = (pres.get("condition") or "").lower()

            if condition == "acidic":
                condition_found = "acidic"
            elif condition in ["high prescription", "thick lenses"]:
                condition_found = "high prescription"

        # Children with no condition
        if age < 18 and not condition_found:
            condition_found = "active"

        # Determine allowed materials & frame types
        if condition_found == "acidic":
            material_filter = ["titanium", "acetate"]
            frame_filter = ["full rim", "half rim", "rim less"]

        elif condition_found == "high prescription":
            material_filter = ["thermoplastic", "acetate", "nylon", "polyamide"]
            frame_filter = ["full rim"]

        elif condition_found == "active":
            material_filter = ["polycarbonate", "trivex"]
            frame_filter = ["full rim"]

        else:
            material_filter = []
            frame_filter = ["full rim", "half rim", "rim less"]

        # Fetch inventory frames
        frame_query = supabase.table("eyeglassframeinventory").select(
            "eyeglass_frame_id, material, frame_type"
        ).execute()
        frame_data = getattr(frame_query, "data", []) or getattr(frame_query, "json", {}).get("data", [])

        valid_ids = set()

        for f in frame_data:
            material = (f.get("material") or "").lower()
            ftype = (f.get("frame_type") or "").lower()

            if (not material_filter or material in material_filter) and \
               (not frame_filter or ftype in frame_filter):

                fid = f.get("eyeglass_frame_id")
                if fid:
                    valid_ids.add(str(fid))  # <-- FIXED

        return valid_ids

    except Exception as e:
        print("[ERROR] get_recommended_frame_ids:", e)
        return set()




# -------------------------------------------------------------------------------------------
# take_photo route (unchanged except ArUco remains)
# -------------------------------------------------------------------------------------------
@app.route('/take_photo', methods=['GET','POST'])
def take_photo():
    global camera, conn, cursor, using_pi_camera

    patient_id = request.form.get('patient_id') or request.args.get('patient_id') or request.json.get('patient_id')
    if not patient_id:
        return jsonify({"error": "Missing patient ID"}), 400
    print(f"[ROUTE] /take_photo called, method={request.method}, patient_id={patient_id}")

    session['selected_patient_id'] = patient_id
    print(f"[SESSION] Stored selected_patient_id: {session['selected_patient_id']}")

    frame = None

    try:
        if using_pi_camera and camera is not None:
            print("[ROUTE] Capturing from PiCamera2")
            # Raspberry Pi
            frame = camera.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.flip(frame, 1)
        elif camera is not None:
            print("[ROUTE] Capturing from USB webcam")
            # USB webcam
            ret, frame = camera.read()
            if not ret or frame is None:
                return jsonify({"error": "Failed to capture image"}), 500
            frame = cv2.flip(frame, 1)
        elif 'file' in request.files:
            print("[ROUTE] Capturing from mobile upload")
            # Mobile/tablet upload
            file = request.files['file']
            npimg = np.frombuffer(file.read(), np.uint8)
            frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        else:
            print("[ROUTE] No camera available")
            return jsonify({"error": "Camera not available"}), 500
    except Exception as e:
        print(f"[ERROR] Failed to capture image: {e}")
        return jsonify({"error": "Failed to capture image"}), 500

    if frame is None:
        return jsonify({"error": "No frame captured"}), 500

    frame = cv2.flip(frame, 1)
    date_folder = datetime.now().strftime("%Y-%m-%d")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    photo_dir = os.path.join("static", "Photos")
    cropped_dir = os.path.join("static", "Cropped")
    os.makedirs(photo_dir, exist_ok=True)
    os.makedirs(cropped_dir, exist_ok=True)

    base_filename = f"{patient_id}_{timestamp}.jpg"
    local_photo_path = os.path.join(photo_dir, base_filename)
    cv2.imwrite(local_photo_path, frame)

    # --- Crop eyeglass region ---
    cropped_path = crop_eyeglass_region(local_photo_path)
    image_to_detect = cropped_path if cropped_path else local_photo_path

    # --- Detect ArUco ---
    aruco_info = detect_aruco(image_to_detect, min_marker_area=50)
    print(f"[INFO] Detected ArUco IDs: {aruco_info}")

    if not aruco_info:
        print("[WARNING] No ArUco detected, deleting image...")
        try:
            os.remove(local_photo_path)
            if cropped_path and os.path.exists(cropped_path):
                os.remove(cropped_path)
        except OSError:
            pass
        return jsonify({"aruco_error": "No ArUco marker detected"}), 400

    # --- Validate ArUco against allowed frames ---
    allowed_frame_ids = get_recommended_frame_ids(patient_id)
    # Convert detected numeric IDs to same string format as frame IDs
    detected_ids = set(f"EYEGLSSFRM-{id}" for id in aruco_info)
    print(f"[DEBUG] Allowed frame IDs: {allowed_frame_ids}")
    print(f"[DEBUG] Formatted detected IDs: {detected_ids}")

    if not allowed_frame_ids:
        print("[WARNING] No allowed frames returned for this patient.")

    # ❌ If none of the detected IDs belong to allowed frame list
    if not (allowed_frame_ids & detected_ids):
        detected_id = list(detected_ids)[0]
        print("[❌] ArUco ID not within allowed frame list.")
        return jsonify({
            "aruco_invalid": True,
            "detected_id": detected_id
        }), 200

    # --- Save / upload image normally ---
    try:
        base_path = f"capturedimage/{patient_id}/{date_folder}"
        img_path = f"{base_path}/img/{base_filename}"
        cropped_img_path = f"{base_path}/cropped_img/{base_filename}"

        # Upload full image
        with open(local_photo_path, "rb") as f:
            supabase.storage.from_("capturedimage").upload(
                path=img_path,
                file=f,
                file_options={"content-type": "image/jpeg"}
            )

        # Upload cropped image
        if cropped_path and os.path.exists(cropped_path):
            with open(cropped_path, "rb") as f:
                supabase.storage.from_("capturedimage").upload(
                    path=cropped_img_path,
                    file=f,
                    file_options={"content-type": "image/jpeg"}
                )

        # Delete local files
        os.remove(local_photo_path)
        if cropped_path and os.path.exists(cropped_path):
            os.remove(cropped_path)

        # --- Generate IDs ---
        cursor.execute("SELECT COUNT(*) AS total FROM capturedimages;")
        row = cursor.fetchone()
        total = row['total'] + 1
        img_id = f"IMG-{total}"
        crop_img_id = f"CRP-{total}"

        # --- Generate signed URLs ---
        img_signed = supabase.storage.from_("capturedimage").create_signed_url(
            img_path, expires_in=31536000
        )
        crop_signed = supabase.storage.from_("capturedimage").create_signed_url(
            cropped_img_path, expires_in=31536000
        )

        image_url_img = img_signed.get("signedURL") or img_signed.get("signed_url")
        image_url_crop_img = crop_signed.get("signedURL") or crop_signed.get("signed_url")

        # --- Insert into DB ---
        cursor.execute("""
            INSERT INTO capturedimages (
                img_id, crop_img_id, date, patient_id, image_url_img, image_url_crop_img
            ) VALUES (%s, %s, %s, %s, %s, %s)
        """, (img_id, crop_img_id, datetime.now().date(), patient_id, image_url_img, image_url_crop_img))
        conn.commit()

        print(f"[✅] Saved record {img_id} / {crop_img_id} for patient {patient_id}")
        return jsonify({
            "message": "Image captured and saved successfully",
            "img_id": img_id,
            "crop_img_id": crop_img_id,
            "image_url": image_url_img,
            "cropped_image_url": image_url_crop_img
        })

    except Exception as e:
        print("[ERROR] Upload or DB save failed!")
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500






# -------------------------------------------------------------------------------------------
# Example integration after detection (kept for debug)
# -------------------------------------------------------------------------------------------
# NOTE: this runs at import time; originally in your file. It detects markers in a folder path.
try:
    aruco_ids = detect_aruco("static/Cropped")  # if static/Cropped is an actual file this will log error, kept as in original
    if aruco_ids:
        marker_id = aruco_ids[0]  # take first detected
        eyeglass_frame_id = get_eyeglass_frame_id(marker_id)
        if eyeglass_frame_id:
            frame_record = fetch_frame_details(eyeglass_frame_id)
            if frame_record:
                print(f"[✅] Marker {marker_id} → Frame {eyeglass_frame_id} → Shape {frame_record['frame_shape']}")
except Exception:
    # make sure this optional debug section never crashes the server start
    pass

# -------------------------------------------------------------------------------------------
# Helper function for ID generation (kept intact)
# -------------------------------------------------------------------------------------------
def generate_unique_id(cursor, table, id_field, prefix, number_only=False, start_number=None):
    if not number_only:
        # Get the latest prefixed ID
        query = sql.SQL("""
            SELECT {id_field}
            FROM {table}
            WHERE {id_field} LIKE %s
            ORDER BY CAST(SUBSTRING({id_field} FROM '[0-9]+$') AS INTEGER) DESC
            LIMIT 1
        """).format(
            id_field=sql.Identifier(id_field),
            table=sql.Identifier(table)
        )
        cursor.execute(query, (f"{prefix}-%",))
        row = cursor.fetchone()

        if row and row[id_field]:
            try:
                latest_num = int(row[id_field].split("-")[1])
            except (IndexError, ValueError):
                latest_num = 0
        else:
            latest_num = 0

        return f"{prefix}-{latest_num + 1}"

    else:
        # For number-only (like invoice_number)
        query = sql.SQL("""
            SELECT {id_field}
            FROM {table}
            ORDER BY CAST({id_field} AS INTEGER) DESC
            LIMIT 1
        """).format(
            id_field=sql.Identifier(id_field),
            table=sql.Identifier(table)
        )
        cursor.execute(query)
        row = cursor.fetchone()

        if row and row[id_field]:
            try:
                latest_num = int(row[id_field])
            except ValueError:
                latest_num = start_number if start_number else 1
        else:
            latest_num = start_number if start_number else 1

        return str(latest_num + 1)

# -------------------------------------------------------------------------------------------
# select_frame route (kept intact)
# -------------------------------------------------------------------------------------------
@app.route("/select_frame", methods=["POST"])
def select_frame():
    import traceback
    try:
        form_data = request.form.to_dict()
        print("Form data received:", form_data)

        # === 1. Get patient_id from session ===
        patient_id = session.get('selected_patient_id')
        if not patient_id:
            flash("⚠️ No patient selected. Please capture a photo first.", "warning")
            return redirect(url_for("choose_frame"))

        # === 2. Fetch patient info ===
        cursor.execute("""
            SELECT patient_fname, patient_minitial, patient_lname, age, gender
            FROM patient
            WHERE patient_id = %s
        """, (patient_id,))
        patient_row = cursor.fetchone()

        if not patient_row:
            flash(f"❌ Patient record not found for ID: {patient_id}", "danger")
            return redirect(url_for("choose_frame"))

        patient_fname = patient_row["patient_fname"]
        patient_minitial = patient_row["patient_minitial"]
        patient_lname = patient_row["patient_lname"]
        patient_age = patient_row["age"]
        patient_gender = patient_row["gender"]

        # === 3. Fetch the most recent prescription for this patient ===
        cursor.execute("""
            SELECT prescription_id
            FROM prescription
            WHERE patient_id = %s
            ORDER BY prescription_date DESC
            LIMIT 1
        """, (patient_id,))
        prescription_row = cursor.fetchone()

        prescription_id = None
        if prescription_row:
            prescription_id = prescription_row["prescription_id"]
            print(f"📜 Found prescription ID: {prescription_id}")
        else:
            print(f"⚠️ No prescription found for patient {patient_id}")

        # === 4. Extract frame details from form ===
        frame_id = form_data.get("frame_id")
        brand = form_data.get("frame_brand")
        model_number = form_data.get("model_number")
        color = form_data.get("frame_color")
        shape = form_data.get("frame_shape")
        price = float(form_data.get("price", 0))
        face_shape_result = form_data.get("face_shape_result")
        image_url = form_data.get("image_url")  # Recommended frame image URL

        # === 5. Fetch most recent captured image for this patient ===
        cursor.execute("""
            SELECT image_url_img
            FROM capturedimages
            WHERE patient_id = %s
            ORDER BY date DESC
            LIMIT 1
        """, (patient_id,))
        captured_row = cursor.fetchone()
        image_url_patient = captured_row["image_url_img"] if captured_row else None

        if not image_url_patient:
            print(f"⚠️ No captured image found for patient {patient_id}")

        # === 6. Validate stock ===
        cursor.execute("""
            SELECT current_stock
            FROM eyeglassframeinventory
            WHERE eyeglass_frame_id = %s
        """, (frame_id,))
        stock_row = cursor.fetchone()

        if not stock_row:
            flash(f"❌ Frame {frame_id} not found in inventory.", "danger")
            return analyze()

        if stock_row["current_stock"] <= 0:
            flash(f"⚠️ Frame {frame_id} is out of stock.", "warning")
            return analyze()

        # === 7. Generate IDs ===
        invoice_id = generate_unique_id(cursor, "invoices", "invoice_id", "INVC")
        invoice_number = generate_unique_id(cursor, "invoices", "invoice_number", "INV", number_only=False)
        eyeglass_frame_invoice_id = generate_unique_id(cursor, "eyeglassframe", "eyeglass_frame_invoice_id", "FRMINVC")
        selected_frame_id = generate_unique_id(cursor, "selected_frame", "selected_frame_id", "SLCTDFRM")
        eye_exam_id = generate_unique_id(cursor, "eyeexam", "eye_exam_id", "EYEEXAM")

        # === 8. Insert invoice ===
        cursor.execute("""
            INSERT INTO invoices (
                invoice_number, invoice_id, patient_id,
                patient_fname, patient_minitial, patient_lname,
                transaction_date, eye_exams_type, eye_exams_price,
                optical_products_type, optical_products_price,
                total_price, balance_due, prescription_id
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            invoice_number, invoice_id, patient_id,
            patient_fname, patient_minitial, patient_lname,
            date.today(), "Eye Exam", "200",
            "Eyeglass Frame", price, price + 200, price + 200,
            prescription_id
        ))

        # === 9. Insert eyeglass frame record ===
        cursor.execute("""
            INSERT INTO eyeglassframe (
                patient_id, patient_fname, patient_minitial, patient_lname,
                price, eyeglass_frame_id, eyeglass_frame_invoice_id,
                date, prescription_id, age, gender
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            patient_id, patient_fname, patient_minitial, patient_lname,
            price, frame_id, eyeglass_frame_invoice_id,
            date.today(), prescription_id, patient_age, patient_gender
        ))

        # === 10. Insert eye exam ===
        cursor.execute("""
            INSERT INTO eyeexam (
                patient_id, patient_fname, patient_minitial, patient_lname,
                price, eye_exam_id, date, prescription_id, age, gender
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            patient_id, patient_fname, patient_minitial, patient_lname,
            "200", eye_exam_id, date.today(), prescription_id, patient_age, patient_gender
        ))

        # === 11. Insert selected frame ===
        recommended_flag = form_data.get("is_recommended")
        recommended_flag = "Yes" if recommended_flag and recommended_flag.strip().lower() in ["yes", "true", "1", "on"] else "No"

        cursor.execute("""
            INSERT INTO selected_frame (
                selected_frame_id, patient_id, prescription_id, invoice_id,
                eyeglass_frame_id, frame_brand, model_number, frame_shape,
                frame_color, price, recommended_frame, date,
                face_shape_result, image_url, image_url_patient
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            selected_frame_id, patient_id, prescription_id, invoice_id,
            frame_id, brand, model_number, shape, color,
            price, recommended_flag, date.today(),
            face_shape_result, image_url, image_url_patient
        ))

        # === 12. Update stock ===
        cursor.execute("""
            UPDATE eyeglassframeinventory
            SET current_stock = current_stock - 1
            WHERE eyeglass_frame_id = %s
        """, (frame_id,))

        # === 13. Insert in-app notification ===
        cursor.execute("""
            INSERT INTO notification (patient_id, message)
            VALUES (%s, %s)
        """, (
            patient_id,
            f"👓 You have successfully selected {brand} ({model_number}) with an Invoice ID of {invoice_id}."
        ))

        conn.commit()
        flash(f"✅ {brand} ({model_number}) saved successfully to invoice {invoice_id}!", "success")
        return redirect(url_for("choose_frame"))

    except Exception as e:
        conn.rollback()
        print(f"[❌] Error in /select_frame: {e}")
        traceback.print_exc()
        flash("❌ Failed to save frame selection. Please try again.", "danger")
        return analyze()



# -------------------------------------------------------------------------------------------
# get_latest_prescription (kept)
# -------------------------------------------------------------------------------------------
@app.route("/get_latest_prescription/<patient_id>")
def get_latest_prescription(patient_id):
    try:
        cursor.execute("""
            SELECT *
            FROM prescription
            WHERE patient_id = %s
            ORDER BY prescription_date DESC
            LIMIT 1;
        """, (patient_id,))
        row = cursor.fetchone()

        if row:
            # Map DB column → nice label
            field_labels = {
                "distance_sph_od": "Distance Sphere OD",
                "distance_sph_os": "Distance Sphere OS",
                "distance_cyl_od": "Distance Cylinder OD",
                "distance_cyl_os": "Distance Cylinder OS",
                "distance_axis_od": "Distance Axis OD",
                "distance_axis_os": "Distance Axis OS",
                "distance_va_od": "Distance VA OD",
                "distance_va_os": "Distance VA OS",
                "distance_add_od": "Distance Add OD",
                "distance_add_os": "Distance Add OS",
                "contact_sph_od": "Contact Sphere OD",
                "contact_sph_os": "Contact Sphere OS",
                "contact_cyl_od": "Contact Cylinder OD",
                "contact_cyl_os": "Contact Cylinder OS",
                "contact_axis_od": "Contact Axis OD",
                "contact_axis_os": "Contact Axis OS",
                "reading_sph_od": "Reading Sphere OD",
                "reading_sph_os": "Reading Sphere OS",
                "reading_cyl_od": "Reading Cylinder OD",
                "reading_cyl_os": "Reading Cylinder OS",
                "reading_axis_od": "Reading Axis OD",
                "reading_axis_os": "Reading Axis OS",
                "pd": "PD"
            }

            # Preserve order
            fields = list(field_labels.keys())

            # Build details string dynamically
            prescription_details = []
            for f in fields:
                value = row[f]
                if value not in (None, "", "null"):
                    prescription_details.append(f"{field_labels[f]}: {value}")

            prescription_text = " | ".join(prescription_details)

            return jsonify(
                success=True,
                prescription_id=row["prescription_id"],
                prescription_date=row["prescription_date"],
                prescription_text=prescription_text
            )

        return jsonify(success=False)

    except Exception as e:
        print("❌ Error fetching prescription:", e)
        return jsonify(success=False)

# -------------------------------------------------------------------------------------------
# choose_frame route (kept)
# -------------------------------------------------------------------------------------------
@app.route('/')
def choose_frame():
    # --- Fetch recent captured images from DB ---
    cursor.execute("""
        SELECT image_url_img
        FROM capturedimages
        ORDER BY date DESC, img_id DESC
    """)
    rows = cursor.fetchall()

    # Directly use the stored URL
    image_paths = [row['image_url_img'] for row in rows]

    page = request.args.get("page", default=1, type=int)

    return render_template('choose_frame.html', image_paths=image_paths)

# -------------------------------------------------------------------------------------------
# open-camera and video_feed endpoints (kept)
# -------------------------------------------------------------------------------------------
@app.route('/open-camera')
def open_camera():
    return render_template('choose_frame.html')  # Make sure this file exists

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# -------------------------------------------------------------------------------------------
# detect_face_shape helper (kept)
# -------------------------------------------------------------------------------------------
def detect_face_shape(image_path):
    features = extract_features(image_path)
    if features:
        prediction = model.predict([features])[0]
        return prediction
    else:
        return "Unknown"

# -------------------------------------------------------------------------------------------
# delete_photo (kept)
# -------------------------------------------------------------------------------------------
@app.route('/delete_photo', methods=['POST'])
def delete_photo():
    filename = request.form.get('filename')  # this should be img_id
    if not filename:
        return redirect(url_for('choose_frame'))

    try:
        # --- Fetch image paths from DB ---
        cursor.execute("""
            SELECT image_url_img, image_url_crop_img
            FROM capturedimages
            WHERE img_id = %s
        """, (filename,))
        row = cursor.fetchone()
        if not row:
            return redirect(url_for('choose_frame'))

        image_url_img = row['image_url_img']
        image_url_crop_img = row['image_url_crop_img']

        # --- Extract Supabase storage path from signed URL ---
        # Assuming your signed URLs look like: https://[bucket].supabase.co/storage/v1/object/public/capturedimage/...
        def extract_path(signed_url):
            # Split on '/object/public/' or '/object/signing...' depending on URL
            parts = signed_url.split('/object/')
            if len(parts) > 1:
                return parts[1].split('?')[0]  # remove query params
            return None

        img_path = extract_path(image_url_img)
        crop_path = extract_path(image_url_crop_img)

        # --- Delete from Supabase ---
        if img_path:
            supabase.storage.from_("capturedimage").remove([img_path])
        if crop_path:
            supabase.storage.from_("capturedimage").remove([crop_path])

        # --- Delete DB record ---
        cursor.execute("DELETE FROM capturedimages WHERE img_id = %s", (filename,))
        conn.commit()
        print(f"[✅] Deleted image {filename} and its cropped version from Supabase & DB.")

    except Exception as e:
        print("[ERROR] Failed to delete image:", e)

    return redirect(url_for('choose_frame'))


# -------------------------------------------------------------------------------------------
# template filter (kept)
# -------------------------------------------------------------------------------------------
@app.template_filter('format_time')
def format_time(value):
    return value.strftime('%I:%M %p') if isinstance(value, datetime) else value

# -------------------------------------------------------------------------------------------
# register route (kept)
# -------------------------------------------------------------------------------------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        role = request.form['role']
        user_fname = request.form['user_fname']
        user_lname = request.form['user_lname']
        user_username = request.form['user_username']
        email = request.form['email']
        password = request.form['password']

        hashed_pw = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

        try:
            if role == 'admin':
                print("Inserting admin record...")
                cursor.execute("""
                    INSERT INTO admin (
                        admin_fname, admin_lname, admin_username, email, password
                    ) VALUES (%s, %s, %s, %s, %s)
                """, (
                    user_fname, user_lname, user_username,
                    email, hashed_pw
                ))
            else:
                print("Inserting user record...")
                cursor.execute("""
                    INSERT INTO users (
                        user_fname, user_lname, user_username, email, password
                    ) VALUES (%s, %s, %s, %s, %s)
                """, (
                    user_fname, user_lname, user_username,
                    email, hashed_pw
                ))

            conn.commit()
            flash('Account created successfully!', 'success')
            return redirect(url_for('login'))

        except psycopg2.Error as e:
            conn.rollback()
            print("Database error:", e.pgerror)
            flash(f"Database error: {e.pgerror}", 'danger')
            return redirect(url_for('register'))

    return render_template('register.html')


@app.route("/check_username")
def check_username():
    username = request.args.get("username")
    cursor.execute("SELECT 1 FROM users WHERE user_username = %s", (username,))
    exists = cursor.fetchone() is not None
    return jsonify({"exists": exists})

@app.route("/check_email")
def check_email():
    email = request.args.get("email")
    cursor.execute("SELECT 1 FROM users WHERE email = %s", (email,))
    exists = cursor.fetchone() is not None
    return jsonify({"exists": exists})

# -------------------------------------------------------------------------------------------
# login route
# -------------------------------------------------------------------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        role = request.form['role']
        username = request.form.get("username")
        password = request.form.get("password")

        try:
            if role == 'admin':
                # --- ADMIN LOGIN ---
                cursor.execute(
                    "SELECT * FROM admin WHERE admin_username = %s OR email = %s",
                    (username, username)
                )
                admin = cursor.fetchone()

                if not admin:
                    flash("❌ Admin not found.", "error")
                    return render_template("login.html")

                stored_pw = admin["password"]
                valid = bcrypt.checkpw(password.encode("utf-8"), stored_pw.encode("utf-8")) \
                    if stored_pw.startswith("$2b$") else password == stored_pw

                if not valid:
                    flash("❌ Invalid password.", "error")
                    return render_template("login.html")

                # ✅ Store session
                session["user"] = {
                    "admin_id": admin["admin_id"],
                    "username": admin["admin_username"],
                    "fname": admin["admin_fname"],
                    "lname": admin["admin_lname"],
                    "role": "admin"
                }

                flash(f"👋 Welcome back, {admin['admin_fname']}!", "success")
                return redirect(url_for("choose_frame"))

            else:
                # --- PATIENT LOGIN ---
                cursor.execute(
                    "SELECT * FROM users WHERE user_username = %s OR email = %s",
                    (username, username)
                )
                user = cursor.fetchone()

                if not user:
                    flash("❌ User not found.", "error")
                    return render_template("login.html")

                stored_pw = user["password"]
                valid = bcrypt.checkpw(password.encode("utf-8"), stored_pw.encode("utf-8")) \
                    if stored_pw.startswith("$2b$") else password == stored_pw

                if not valid:
                    flash("❌ Invalid password.", "error")
                    return render_template("login.html")

                # ✅ Build patient code
                patient_code = f"PT-{user['user_id']}"

                # ✅ Retrieve corresponding patient_id
                cursor.execute("SELECT patient_id FROM patient WHERE user_id = %s", (user['user_id'],))
                patient_record = cursor.fetchone()
                patient_id = patient_record['patient_id'] if patient_record else None

                # ✅ Store session
                session["user"] = {
                    "user_id": user["user_id"],
                    "patient_id": patient_id,
                    "patient_code": patient_code,
                    "username": user["user_username"],
                    "fname": user["user_fname"],
                    "lname": user["user_lname"],
                    "role": "patient"
                }

                # ✅ Detect device & browser info
                ua_string = request.headers.get('User-Agent', '')
                user_agent = parse_user_agent(ua_string)
                browser = user_agent.browser.family or "Unknown Browser"
                os = user_agent.os.family or "Unknown OS"
                device_type = (
                    "Mobile" if user_agent.is_mobile else
                    "Tablet" if user_agent.is_tablet else
                    "PC" if user_agent.is_pc else
                    "Other"
                )

                # ✅ Create message
                message = f"👤 You logged in using {browser} on {os} ({device_type})."

                # ✅ Insert into notification table
                if patient_id:
                    cursor.execute("""
                        INSERT INTO notification (message, patient_id, read)
                        VALUES (%s, %s, false)
                    """, (message, patient_id))
                    conn.commit()

                flash("✅ Login successful!", "success")
                return redirect(url_for("choose_frame"))

        except Exception as e:
            conn.rollback()
            print(">>> Login error:", e)
            flash("⚠️ Unexpected error occurred. Please try again.", "error")
            return render_template("login.html")

    return render_template("login.html")

# -------------------------------------------------------------------------------------------
# forgot and reset password route
# -------------------------------------------------------------------------------------------
# ✅ Load Brevo credentials
BREVO_API_KEY = os.getenv("BREVO_API_KEY")
MAIL_SENDER_EMAIL = os.getenv("MAIL_SENDER_EMAIL")
MAIL_SENDER_NAME = os.getenv("MAIL_SENDER_NAME")

@app.route("/forgot_password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form.get("email")

        if not email:
            flash("⚠️ Please enter your email address.", "warning")
            return redirect(url_for("forgot_password"))

        # ✅ Check if user exists
        cursor.execute("SELECT user_id, user_fname, user_lname FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()

        if not user:
            flash("❌ No account found with that email address.", "danger")
            return redirect(url_for("forgot_password"))

        # ✅ Generate reset token
        token = secrets.token_urlsafe(32)
        expiry_time = datetime.now() + timedelta(hours=1)

        # ✅ Save token to DB
        cursor.execute("""
            UPDATE users
            SET reset_token = %s, token_expiry = %s
            WHERE email = %s
        """, (token, expiry_time, email))
        conn.commit()

        # ✅ Construct reset link
        reset_link = url_for("reset_password", token=token, _external=True)

        # ✅ Send password reset email using Brevo
        try:
            configuration = sib_api_v3_sdk.Configuration()
            configuration.api_key['api-key'] = BREVO_API_KEY
            api_instance = sib_api_v3_sdk.TransactionalEmailsApi(sib_api_v3_sdk.ApiClient(configuration))

            subject = "Password Reset Request - Eye Can See Optical Clinic"
            sender = {"email": MAIL_SENDER_EMAIL, "name": MAIL_SENDER_NAME}
            to = [{"email": email, "name": f"{user['user_fname']} {user['user_lname']}"}]

            html_content = f"""
                <html>
                    <body style="font-family: Arial, sans-serif; color: #333;">
                        <p>Hello {user['user_fname']},</p>
                        <p>We received a request to reset your password for your Eye Can See Optical Clinic account.</p>
                        <p>Click the link below to reset your password:</p>
                        <p><a href="{reset_link}" 
                              style="background-color: #9C2627; color: white; padding: 10px 20px; 
                                     text-decoration: none; border-radius: 5px;">Reset Password</a></p>
                        <p>This link will expire in 1 hour.</p>
                        <p>If you didn’t request this, please ignore this email.</p>
                        <br>
                        <p>Thank you,<br><strong>Eye Can See Optical Clinic</strong></p>
                    </body>
                </html>
            """

            send_smtp_email = sib_api_v3_sdk.SendSmtpEmail(
                to=to, html_content=html_content, sender=sender, subject=subject
            )
            api_instance.send_transac_email(send_smtp_email)

            flash("✅ Password reset email sent successfully!", "success")

        except ApiException as e:
            print("⚠️ Email sending failed:", e)
            flash("⚠️ Failed to send password reset email. Please try again later.", "danger")

        return redirect(url_for("forgot_password"))

    return render_template("forgot-password.html")


@app.route("/reset_password/<token>", methods=["GET", "POST"])
def reset_password(token):
    # Check if token is valid
    cursor.execute("""
        SELECT user_id, user_fname 
        FROM users 
        WHERE reset_token = %s AND token_expiry > NOW()
    """, (token,))
    user = cursor.fetchone()

    if not user:
        flash("❌ Invalid or expired reset link.", "danger")
        return redirect(url_for("forgot_password"))

    if request.method == "POST":
        new_password = request.form.get("new_password")

        if not new_password:
            flash("⚠️ Please enter a new password.", "warning")
            return redirect(url_for("reset_password", token=token))

        # Hash new password
        hashed_password = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

        # Update in DB and clear token
        cursor.execute("""
            UPDATE users 
            SET password = %s, reset_token = NULL, token_expiry = NULL
            WHERE user_id = %s
        """, (hashed_password, user["user_id"]))
        conn.commit()

        flash("✅ Password successfully reset! You can now log in.", "success")
        return redirect(url_for("login"))

    return render_template("reset_password.html", token=token)


# -------------------------------------------------------------------------------------------
# run app
# -------------------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
