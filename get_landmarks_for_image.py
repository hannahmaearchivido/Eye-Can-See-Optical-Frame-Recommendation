# get_landmarks_for_image.py

import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

def get_face_landmarks(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not read image at {image_path}")
        return None

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7
    ) as face_mesh:
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            print(f"[!] Warning: No face landmarks detected in {image_path}")
            return None

        h, w, _ = image.shape
        face_landmarks = {}
        for idx, lm in enumerate(results.multi_face_landmarks[0].landmark):
            x, y = int(lm.x * w), int(lm.y * h)
            face_landmarks[idx] = (x, y)

        if len(face_landmarks) < 50:
            print(f"[!] Warning: Low landmark count ({len(face_landmarks)}) in {image_path}")
            return None

        return face_landmarks
