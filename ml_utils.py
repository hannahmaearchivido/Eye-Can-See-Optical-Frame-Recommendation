import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

def angle_between(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.arccos(np.clip(cosine_angle, -1.0, 1.0))

def extract_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not load image: {image_path}")
        return None

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            print(f"[WARNING] No face detected in: {image_path}")
            return None

        landmarks = results.multi_face_landmarks[0].landmark

        def point(i):
            return np.array([landmarks[i].x, landmarks[i].y])

        def dist(a, b):
            return np.linalg.norm(point(a) - point(b))

        # Base measurement for normalization
        face_height = dist(10, 152)

        # Normalized distances
        jaw_width = dist(234, 454) / face_height
        cheekbone_width = dist(93, 323) / face_height
        chin_angle = angle_between(point(234), point(152), point(454))  # Keep angle as-is

        return [jaw_width, 1.0, cheekbone_width, chin_angle]
