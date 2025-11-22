import cv2
import numpy as np
import mediapipe as mp
import logging
import os

logger = logging.getLogger("mediapipe_utils")
logger.setLevel(logging.INFO)

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)

def _dist(a, b):
    """Compute Euclidean distance between 2D points."""
    return float(np.linalg.norm(np.array(a) - np.array(b)))

def _coords(lm, idx, w, h):
    """Convert landmark index to pixel coordinates."""
    return (int(lm[idx].x * w), int(lm[idx].y * h))

def estimate_eye_bridge_temple(image_path):
    """
    Extracts user facial measurement distances in **pixels** using MediaPipe Face Mesh.
    Returns:
        {
            "eye_size": float(px),
            "bridge_width": float(px),
            "temple_length": float(px)
        }

    Landmarks used (468-point model):
      - Left eye outer: 33
      - Left eye inner: 133
      - Right eye outer: 263
      - Right eye inner: 362
      - Left face edge (temple area): 234
      - Right face edge (temple area): 454

    You can adjust these indices if your results are slightly off.
    """
    if not image_path or not os.path.exists(image_path):
        logger.error(f"[USER PX] Image path invalid: {image_path}")
        return None

    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"[USER PX] Could not read image: {image_path}")
        return None

    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        logger.warning(f"[USER PX] No face detected in {image_path}")
        return None

    lm = results.multi_face_landmarks[0].landmark
    n = len(lm)

    # Safe index check
    def safe_idx(i): return i if 0 <= i < n else None

    try:
        left_outer = _coords(lm, safe_idx(33), w, h)
        left_inner = _coords(lm, safe_idx(133), w, h)
        right_outer = _coords(lm, safe_idx(263), w, h)
        right_inner = _coords(lm, safe_idx(362), w, h)
        left_edge = _coords(lm, safe_idx(234), w, h)
        right_edge = _coords(lm, safe_idx(454), w, h)
    except Exception:
        # If any landmark fails, fallback
        xs = [int(pt.x * w) for pt in lm]
        ys = [int(pt.y * h) for pt in lm]
        face_width = max(xs) - min(xs)
        eye = face_width * 0.18
        bridge = face_width * 0.10
        temple = face_width * 0.95
        logger.info(f"[USER PX] Fallback heuristics -> eye:{eye:.2f} bridge:{bridge:.2f} temple:{temple:.2f}")
        return {
            "eye_size": round(eye, 2),
            "bridge_width": round(bridge, 2),
            "temple_length": round(temple, 2)
        }

    # Compute distances
    left_eye_w = _dist(left_outer, left_inner)
    right_eye_w = _dist(right_outer, right_inner)
    eye_size = (left_eye_w + right_eye_w) / 2.0
    bridge = _dist(left_inner, right_inner)
    temple = _dist(left_edge, right_edge)

    # Logging for debugging
    logger.info(f"[USER PX] {os.path.basename(image_path)} -> "
                f"eye:{eye_size:.2f}px bridge:{bridge:.2f}px temple:{temple:.2f}px")

    return {
        "eye_size": round(eye_size, 2),
        "bridge_width": round(bridge, 2),
        "temple_length": round(temple, 2)
    }
