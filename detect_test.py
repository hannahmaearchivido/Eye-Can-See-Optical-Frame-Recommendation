import cv2
import os

# ✅ Use the same dictionary as generation
DICT = cv2.aruco.DICT_4X4_100

# Folder where your markers are stored
folder = r"C:\Users\bruce\Desktop\new\choose eye frame\aruco_markers_valid"

# Prepare detector
aruco_dict = cv2.aruco.getPredefinedDictionary(DICT)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

for filename in sorted(os.listdir(folder)):
    if filename.endswith(".png"):
        path = os.path.join(folder, filename)
        img = cv2.imread(path)
        if img is None:
            print(f"[ERROR] Could not read {path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        print(f"\n🔎 Checking {filename}")

        if ids is not None and len(ids) > 0:
            detected_ids = ", ".join(map(str, ids.flatten()))
            print(f"✅ {filename} → Detected ID(s): {detected_ids}")
        else:
            print(f"❌ {filename} → No markers detected")
