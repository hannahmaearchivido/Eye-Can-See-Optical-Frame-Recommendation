import cv2
import os

# ✅ Dictionary must match for both generation & detection
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)

# Output folder
output_dir = "aruco_markers_valid"
os.makedirs(output_dir, exist_ok=True)

# Detector
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

for i in range(1, 59):  # IDs 1 to 58
    marker_id = i

    # Generate marker (400x400, border size = 1 for padding)
    img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, 400)

    # Add a white border around (quiet zone)
    img = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)

    # Save image
    filename = os.path.join(output_dir, f"EYEGLSSFRM-{i}_aruco.png")
    cv2.imwrite(filename, img)

    # Detect immediately
    corners, ids, _ = detector.detectMarkers(img)

    if ids is not None and marker_id in ids:
        print(f"✅ Marker {marker_id} generated and detected.")
    else:
        print(f"❌ Marker {marker_id} failed detection!")

print(f"\n✅ Done. Markers saved in '{output_dir}'")
