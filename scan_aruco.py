import cv2
import os

def scan_aruco_images(folder=r"C:\Users\bruce\Desktop\new\choose eye frame\aruco_markers_valid", min_marker_area=100):
    # Load the ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    results = {}

    for file in os.listdir(folder):
        if file.endswith("_aruco.png"):
            path = os.path.join(folder, file)
            img = cv2.imread(path)
            if img is None:
                print(f"[❌] Could not read {file}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)

            if ids is not None:
                marker_ids = [int(id_) for id_ in ids.flatten()]
                results[file] = marker_ids
                print(f"[✅] {file} → Detected IDs: {marker_ids}")
            else:
                results[file] = []
                print(f"[⚠️] {file} → No markers detected")

    return results


if __name__ == "__main__":
    detected = scan_aruco_images()

    print("\n📊 Summary:")
    for fname, ids in detected.items():
        print(f"{fname}: {ids}")
