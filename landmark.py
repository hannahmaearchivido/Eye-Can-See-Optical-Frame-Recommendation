# test_landmark_detection.py

from get_landmarks_for_image import get_face_landmarks
import cv2

# Manually specify an image path (adjust this to your actual image path)
image_path = "static/Photos/photo_20250727160635.jpg"

# Call the landmark function
landmarks = get_face_landmarks(image_path)

# Display results
if landmarks is None:
    print("[FAIL] Landmarks not detected.")
else:
    print("[OK] Landmarks successfully detected.")
    print(f"Total landmarks: {len(landmarks)}")

    # Optional: Draw and show the landmarks
    image = cv2.imread(image_path)
    for (x, y) in landmarks.values():
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    cv2.imshow("Landmarks", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
