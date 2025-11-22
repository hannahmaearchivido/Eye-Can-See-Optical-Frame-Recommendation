import os
import csv
from ml_utils import extract_features  # Assuming your function is saved in ml_utils.py

dataset_dir = "testing_set"
output_csv = "face_shape_dataset.csv"

labels = []
features = []

# Loop through each face shape folder
for face_shape in os.listdir(dataset_dir):
    folder_path = os.path.join(dataset_dir, face_shape)
    if not os.path.isdir(folder_path):
        continue

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)
            feature_vector = extract_features(image_path)
            if feature_vector:
                features.append(feature_vector)
                labels.append(face_shape)

# Write to CSV
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['jaw_width', 'face_height', 'cheekbone_width', 'chin_angle', 'label'])
    for feat, label in zip(features, labels):
        writer.writerow(feat + [label])

print(f"[✅] Dataset saved to {output_csv}")