import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm

# Directory of unlabeled images
IMAGE_DIR = "unlabeled_eyewear/"
OUTPUT_DIR = "labeled_eyewear/"
NUM_CLUSTERS = 6  # Adjust based on expected frame types

# Load pretrained ResNet18
model = resnet18(pretrained=True)
model.fc = torch.nn.Identity()  # Remove classification head
model.eval()

# Image preprocessor
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def extract_features(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)  # [1, 3, 224, 224]
    with torch.no_grad():
        features = model(img_tensor).squeeze().numpy()  # 512-dim
    return features

# Load images and extract features
image_paths = [os.path.join(IMAGE_DIR, fname) for fname in os.listdir(IMAGE_DIR) if fname.endswith(('.jpg', '.png'))]
features = []

print("[*] Extracting features...")
for img_path in tqdm(image_paths):
    try:
        feat = extract_features(img_path)
        features.append(feat)
    except:
        print(f"Skipping corrupted image: {img_path}")

# Cluster images
print("[*] Clustering into groups...")
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
labels = kmeans.fit_predict(features)

# Organize clustered images for manual labeling
for idx, img_path in enumerate(image_paths):
    label = labels[idx]
    cluster_dir = os.path.join(OUTPUT_DIR, f"cluster_{label}")
    os.makedirs(cluster_dir, exist_ok=True)
    img = Image.open(img_path)
    img.save(os.path.join(cluster_dir, os.path.basename(img_path)))

print(f"[✓] Images grouped into {NUM_CLUSTERS} folders under '{OUTPUT_DIR}'.")
print("Now review each cluster folder and rename them to actual frame types like 'round', 'cat-eye', etc.")
