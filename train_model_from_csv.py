import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# 📥 Load dataset
df = pd.read_csv("face_shape_dataset.csv")

# 📊 Use only first 4 features (modify as needed)
features = df.iloc[:, :4].values  # First 4 columns
labels = df["label"].values

# 🔃 Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# 💾 Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# 🎯 Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, labels, test_size=0.2, random_state=42
)

# 🌲 Train Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=4,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

# 💾 Save model
with open("face_shape_model.pkl", "wb") as f:
    pickle.dump(model, f)

# 🧪 Evaluate
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print("✅ Training Accuracy:", accuracy_score(y_train, y_pred_train))
print("✅ Testing Accuracy:", accuracy_score(y_test, y_pred_test))
print("\n📊 Classification Report on Test Set:\n", classification_report(y_test, y_pred_test))

# 📈 Plot feature importance
importances = model.feature_importances_
plt.figure(figsize=(10, 4))
plt.title("Feature Importances (First 4 Features)")
plt.bar(range(len(importances)), importances)
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()
