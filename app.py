import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
import joblib
from PIL import Image

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="MNIST Classifier Dashboard", layout="centered")
st.title("🧠 MNIST Digit Classifier: CNN vs SVM")

# Load MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_test_norm = X_test / 255.0
X_test_cnn = X_test_norm.reshape(-1, 28, 28, 1)
X_test_flat = X_test_norm.reshape(-1, 784)

# Load models
@st.cache_resource
def load_models():
    cnn = load_model("model/cnn_mnist.h5")
    svm = joblib.load("model/svm_mnist.pkl")
    return cnn, svm

cnn_model, svm_model = load_models()

# Preprocessing for SVM
scaler = StandardScaler()
X_train_flat = X_train.reshape(-1, 784) / 255.0
X_train_scaled = scaler.fit_transform(X_train_flat)
X_test_scaled = scaler.transform(X_test_flat)

pca = PCA(n_components=100)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Predict on full test set
cnn_preds = np.argmax(cnn_model.predict(X_test_cnn), axis=1)
svm_preds = svm_model.predict(X_test_pca)

cnn_acc = accuracy_score(y_test, cnn_preds)
svm_acc = accuracy_score(y_test, svm_preds)

# Sidebar: Model Accuracy
st.sidebar.title("📊 Model Accuracy")
st.sidebar.markdown(f"**✅ CNN:** {cnn_acc:.4f}")
st.sidebar.markdown(f"**🧠 SVM:** {svm_acc:.4f}")

# Sidebar: Misclassification Filter
st.sidebar.markdown("## 🔍 Image Filter")
filter_mode = st.sidebar.radio("Show images that are:", [
    "All", "Misclassified by CNN", "Misclassified by SVM", "Misclassified by Both"
])

# Get valid indices based on filter
if filter_mode == "All":
    valid_indices = np.arange(len(X_test))
elif filter_mode == "Misclassified by CNN":
    valid_indices = np.where(cnn_preds != y_test)[0]
elif filter_mode == "Misclassified by SVM":
    valid_indices = np.where(svm_preds != y_test)[0]
else:
    valid_indices = np.where((cnn_preds != y_test) & (svm_preds != y_test))[0]

if len(valid_indices) == 0:
    st.error("No matching images found with this filter.")
    st.stop()

# Choose input mode
use_uploaded = st.checkbox("📁 Use custom image (28x28 grayscale)")

if use_uploaded:
    uploaded = st.file_uploader("Upload PNG image", type=["png", "jpg", "jpeg"])
    if uploaded:
        image = Image.open(uploaded).convert("L").resize((28, 28))
        img_arr = np.array(image) / 255.0
        test_image = img_arr.reshape(1, 28, 28)
        test_cnn = test_image.reshape(1, 28, 28, 1)
        test_flat = test_image.reshape(1, 784)
        test_scaled = scaler.transform(test_flat)
        test_pca = pca.transform(test_scaled)
        true_label = None
    else:
        st.warning("Please upload a valid image.")
        st.stop()
else:
    selected_idx = st.slider("Select image index", 0, len(valid_indices) - 1, 0)
    img_index = valid_indices[selected_idx]
    test_image = X_test_norm[img_index:img_index+1]
    test_cnn = X_test_cnn[img_index:img_index+1]
    test_pca = X_test_pca[img_index:img_index+1]
    true_label = y_test[img_index]

# Predict
cnn_pred = np.argmax(cnn_model.predict(test_cnn), axis=1)[0]
svm_pred = svm_model.predict(test_pca)[0]

# Show image and predictions
st.subheader("🔍 Image and Predictions")

col1, col2 = st.columns(2)
with col1:
    st.image(test_image.reshape(28, 28), width=150,
             caption="Uploaded Image" if use_uploaded else f"True Label: {true_label}")
with col2:
    st.markdown(f"### ✅ CNN Prediction: `{cnn_pred}`")
    st.markdown(f"### 🧠 SVM Prediction: `{svm_pred}`")
