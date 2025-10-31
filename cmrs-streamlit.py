import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
import streamlit as st
import os, zipfile

from model import CrossModalModel

# ------------------------------
# Streamlit Page Config
# ------------------------------
st.set_page_config(page_title="Cross-Modal Retrieval (Flickr30k)", layout="wide")
st.title("🔍 Cross-Modal Retrieval System – Flickr30k")
st.markdown("Search images with text, or captions with images.")

import subprocess # Use subprocess for better error checking
def ensure_images_from_kaggle():
    image_dir = "flickr30k_images"
    zip_path = "flickr30k.zip" 
    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")

    if not (os.path.exists(image_dir) and len(os.listdir(image_dir)) > 5):
        st.info("📦 Setting up Kaggle credentials...")
        os.makedirs(kaggle_dir, exist_ok=True)

        try:
            kaggle_username = st.secrets["KAGGLE_USERNAME"]
            kaggle_key = st.secrets["KAGGLE_KEY"]
            with open(kaggle_json, "w") as f:
                f.write(f'{{"username":"{kaggle_username}","key":"{kaggle_key}"}}')
            os.chmod(kaggle_json, 0o600)
        except FileNotFoundError:
            if not os.path.exists(kaggle_json):
                st.error("Kaggle API credentials not found.")
                st.stop("Add KAGGLE_USERNAME/KEY to Streamlit Secrets & reboot.")

        st.info("📦 Installing Kaggle CLI...")
        subprocess.run(["pip", "install", "-q", "kaggle"], check=True)

        st.info("📦 Downloading Flickr30k dataset (this may take a moment)...")
        
        download_command = [
            "kaggle", "datasets", "download",
            "-d", "eeshawn/flickr30k"
        ]
        
        result = subprocess.run(download_command, capture_output=True, text=True)

        if result.returncode != 0:
            st.error(f"Kaggle download failed. Error:\n{result.stderr}")
            st.warning("Please ensure you have accepted the dataset rules on the Kaggle website (and rebooted the app).")
            st.stop()
        
        st.info("Extracting images...")
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(image_dir)
            os.remove(zip_path)

            nested_img_dir = os.path.join(image_dir, "flickr30k_images") 

            if os.path.exists(nested_img_dir):
                st.info("Fixing directory structure...")
                for item in os.listdir(nested_img_dir):
                    shutil.move(os.path.join(nested_img_dir, item), os.path.join(image_dir, item))
                os.rmdir(nested_img_dir)

            st.success("✅ Images ready!")
            
        except (zipfile.BadZipFile, FileNotFoundError):
            st.error(f"Failed to unzip file. Expected '{zip_path}' but it was not found. Download may have failed.")
            st.stop()
            
    return image_dir
# ------------------------------
# 1. Load Model and Data
# ------------------------------
@st.cache_resource
def load_model_and_data():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = CrossModalModel(embed_dim=1024)
    model.load_state_dict(torch.load("model-checkpoints/model_epoch_10.pt", map_location=device))
    model = model.to(device)
    model.eval()

    # Load embeddings
    img_embeds = torch.load("model-checkpoints/img_embeds_epoch_10.pt", map_location=device)
    txt_embeds = torch.load("model-checkpoints/txt_embeds_epoch_10.pt", map_location=device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")

    # Load captions
    captions = pd.read_csv("captions.txt", sep=",")[["image_name", "comment"]]

    # Ensure images exist (download if needed)
    image_root = ensure_images_from_kaggle()
    image_paths = [os.path.join(image_root, img) for img in captions["image_name"].unique()]

    return model, tokenizer, img_embeds, txt_embeds, captions, image_paths, device


model, tokenizer, img_embeds, txt_embeds, captions, image_paths, device = load_model_and_data()

# ------------------------------
# 2. Image Preprocessing
# ------------------------------
val_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ------------------------------
# 3. Retrieval Functions
# ------------------------------
def retrieve_images(text_query, top_k=10):
    tokens = tokenizer(
        text_query,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    with torch.no_grad():
        text_embed = model.text_encoder(
            tokens["input_ids"].to(device),
            tokens["attention_mask"].to(device)
        )
    sims = torch.matmul(text_embed, img_embeds.T).squeeze(0)
    topk_idx = torch.topk(sims, top_k).indices.cpu().numpy()
    results = [image_paths[i] for i in topk_idx]
    scores = [sims[i].item() for i in topk_idx]
    return results, scores


def retrieve_texts(uploaded_image, top_k=10):
    image = val_test_transform(uploaded_image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embed = model.image_encoder(image)
    sims = torch.matmul(image_embed, txt_embeds.T).squeeze(0)
    topk_idx = torch.topk(sims, top_k).indices.cpu().numpy()
    results = captions.iloc[topk_idx][["comment", "image_name"]]
    scores = [sims[i].item() for i in topk_idx]
    return results, scores

# ------------------------------
# 4. Streamlit Interface
# ------------------------------
mode = st.radio("Choose a retrieval mode:", ["🗨️ Text → Image", "🖼️ Image → Text"], horizontal=True)

if mode == "🗨️ Text → Image":
    query = st.text_input("Enter a caption or text query:")
    if query:
        st.info("Searching for the most similar images (Top 10)...")
        images, scores = retrieve_images(query)
        for row_start in range(0, 10, 5):
            cols = st.columns(5)
            for i, col in enumerate(cols):
                idx = row_start + i
                if idx < len(images):
                    col.image(images[idx], caption=f"Score: {scores[idx]:.3f}", width=220)

elif mode == "🖼️ Image → Text":
    uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Query Image", width=300)
        st.info("Retrieving the most similar captions...")
        results, scores = retrieve_texts(image)
        for i, (idx, row) in enumerate(results.iterrows()):
            st.markdown(f"**{i+1}.** *{row['comment']}*  \n**Score:** {scores[i]:.3f}")






