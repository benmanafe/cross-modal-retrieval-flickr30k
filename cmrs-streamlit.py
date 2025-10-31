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

st.set_page_config(page_title="Cross-Modal Retrieval (Flickr30k)", layout="wide")
st.title("🔍 Cross-Modal Retrieval System – Flickr30k")
st.markdown("Search images with text, or captions with images.")

def ensure_images_from_kaggle():
    image_dir = "flickr30k_images"
    zip_path = "flickr30k_images.zip"

    if not os.path.exists(image_dir):
        st.info("📦 Downloading Flickr30k images from Kaggle (this happens only once)...")

        os.system("pip install -q kaggle")

        os.system("kaggle datasets download -d eeshawn/flickr30k -f flickr30k_images.zip")

        # Extract and clean up
        with zipfile.ZipFile("flickr30k_images.zip", 'r') as zip_ref:
            zip_ref.extractall(image_dir)
        os.remove("flickr30k_images.zip")

    return image_dir


@st.cache_resource
def load_model_and_data():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CrossModalModel(embed_dim=1024)
    model.load_state_dict(torch.load("model-checkpoints/model_epoch_10.pt", map_location=device))
    model = model.to(device)
    model.eval()

    img_embeds = torch.load("model-checkpoints/img_embeds_epoch_10.pt", map_location=device)
    txt_embeds = torch.load("model-checkpoints/txt_embeds_epoch_10.pt", map_location=device)

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")

    captions = pd.read_csv("captions.txt", sep=",")[["image_name", "comment"]]

    image_root = ensure_images_from_kaggle()
    image_paths = [os.path.join(image_root, img) for img in captions["image_name"].unique()]

    return model, tokenizer, img_embeds, txt_embeds, captions, image_paths, device


model, tokenizer, img_embeds, txt_embeds, captions, image_paths, device = load_model_and_data()

val_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

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


