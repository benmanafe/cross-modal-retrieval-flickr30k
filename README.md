# 🔍 Cross-Modal Retrieval System – Flickr30k

This project implements a **CLIP-style image–text retrieval system** trained from scratch on the **Flickr30k** dataset.
It allows users to **search for images using text captions** or **find captions describing an uploaded image** — all powered by deep visual–language embeddings and a Streamlit web app.

---

## 🧠 Overview

This system learns a **shared embedding space** between images and text.
When you query with text or an image, it finds the most semantically similar counterparts based on **cosine similarity** of learned representations.

Key components:
* **Image Encoder:** EfficientNet-B0 (fine-tuned on Flickr30k)
* **Text Encoder:** MiniLM (Sentence-Transformers)
* **Loss Function:** Symmetric CLIP-style contrastive loss
* **Deployment:** Streamlit interactive interface

---

## 🚀 Features

✅ **Text → Image Retrieval** – Enter any caption and find the top-10 most similar Flickr30k images.
✅ **Image → Text Retrieval** – Upload an image and see the top-10 matching captions.
✅ **Custom Fine-Tuned Model** – Jointly trained EfficientNet & MiniLM encoders.
✅ **GPU-Accelerated Training** – PyTorch implementation with AdamW optimizer.
✅ **Interactive Deployment** – Streamlit frontend for local or online demo.

---

## 🧩 Project Structure
