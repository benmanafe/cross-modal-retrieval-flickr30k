# 🔍 Cross-Modal Retrieval System – Flickr30k

This project implements a **CLIP-style image–text retrieval system** trained from scratch on the **Flickr30k** dataset.  
It allows users to **search for images using text captions** or **find captions describing an uploaded image** — all powered by deep visual–language embeddings and a Streamlit web app.

---

## 🧠 Overview

This system learns a **shared embedding space** between images and text.  
When you query with text or an image, it finds the most semantically similar counterparts based on **cosine similarity** of learned representations.

Key components:
- **Image Encoder:** EfficientNet-B0 (fine-tuned on Flickr30k)
- **Text Encoder:** MiniLM (Sentence-Transformers)
- **Loss Function:** Symmetric CLIP-style contrastive loss
- **Deployment:** Streamlit interactive interface

---

## 🚀 Features

✅ **Text → Image Retrieval** – Enter any caption and find the top-10 most similar Flickr30k images.  
✅ **Image → Text Retrieval** – Upload an image and see the top-10 matching captions.  
✅ **Custom Fine-Tuned Model** – Jointly trained EfficientNet & MiniLM encoders.  
✅ **GPU-Accelerated Training** – PyTorch implementation with AdamW optimizer.  
✅ **Interactive Deployment** – Streamlit frontend for local or online demo.

---

## 🧩 Project Structure

```
📦 cross-modal-retrieval-flickr30k/
├── app.py                         # Streamlit web app
├── model.py                       # Model definitions (ImageEncoder, TextEncoder, CrossModalModel)
├── training_script.py              # Model training & fine-tuning code
├── captions.txt                   # Flickr30k captions file
├── flickr30k_images/              # Flickr30k image directory
├── model-checkpoints/
│   ├── model_epoch_10.pt
│   ├── img_embeds_epoch_10.pt
│   ├── txt_embeds_epoch_10.pt
└── README.md                      # This file
```

---

## 🧱 Model Architecture

### 🔹 Image Encoder
- Backbone: `EfficientNet-B0`
- Output: 1280 → 1024 linear projection  
- Normalized embeddings

### 🔹 Text Encoder
- Backbone: `MiniLM-L12-v2` (from `sentence-transformers`)
- Output: 384 → 1024 linear projection  
- Mean pooling across token embeddings

### 🔹 Contrastive Loss (CLIP-style)
```python
loss = (CE(image_logits, labels) + CE(text_logits, labels)) / 2
```
Where logits are computed via scaled cosine similarity.

---

## 📈 Training Details

| Parameter | Value |
|------------|--------|
| Dataset | Flickr30k |
| Train Size | 145,000 image-caption pairs |
| Optimizer | AdamW |
| LR | 1e-4 (fine-tune: 5e-6) |
| Batch Size | 64 |
| Epochs | 10 |
| Embed Dim | 1024 |
| Loss | Symmetric Cross-Entropy |

During training, the model consistently reduced loss from **3.37 → 1.48**, showing strong convergence.

---

## 💻 Deployment (Streamlit)

Run locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```

The app allows:
- Typing a text prompt to retrieve the **top 10 most relevant images**
- Uploading an image to retrieve **top 10 matching captions**

### Example UI

#### 🗨️ Text → Image
> “Two men in green shirts drinking”

➡ Displays 10 similar Flickr30k images.

#### 🖼️ Image → Text
> Upload an image → Returns top 10 captions ranked by similarity score.

---

## 📊 Results

**Validation Loss Progression:**
| Epoch | Train Loss | Val Loss |
|:------|-------------|-----------|
| 1 | 3.38 | 2.80 |
| 5 | 1.84 | 2.40 |
| 10 | 1.48 | 2.31 |

**Similarity Heatmap Example:**

A strong diagonal indicates learned alignment between images and captions.

![Cosine Similarity Matrix](assets/similarity_matrix.png)

---

## 🧠 Insights

- The model learns cross-modal alignment effectively but benefits from additional fine-tuning (smaller LR, larger batches).
- Averaging multiple captions per image improves retrieval stability.
- Embedding normalization is essential for consistent cosine similarity results.

---

## 🧩 Future Improvements

- [ ] Extend training with CLIP-like pretraining datasets (e.g., LAION)
- [ ] Add hard negative sampling
- [ ] Try larger embedding dimensions or ViT backbone
- [ ] Quantitative evaluation (Recall@K metrics)
- [ ] Deploy with FastAPI + Streamlit hybrid backend

---

## ⚙️ Requirements

```bash
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.40.0
pandas
numpy
pillow
matplotlib
seaborn
tqdm
streamlit
```

---

## 🧑‍💻 Author

**Your Name**  
📧 your.email@example.com  
🔗 [LinkedIn Profile](https://linkedin.com/in/yourprofile)  
💼 [GitHub Repository](https://github.com/yourusername/cross-modal-retrieval-flickr30k)

---

## 🪪 License
MIT License © 2025 Your Name.  
Feel free to use, modify, and distribute this project with attribution.

---

### ⭐ If you find this useful, consider giving the repo a star!
