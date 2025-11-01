# 🔍 Cross-Modal Retrieval System (Flickr30k)

This repository contains a **CLIP-style cross-modal retrieval system** trained from scratch on the **Flickr30k dataset**.  
It aligns image and text embeddings to enable **text→image** and **image→text** retrieval using deep learning.

> ⚠️ **Note:** The Streamlit demo version is still under development and **not yet running successfully on Streamlit Cloud** due to dataset download issues.  
> The local version works correctly when all assets (model + embeddings + images) are available.

---

## 🧠 Overview

The system learns a **shared embedding space** for both modalities:
- **Image Encoder:** EfficientNet-B0  
- **Text Encoder:** MiniLM (Sentence-Transformers)  
- **Loss:** Symmetric CLIP-style contrastive loss  
- **Framework:** PyTorch  
- **Deployment target:** Streamlit app (in progress)

---

## 🚀 Features

✅ **Text → Image Retrieval** – Find top-10 most relevant Flickr30k images from a text query  
✅ **Image → Text Retrieval** – Retrieve top-10 captions that describe an uploaded image  
✅ **Custom Trained Model** – Joint image and text encoders with 1024-dimensional embedding space  
✅ **Embeddings Precomputed** – Faster inference and retrieval  
✅ **Planned Streamlit Integration** – Interactive UI for real-time retrieval (currently not working)

---

## 📂 Repository Structure

```
📦 cross-modal-retrieval-flickr30k/
├── app.py                        # Streamlit app (under debugging)
├── model.py                      # ImageEncoder, TextEncoder, CrossModalModel definitions
├── training_script.py            # Model training script
├── captions.txt                  # Flickr30k captions
├── model-checkpoints/
│   ├── model_epoch_10.pt
│   ├── img_embeds_epoch_10.pt
│   ├── txt_embeds_epoch_10.pt
├── requirements.txt              # Project dependencies
└── README.md                     # This file
```

---

## ⚙️ Setup (Local Run)

1. **Clone this repository**
   ```bash
   git clone https://github.com/yourusername/cross-modal-retrieval-flickr30k.git
   cd cross-modal-retrieval-flickr30k
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare files**
   - Place trained `.pt` model and embeddings in `model-checkpoints/`
   - Add `captions.txt`
   - Add Flickr30k image dataset under `flickr30k_images/`

4. **Run locally**
   ```bash
   streamlit run app.py
   ```

---

## 🧩 Current Streamlit Status

| Component | Status | Notes |
|------------|---------|--------|
| UI layout | ✅ Working | Clean two-mode interface (text→image / image→text) |
| Model loading | ✅ Working | Loads model + embeddings successfully |
| Dataset download | ⚠️ Issue | Kaggle dataset not downloading correctly on Streamlit Cloud |
| Image rendering | ⚠️ Blocked | Fails because image folder not found in deployed environment |
| Local run | ✅ Works fine | Fully functional with local dataset present |

---

## 🧠 Next Steps

- [ ] Fix dataset retrieval from Kaggle inside Streamlit environment  
- [ ] Re-deploy Streamlit app once dataset download succeeds  
- [ ] Optionally host demo on Hugging Face Spaces for faster loading  
- [ ] Add evaluation metrics (Recall@K, mAP)

---

## 💡 Technical Notes

**Model architecture**
- Image encoder: EfficientNet-B0 → Linear(1280 → 1024) → L2 normalization  
- Text encoder: MiniLM-L12-v2 → Linear(384 → 1024) → L2 normalization  
- Loss: Symmetric contrastive loss across modalities  

**Training details**
| Setting | Value |
|----------|--------|
| Epochs | 10 |
| Batch size | 64 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Dataset | Flickr30k |
| Final train loss | ≈ 1.48 |
| Final val loss | ≈ 2.31 |

---

## 📊 Example Results (Local)

| Query | Retrieved Images |
|--------|------------------|
| “Two men in green shirts drinking” | Returns several Flickr30k images with 2 men, often related but not always exact. |
| “A child playing with a dog” | Shows multiple relevant matches. |

---

## ⚠️ Streamlit Deployment Note

The current version of `app.py` tries to automatically download the dataset:
```python
kaggle datasets download -d eeshawn/flickr30k -f flickr30k_images.zip
```
However, **the Streamlit Cloud runtime fails to authenticate or download large datasets**.  
A fix is planned (using pre-hosted subset or Hugging Face dataset hosting).
