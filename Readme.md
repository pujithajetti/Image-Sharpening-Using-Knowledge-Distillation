# 🔍 AI Image Sharpening using Knowledge Distillation

A lightweight, real-time image sharpening pipeline for degraded video frames using a student-teacher model design. Ideal for video conferencing enhancement.

---

## 📦 Project Structure

```
project/
├── datasets/
│   ├── original/      # High-quality sharp images (input for training)
│   └── noisy/         # Blurry versions generated using degrader
├── models/            # Trained model saved here (student_model.pth)
├── degrader.py        # Script to degrade sharp images to simulate poor quality
├── train.py           # Knowledge distillation-based training script
├── models.py          # Student model definition (lightweight CNN)
├── inference.py       # Model inference utilities
├── app.py             # Streamlit UI for sharpening images
├── requirements.txt   # Dependencies
└── README.md
```

---

## 🚀 Steps to Run

### 1. 📁 Prepare Dataset
Place your **sharp images** in:
```bash
datasets/original/
```

### 2. 🌀 Generate Blurry Images
```bash
python degrader.py
```
This creates degraded versions in `datasets/noisy/`

---

### 3. 🧠 Train Student Model (CPU/GPU Auto-detect)
```bash
python train.py
```
This will train the student model using blurry→sharp pairs and save to:
```bash
models/student_model.pth
```

---

### 4. 🖼️ Try It Out with Streamlit
```bash
streamlit run app.py
```
Upload a blurry image → get the sharpened output in real time.

---

## ⚙️ Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```
If using GPU:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## ✅ Features
- Knowledge distillation from teacher → lightweight student
- Supports CPU + GPU
- Real-time sharpening with Streamlit app
- Clean blurry images caused by low network, compression, blur

---

## 🤖 Model
**StudentNet**: 3-layer CNN (fast, light)  
Loss: `L1 + Perceptual Loss (LPIPS)`

---


