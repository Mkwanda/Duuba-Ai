# Duuba-AI

A Streamlit-based web application for **real-time cocoa pod disease detection** using TensorFlow Object Detection API. The app identifies healthy and infected cocoa pods from images and videos with confidence scores and visual annotations.

---

## 🎯 Features

- **Real-time Detection**: Upload images or use your webcam to detect cocoa pod diseases instantly.
- **Two Disease Classes**: Classifies pods as **Healthy** or **Infected** (with disease type).
- **Confidence Filtering**: Adjust confidence threshold with a slider to fine-tune detections.
- **Annotated Output**: Download 800×800 pixel annotated images with bounding boxes and confidence scores.
- **Model Auto-Download**: Automatically downloads the pre-trained model from Google Drive if not present.
- **Lightweight Deployment**: Docker and Heroku-ready with Procfile and Dockerfile included.

---

## 📋 Prerequisites

- **Python 3.8+** (3.10 recommended)
- **pip** or **conda** for package management
- **Virtual environment** (recommended)

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Mkwanda/Duuba-AI.git
cd Duuba-AI
```

### 2. Create and Activate Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements-deploy.txt
```

### 4. Run the Streamlit App

```bash
streamlit run Duuba-AI.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📁 Project Structure

```
Duuba-AI/
├── Duuba-AI.py                 # Main Streamlit app
├── download_model.py           # Model downloader utility (Google Drive support)
├── requirements-deploy.txt     # Python dependencies
├── Dockerfile                  # Docker container configuration
├── Procfile                    # Heroku deployment config
├── DEPLOY.md                   # Deployment guide (Docker, Heroku, Streamlit Cloud)
├── README.md                   # This file
├── LICENSE                     # MIT License
├── annotations/
│   ├── label_map.pbtxt        # Class labels (Healthy, Infected)
│   ├── train.record           # TFRecord training data
│   └── test.record            # TFRecord test data
├── Duuba-AI-workspace/        # Lightweight workspace for deployment
│   ├── Duuba-AI.py
│   ├── download_model.py
│   ├── requirements-deploy.txt
│   ├── Dockerfile
│   ├── Procfile
│   └── Cocoa/Tensorflow/workspace/models/
├── Cocoa/Tensorflow/          # TensorFlow setup (models, scripts)
├── my_ssd_mobnetpod/          # Pre-trained SSD MobileNet v2 model
│   ├── checkpoint
│   ├── pipeline.config
│   ├── train/
│   ├── eval/
│   ├── export/
│   └── tfliteexport/
├── upload_images/             # Sample test images
└── data/
    ├── healthy/               # Training images (healthy pods)
    └── infected/              # Training images (infected pods)
```

---

## 🖼️ Usage

### Upload an Image

1. Open the Streamlit app.
2. Click **"Upload Image"** in the sidebar.
3. Select a JPG/PNG image of a cocoa pod.
4. Adjust the **Confidence Threshold** slider if desired.
5. Click **"Detect Disease"**.
6. View results: bounding boxes, confidence scores, and disease type.
7. Download the annotated 800×800 image using the **Download** button.

### Use Webcam (Camera)

1. Click **"Use Webcam"** in the sidebar.
2. Allow browser camera access.
3. Adjust confidence threshold.
4. Click **"Capture and Detect"**.
5. View and download the annotated frame.

### Video Upload

1. Upload a video file (MP4, AVI, MOV).
2. The app processes frames and displays detections with timestamps.

---

## ⚙️ Configuration

### Model URL

The app uses a pre-trained SSD MobileNet v2 model. To use a custom model:

1. Update `DEFAULT_MODEL_URL` in `Duuba-AI.py`:
   ```python
   DEFAULT_MODEL_URL = "https://your-drive-link-here"
   ```

2. The downloader will fetch and extract it automatically on first run.

### Confidence Threshold

Adjust the slider in the sidebar to filter detections by confidence (0.0 - 1.0).

---

## 🌐 Web Deployment

Duuba-AI is optimized for cloud deployment with simplified paths and environment variables. See [WEB_DEPLOYMENT.md](WEB_DEPLOYMENT.md) for detailed instructions.

### Quick Deploy Options:

#### Streamlit Cloud (Recommended for beginners)
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository and add MODEL_URL secret
4. Deploy with one click!

#### Heroku
```bash
heroku create your-app-name
heroku config:set MODEL_URL="your-google-drive-url"
git push heroku main
```

#### Docker
```bash
docker build -t duuba-ai .
docker run -p 8501:8501 -e MODEL_URL="your-url" duuba-ai
```

### Environment Variables
- `MODEL_URL`: URL to download model (Google Drive, S3, etc.)
- `PORT`: Server port (auto-detected on most platforms)

---

## 🐳 Docker Deployment

### Build and Run Locally

```bash
docker build -t duuba-ai .
docker run -p 8501:8501 duuba-ai
```

Visit `http://localhost:8501` in your browser.

### Push to Docker Hub

```bash
docker tag duuba-ai <your-username>/duuba-ai:latest
docker push <your-username>/duuba-ai:latest
```

---

## ☁️ Cloud Deployment

### Streamlit Community Cloud (Recommended)

1. Push your code to GitHub.
2. Go to [Streamlit Cloud](https://streamlit.io/cloud).
3. Click **"New App"** → select your repository.
4. Set main file to `Duuba-AI.py`.
5. Deploy!

### Heroku

```bash
heroku create duuba-ai
git push heroku main
heroku open
```

See `DEPLOY.md` for detailed instructions.

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| Model download fails | Check your internet connection; verify Google Drive link is shared publicly. |
| `IndexError: list index out of range` | Ensure you're running from the repo root directory. |
| Streamlit port already in use | Run `streamlit run Duuba-AI.py --server.port 8502`. |
| No detections on valid images | Lower the **Confidence Threshold** slider; check model is loaded. |

---

## 📊 Model Information

- **Architecture**: SSD MobileNet v2 (lightweight, suitable for mobile/edge devices)
- **Training Data**: ~500 labeled cocoa pod images (healthy & infected)
- **Input Size**: 320×320 pixels (auto-resized)
- **Output**: Bounding boxes + class labels + confidence scores
- **Accuracy**: ~85–87% on test set (varies by pod angle/lighting)

---

## 🛠️ Development

### Install Dev Dependencies

```bash
pip install -r requirements-deploy.txt
pip install jupyter notebook ipykernel
```

### Train a Custom Model

Refer to the included Jupyter notebooks:
- `COCOA Train model.ipynb` — Full training pipeline
- `Training and Detection.ipynb` — Detection examples

### Run Tests

```bash
pytest tests/  # (if test suite exists)
```

---

## 📜 License

This project is licensed under the [MIT License](LICENSE). See the LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`.
3. Commit changes: `git commit -m "Add your feature"`.
4. Push to branch: `git push origin feature/your-feature`.
5. Open a Pull Request.

---

## 📧 Contact

- **Author**: Mubarak Kwanda
- **GitHub**: [github.com/Mkwanda](https://github.com/Mkwanda)
- **LinkedIn**: [Mubarak Kwanda](https://linkedin.com/in/mubarak-kwanda)

---

## 🙏 Acknowledgments

- TensorFlow Object Detection API
- Streamlit for the interactive web framework
- SSD MobileNet v2 pre-trained weights
- Cocoa research and extension communities for data and domain expertise

---

## 📝 Release Notes

### v1.0.0 (December 2025)
- ✅ Initial release with real-time detection
- ✅ Image upload and webcam support
- ✅ Model auto-download from Google Drive
- ✅ Docker and Heroku deployment configs
- ✅ Confidence threshold slider
- ✅ Annotated image export

---

## ⭐ Show Your Support

If this project helps you, please star it on GitHub and share it with others!

Happy detecting! 🌱🍫
