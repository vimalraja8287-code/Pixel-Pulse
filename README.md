# ParaDetect AI – Deep Learning–Based Malaria Diagnosis

Automated malaria detection from microscopic blood smear images using a CNN. Supports faster, accurate screening and helps healthcare workers in resource-limited settings.

## 🌟 New: Professional Web Interface

ParaDetect AI now includes a complete medical-grade web application with:
- **Modern Medical UI**: Clean, professional interface designed for healthcare environments
- **Drag & Drop Upload**: Intuitive image upload with real-time preview
- **Instant Analysis**: AI-powered results in seconds with confidence scores
- **Multi-page Application**: Main diagnosis, About, and comprehensive Help pages
- **Demo Mode**: Test the interface without requiring a trained model
- **Responsive Design**: Works on desktop, tablet, and mobile devices

### 🚀 Quick Start - Web Interface

```bash
cd paradetect_ai

# Install dependencies
pip install -r requirements.txt

# Start the web application (auto-detects if model is available)
python run_demo.py

# Or use the Windows batch file
start_app.bat
```

Open your browser and go to: **http://localhost:5000**

## Features

- **Parasite detection**: Binary classification (Parasitized vs Uninfected)
- **Image classification**: CNN trained on blood smear cell images
- **Automated diagnosis**: Single-image and batch inference
- **Model accuracy analysis**: Precision, recall, F1, AUC, confusion matrix
- **Healthcare support**: Batch report for screening workflows
- **Web Interface**: Professional medical-grade frontend for easy use

## Technologies

- **Python 3.8+**
- **TensorFlow & Keras** – CNN training and inference
- **Flask** – Web application framework
- **OpenCV** – Image loading and preprocessing
- **scikit-learn** – Evaluation metrics
- **Modern Web Stack** – HTML5, CSS3, JavaScript

## Project structure

```
paradetect_ai/
├── config.py           # Paths and hyperparameters
├── requirements.txt
├── train.py            # Train the model
├── predict.py          # Run diagnosis (single image or folder)
├── evaluate.py         # Accuracy analysis
├── healthcare_report.py # Batch screening report
├── app.py              # Web application (production)
├── demo_app.py         # Web application (demo mode)
├── run_demo.py         # Quick start script
├── start_app.bat       # Windows startup script
├── test_frontend.py    # Frontend testing script
├── templates/          # Web interface templates
│   ├── index.html      # Main diagnosis page
│   ├── about.html      # About page
│   └── help.html       # Help and FAQ page
├── static/             # Web assets (CSS, JS)
├── data/
│   └── cell_images/    # Put dataset here (see below)
│       ├── Parasitized/
│       └── Uninfected/
├── models/             # Saved .keras models
├── results/            # History, reports, plots
└── src/
    ├── data_loader.py  # Dataset loading
    ├── preprocess.py   # OpenCV preprocessing
    └── model.py        # CNN definition
```

## Dataset

Use the **Malaria Cell Images** dataset (Parasitized vs Uninfected):

1. **Source**: [NIH Malaria Dataset](https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#malaria-datasets) or [Kaggle – Cell Images for Detecting Malaria](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria).
2. **Google Drive** (reference): [Link](https://drive.google.com/file/d/1j14PwR48HjmOtcR96mfYvdpsWKs2GgfP/view) – download and unzip.

Required layout:

- `paradetect_ai/data/cell_images/Parasitized/` – infected cell images  
- `paradetect_ai/data/cell_images/Uninfected/` – uninfected cell images  

Supported formats: `.png`, `.jpg`, `.jpeg`.

## Setup

```bash
cd paradetect_ai
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

## Usage

### 🌐 Web Interface (Recommended)

**Quick Start:**
```bash
python run_demo.py
```
Then open http://localhost:5000 in your browser.

**Features:**
- Upload blood smear images via drag & drop
- Get instant AI analysis results
- View confidence scores and probability distributions
- Professional medical interface
- Works with or without trained model (demo mode)

### 📊 Command Line Interface

#### 1. Train the model

```bash
python train.py
# Options: --data-dir, --epochs, --batch-size, --save-name
```

Best model is saved under `models/paradetect_<timestamp>.keras`.

#### 2. Run diagnosis

**Single image:**

```bash
python predict.py --model models/paradetect_20240224_1200.keras --image path/to/cell.png
```

**Folder (batch):**

```bash
python predict.py --model models/paradetect_20240224_1200.keras --folder path/to/cells/ --output results.csv
```

#### 3. Model accuracy analysis

```bash
python evaluate.py --model models/paradetect_20240224_1200.keras --data-dir data/cell_images
```

Writes `results/evaluation_report.json` and `results/confusion_matrix.png`.

#### 4. Healthcare batch report

```bash
python healthcare_report.py --model models/paradetect_20240224_1200.keras --folder path/to/smear_images/ --output report.txt
```

## 🧪 Testing

Test the web interface:
```bash
# Start the server first
python run_demo.py

# In another terminal, run tests
python test_frontend.py
```

## 📱 Web Interface Features

### Main Diagnosis Page
- Professional medical-grade design
- Drag & drop image upload
- Real-time image preview
- Instant AI analysis results
- Confidence visualization
- Processing time metrics

### About Page
- Detailed system information
- Performance metrics
- Technology stack overview
- Model training details

### Help Page
- Comprehensive user guide
- FAQ section
- Image requirements
- Best practices
- Medical disclaimers

## Disclaimer

This tool is for **research and screening support** only. It does not replace clinical diagnosis. Always confirm results with qualified healthcare professionals and standard protocols.

## References

- NIH Malaria Datasets: https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#malaria-datasets  
- Kaggle – Cell Images for Detecting Malaria  
- Google Drive reference: https://drive.google.com/file/d/1j14PwR48HjmOtcR96mfYvdpsWKs2GgfP/view
