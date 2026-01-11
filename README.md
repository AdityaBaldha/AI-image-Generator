# AI Image Detector

A machine learning application that detects whether images are AI-generated or real using deep learning and FFT analysis.

## Features

- Upload images (PNG, JPG, JPEG) to classify as AI-generated or real
- Uses hybrid model combining neural networks with FFT (Fast Fourier Transform) analysis
- TorchScript model for optimized inference
- Web-based interface using Streamlit

## Requirements

- Python 3.7+
- PyTorch
- Streamlit
- OpenCV
- NumPy
- Pillow

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AI-image-detector.git
cd AI-image-detector
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run "app (1) (1).py"
```

The app will open in your default browser at `http://localhost:8501`

## How It Works

1. **Image Preprocessing**: Images are resized to 224x224 and normalized
2. **FFT Analysis**: Fast Fourier Transform is applied to extract frequency domain features
3. **Model Inference**: A hybrid deep learning model analyzes both RGB and FFT features
4. **Classification**: Returns prediction of whether image is AI-generated or real

## Model

The app uses a pre-trained TorchScript model (`ai_detector_hybrid.pt`) that combines:
- RGB image features from a neural network
- Frequency domain features from FFT analysis

## License

MIT License

## Author

Created with Python, PyTorch, and Streamlit
