# Khmer OCR Web Interface 📝🇰🇭

A modern web interface for Khmer text recognition using the fine-tuned TrOCR model from [songhieng/khmer-trocr-ocr-v1.0](https://huggingface.co/songhieng/khmer-trocr-ocr-v1.0).

## Features

- 🎯 **Specialized Khmer Recognition**: Uses a fine-tuned TrOCR model specifically trained for Khmer script
- 🚀 **Fast Processing**: Quick text extraction with GPU acceleration support
- 📱 **Responsive Design**: Works on desktop and mobile devices
- 🎨 **Modern UI**: Beautiful, intuitive interface with drag-and-drop functionality
- 🔒 **Privacy Focused**: Images are processed locally and not stored
- 📋 **Easy Copy**: One-click text copying functionality

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-compatible GPU for faster processing

### Setup

1. **Clone or download the project**:
   ```bash
   cd /home/hello/thareah/uiocr
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   ./start.sh
   ```
   
   Or manually:
   ```bash
   source env/bin/activate
   python app.py
   ```

   **⚠️ Important**: On first run, the application will download the Khmer TrOCR model (~1.5GB). This may take 5-10 minutes depending on your internet connection. The model will be cached for future runs.

5. **Open your browser** and navigate to:
   ```
   http://localhost:5001
   ```

## Usage

1. **Upload an Image**: 
   - Drag and drop an image file onto the upload area, or
   - Click "Choose Image" to select a file from your device

2. **Extract Text**: 
   - Click "Extract Text" to process the image
   - Wait for the AI model to analyze your image

3. **View Results**: 
   - The extracted Khmer text will appear below the image
   - Click "Copy Text" to copy the results to your clipboard

## Supported Image Formats

- PNG
- JPG/JPEG
- GIF
- BMP
- Maximum file size: 16MB

## API Endpoints

### POST /api/ocr
Upload an image file for OCR processing.

**Request**: Multipart form data with `image` field
**Response**: 
```json
{
  "success": true,
  "text": "ខ្មែរ",
  "language": "Khmer"
}
```

### POST /api/ocr-base64
Process base64-encoded image data.

**Request**: 
```json
{
  "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
}
```

### GET /health
Check application health and model status.

**Response**: 
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

## Technical Details

### Model Information
- **Base Model**: Microsoft TrOCR (microsoft/trocr-base-stage1)
- **Fine-tuned Model**: songhieng/khmer-trocr-ocr-v1.0
- **Architecture**: Vision Encoder-Decoder (ViT + RoBERTa)
- **Specialization**: Khmer script recognition
- **Input Size**: 512x64 RGB images (auto-resized)

### Performance
- **CPU Processing**: ~2-5 seconds per image
- **GPU Processing**: ~0.5-1 second per image (with CUDA)
- **Memory Usage**: ~2GB RAM (model loading)

## Troubleshooting

### Model Loading Issues
If you see "Model is loading..." error:
1. **First Run**: Wait 5-10 minutes for the model to download (~1.5GB)
2. **Slow Internet**: The download may take longer on slower connections
3. **Check Logs**: Monitor the terminal for download progress
4. **Refresh Page**: Once "Model loaded successfully" appears, refresh the browser
5. **Check Internet**: Ensure stable internet connection for model download

### CUDA/GPU Issues
If GPU is not detected but you have CUDA installed:
1. Verify PyTorch CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
2. Install CUDA-compatible PyTorch: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

### Memory Issues
If you encounter out-of-memory errors:
1. Close other applications to free RAM
2. Process smaller images
3. Use CPU-only mode (automatic fallback)

## Development

### Project Structure
```
uiocr/
├── app.py              # Flask application
├── templates/
│   └── index.html      # Web interface
├── requirements.txt    # Dependencies
└── README.md          # This file
```

### Running in Development Mode
```bash
export FLASK_ENV=development
python app.py
```

### Production Deployment
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## License

This project uses the Apache-2.0 licensed model from Hugging Face. Free for research and commercial use.

## Acknowledgments

- [Microsoft TrOCR](https://github.com/microsoft/unilm/tree/master/trocr) for the base architecture
- [songhieng/khmer-trocr-ocr-v1.0](https://huggingface.co/songhieng/khmer-trocr-ocr-v1.0) for the fine-tuned Khmer model
- [Hugging Face Transformers](https://huggingface.co/transformers/) for the model framework
- Khmer Unicode fonts project for text rendering support

## Support

For issues related to:
- **Web Interface**: Create an issue in this repository
- **Model Performance**: Check the [original model page](https://huggingface.co/songhieng/khmer-trocr-ocr-v1.0)
- **TrOCR Framework**: See [Microsoft TrOCR documentation](https://github.com/microsoft/unilm/tree/master/trocr)
