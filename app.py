from flask import Flask, request, jsonify, render_template
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import io
import base64
import os
from werkzeug.utils import secure_filename
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables for model and processor
model = None
processor = None
device = None

def load_model():
    """Load the Khmer TrOCR model and processor"""
    global model, processor, device
    
    try:
        logger.info("Loading Khmer TrOCR model...")
        model_name = "songhieng/khmer-trocr-ocr-v1.0"
        
        # Set device first
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Load processor and model with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Download attempt {attempt + 1}/{max_retries}")
                
                # Load processor
                logger.info("Loading processor...")
                processor = TrOCRProcessor.from_pretrained(
                    model_name,
                    resume_download=True,
                    force_download=False
                )
                
                # Load model
                logger.info("Loading model...")
                model = VisionEncoderDecoderModel.from_pretrained(
                    model_name,
                    resume_download=True,
                    force_download=False,
                    torch_dtype=torch.float32 if device == "cpu" else torch.float16
                )
                
                # Move to device
                model = model.to(device)
                
                # Test the model with a simple forward pass
                logger.info("Testing model...")
                test_input = torch.randn(1, 3, 384, 384).to(device)
                with torch.no_grad():
                    _ = model.encoder(test_input)
                
                logger.info(f"Model loaded and tested successfully on {device}")
                return True
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    logger.info("Retrying in 10 seconds...")
                    import time
                    time.sleep(10)
                else:
                    logger.error("All download attempts failed")
                    raise e
        
        return False
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error("Please check your internet connection and try again")
        return False

def process_image(image):
    """Process image and return OCR results"""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Process image
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)
        
        # Generate prediction
        with torch.no_grad():
            generated_ids = model.generate(pixel_values, max_length=256)
            predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return predicted_text.strip()
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise e

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/ocr', methods=['POST'])
def ocr_endpoint():
    """OCR API endpoint"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Check if file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file type
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            return jsonify({'error': 'Invalid file type. Please upload an image file.'}), 400
        
        # Read and process image
        image_data = file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Process OCR
        predicted_text = process_image(image)
        
        return jsonify({
            'success': True,
            'text': predicted_text,
            'language': 'Khmer'
        })
        
    except Exception as e:
        logger.error(f"OCR endpoint error: {str(e)}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/api/ocr-base64', methods=['POST'])
def ocr_base64_endpoint():
    """OCR API endpoint for base64 images"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        try:
            # Remove data URL prefix if present
            image_data = data['image']
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            # Decode base64
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
        except Exception as e:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Process OCR
        predicted_text = process_image(image)
        
        return jsonify({
            'success': True,
            'text': predicted_text,
            'language': 'Khmer'
        })
        
    except Exception as e:
        logger.error(f"OCR base64 endpoint error: {str(e)}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': device
    })

if __name__ == '__main__':
    # Load model on startup
    if load_model():
        logger.info("Starting Flask application...")
        app.run(host='0.0.0.0', port=5001, debug=True)
    else:
        logger.error("Failed to load model. Exiting...")
        exit(1)
