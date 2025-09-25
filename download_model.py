#!/usr/bin/env python3
"""
Manual model download script for Khmer TrOCR
Use this if the automatic download fails
"""

import os
import sys
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_model():
    """Download the Khmer TrOCR model manually"""
    model_name = "songhieng/khmer-trocr-ocr-v1.0"
    
    try:
        logger.info("🚀 Starting manual model download...")
        logger.info(f"Model: {model_name}")
        logger.info("This may take 10-30 minutes depending on your internet connection")
        
        # Download processor
        logger.info("📥 Downloading processor...")
        processor = TrOCRProcessor.from_pretrained(
            model_name,
            resume_download=True,
            force_download=False
        )
        logger.info("✅ Processor downloaded successfully")
        
        # Download model
        logger.info("📥 Downloading model (this is the large file ~1.5GB)...")
        model = VisionEncoderDecoderModel.from_pretrained(
            model_name,
            resume_download=True,
            force_download=False,
            torch_dtype=torch.float32  # Use float32 for compatibility
        )
        logger.info("✅ Model downloaded successfully")
        
        # Test the model
        logger.info("🧪 Testing model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        # Simple test
        test_input = torch.randn(1, 3, 384, 384).to(device)
        with torch.no_grad():
            _ = model.encoder(test_input)
        
        logger.info(f"✅ Model test successful on {device}")
        logger.info("🎉 Model download and setup complete!")
        logger.info("You can now run the web application with: python app.py")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Download failed: {str(e)}")
        logger.error("💡 Try the following:")
        logger.error("   1. Check your internet connection")
        logger.error("   2. Clear cache: rm -rf ~/.cache/huggingface/hub/models--songhieng--khmer-trocr-ocr-v1.0")
        logger.error("   3. Run this script again")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🇰🇭 Khmer TrOCR Model Download Script")
    print("=" * 60)
    
    success = download_model()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ SUCCESS: Model ready for use!")
        print("Run: python app.py")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("❌ FAILED: Model download unsuccessful")
        print("Please check the error messages above")
        print("=" * 60)
        sys.exit(1)
