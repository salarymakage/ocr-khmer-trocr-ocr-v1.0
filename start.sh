#!/bin/bash

# Khmer OCR Web Application Startup Script

echo "🚀 Starting Khmer OCR Web Application..."
echo "📍 Working directory: $(pwd)"

# Check if virtual environment exists
if [ ! -d "env" ]; then
    echo "❌ Virtual environment not found. Please run setup first:"
    echo "   python3 -m venv env"
    echo "   source env/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source env/bin/activate

# Check if dependencies are installed
if ! python -c "import transformers" 2>/dev/null; then
    echo "❌ Dependencies not installed. Installing now..."
    pip install -r requirements.txt
fi

echo "📥 Loading Khmer TrOCR model (this may take 5-10 minutes on first run)..."
echo "💡 The model is ~1.5GB and will be cached for future runs"
echo ""
echo "🌐 Once loaded, the application will be available at:"
echo "   http://localhost:5001"
echo ""
echo "⏳ Please wait while the model downloads and loads..."
echo "   (You can monitor progress below)"
echo ""

# Start the application
python app.py
