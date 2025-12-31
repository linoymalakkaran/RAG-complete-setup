#!/bin/bash
# Quick start script for RAG project (Mac/Linux)
# Run this to set up and start the project

echo "========================================"
echo "RAG Project Quick Start"
echo "========================================"
echo

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "Error creating virtual environment!"
        exit 1
    fi
    echo "Virtual environment created!"
    echo
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo

# Install dependencies
echo "Checking dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error installing dependencies!"
    exit 1
fi
echo "Dependencies installed!"
echo

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo
    echo "⚠️  IMPORTANT: Edit .env file and add your API keys!"
    echo "   - OPENAI_API_KEY (required for embeddings and LLM)"
    echo
    read -p "Press enter to continue..."
fi

# Run setup verification
echo "Running setup verification..."
python setup_verify.py
if [ $? -ne 0 ]; then
    echo
    echo "⚠️  Setup verification found issues. Please fix them before continuing."
    exit 1
fi
echo

# Ask user what to start
echo "========================================"
echo "What would you like to run?"
echo "========================================"
echo "1. Streamlit UI (recommended for beginners)"
echo "2. Jupyter Notebooks (recommended for learning)"
echo "3. Setup only (I'll run manually later)"
echo
read -p "Enter choice (1-3): " choice

if [ "$choice" = "1" ]; then
    echo
    echo "Starting Streamlit UI..."
    echo "Open http://localhost:8501 in your browser"
    echo
    streamlit run ui/app.py
elif [ "$choice" = "2" ]; then
    echo
    echo "Starting Jupyter..."
    echo
    jupyter notebook notebooks/
else
    echo
    echo "Setup complete! You can now:"
    echo "  - Run UI: streamlit run ui/app.py"
    echo "  - Run notebooks: jupyter notebook"
    echo
fi
