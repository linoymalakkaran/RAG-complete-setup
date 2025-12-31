@echo off
REM Quick start script for RAG project (Windows)
REM Run this to set up and start the project

echo ========================================
echo RAG Project Quick Start
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo Error creating virtual environment!
        pause
        exit /b 1
    )
    echo Virtual environment created!
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Install dependencies
echo Checking dependencies...
pip install -q --upgrade pip
pip install -q -r requirements.txt
if errorlevel 1 (
    echo Error installing dependencies!
    pause
    exit /b 1
)
echo Dependencies installed!
echo.

REM Check if .env exists
if not exist ".env" (
    echo Creating .env file from template...
    copy .env.example .env
    echo.
    echo ⚠️  IMPORTANT: Edit .env file and add your API keys!
    echo    - OPENAI_API_KEY (required for embeddings and LLM)
    echo.
    pause
)

REM Run setup verification
echo Running setup verification...
python setup_verify.py
if errorlevel 1 (
    echo.
    echo ⚠️  Setup verification found issues. Please fix them before continuing.
    pause
    exit /b 1
)
echo.

REM Ask user what to start
echo ========================================
echo What would you like to run?
echo ========================================
echo 1. Streamlit UI (recommended for beginners)
echo 2. Jupyter Notebooks (recommended for learning)
echo 3. Setup only (I'll run manually later)
echo.
set /p choice="Enter choice (1-3): "

if "%choice%"=="1" (
    echo.
    echo Starting Streamlit UI...
    echo Open http://localhost:8501 in your browser
    echo.
    streamlit run ui\app.py
) else if "%choice%"=="2" (
    echo.
    echo Starting Jupyter...
    echo.
    jupyter notebook notebooks\
) else (
    echo.
    echo Setup complete! You can now:
    echo   - Run UI: streamlit run ui\app.py
    echo   - Run notebooks: jupyter notebook
    echo.
    pause
)
