# Sentiment Studio — AI-Based Sentiment Analyzer

## Run on Windows with VS Code
1. Extract the ZIP and open this folder in VS Code.
2. In the terminal run:
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   python app.py
   ```
3. Visit http://127.0.0.1:5000 in your browser.
4. The model downloads the first time it is used, so allow time and keep the internet connection on.

If PowerShell activation is blocked, use Command Prompt and run `.venv\\Scripts\\activate.bat`.

## Features
- Single-review sentiment with friendly emoji feedback
- Bulk analysis of up to 100 reviews
- Circular doughnut chart for Positive / Neutral / Negative distribution
- Percentage breakdown and per-review results table
- Optional MongoDB storage via the `MONGO_URI` environment variable

The model score is not a guarantee of correctness. Sarcasm, mixed opinions, slang and domain-specific text can cause errors. This is an educational demo; do not expose Flask's development server publicly.
