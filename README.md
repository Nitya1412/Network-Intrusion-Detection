# Network Intrusion Detection System

Detects whether a single network connection record is benign (normal) or an intrusion (attack).

## How to Run:

1. **Open command prompt** in this folder
2. **Run:** `run_flask.bat`
   - This will create a virtual environment (venv)
   - Install all required dependencies
   - Generate synthetic train/test CSV files
   - Train a machine learning model (rf_model.pkl)
   - Start the Flask server
3. **Open** http://127.0.0.1:5000/ in your browser to use the HTML form

## Requirements:
- Python 3.8 or higher
- If Python is not installed, download from: https://www.python.org/downloads/

## Project Structure:
- `run_flask.bat` - Main setup and execution script
- `app.py` - Flask web application
- `train.py` - Model training script
- `evaluate.py` - Model evaluation script
- `rf_model.pkl` - Pre-trained Random Forest model
- `nsl_kdd_800_15features.csv` - Processed dataset

## Model Performance:
- **Accuracy:** 98.12%
- **Attack Detection Rate:** 97.5%
- **False Positive Rate:** 1.25%

The system uses a Random Forest classifier trained on optimized NSL-KDD dataset for real-time network intrusion detection.


