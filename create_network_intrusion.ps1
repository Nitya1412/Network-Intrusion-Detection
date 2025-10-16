# create_network_intrusion.ps1 - automate full workflow for NSL-KDD dataset

Write-Host "=== Starting NSL-KDD Intrusion Detection Pipeline ===" -ForegroundColor Cyan

# Step 0: Split data into train and test (run once)
if (-not (Test-Path "train_data.csv") -or -not (Test-Path "test_data.csv")) {
    Write-Host "`n[Step 0] Splitting dataset into train/test..." -ForegroundColor Yellow
    python split_data.py
    if ($LASTEXITCODE -ne 0) { Write-Host "❌ Data split failed" -ForegroundColor Red; exit 1 }
} else {
    Write-Host "`n[Step 0] Train/test split already exists (skipping)" -ForegroundColor Green
}

# Step 1: Train on training data
Write-Host "`n[Step 1] Training model..." -ForegroundColor Yellow
python train.py --data_csv "train_data.csv" --model_out "rf_model.pkl"
if ($LASTEXITCODE -ne 0) { Write-Host "❌ Training failed" -ForegroundColor Red; exit 1 }

# Step 2: Evaluate on separate test data
Write-Host "`n[Step 2] Evaluating on test data..." -ForegroundColor Yellow
python evaluate.py --test_csv "test_data.csv" --model "rf_model.pkl" --out_dir "./results"
if ($LASTEXITCODE -ne 0) { Write-Host "❌ Evaluation failed" -ForegroundColor Red; exit 1 }

# Step 3: Show summary
Write-Host "`n=== Evaluation Metrics ===" -ForegroundColor Cyan
Get-Content "./results/evaluation_metrics.txt"

Write-Host "`n✅ Pipeline complete!" -ForegroundColor Green
Write-Host "📁 Model saved: rf_model.pkl" -ForegroundColor Green
Write-Host "📁 Results saved: ./results/" -ForegroundColor Green

# Step 4: Start Flask App (optional)
# Write-Host "`nStarting Flask API..." -ForegroundColor Yellow
# python app.py