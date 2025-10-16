# evaluate.py - evaluate model on test set
import argparse, pandas as pd, joblib, os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json

def main(test_csv, model_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    clf = joblib.load(model_path)
    df = pd.read_csv(test_csv)
    
    # Use label_encoded (numeric) instead of binary_label (string)
    label_col = 'label_encoded'
    
    # Remove label columns from features
    cols_to_drop = ['binary_label', 'label_encoded']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    y_true = df[label_col]
    
    # Predict
    y_pred = clf.predict(X)
    
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=['attack', 'normal'])
    
    # Save results
    metrics_file = os.path.join(out_dir, "evaluation_metrics.txt")
    with open(metrics_file, 'w') as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(cm))
    
    # Save confusion matrix as JSON for easier parsing
    cm_json = os.path.join(out_dir, "confusion_matrix.json")
    with open(cm_json, 'w') as f:
        json.dump({
            "confusion_matrix": cm.tolist(),
            "accuracy": float(acc),
            "labels": ["attack", "normal"]
        }, f, indent=2)
    
    print(f"✅ Evaluation complete!")
    print(f"📊 Accuracy: {acc:.4f}")
    print(f"📁 Results saved to {out_dir}/")
    print(f"\n{report}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--test_csv', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--out_dir', default='./results')
    args = p.parse_args()
    main(args.test_csv, args.model, args.out_dir)