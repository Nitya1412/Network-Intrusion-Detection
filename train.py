 # train.py - train RandomForest model on NSL-KDD dataset
import argparse, pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def main(data_csv, model_out):
    df = pd.read_csv(data_csv)

    # Use label_encoded (numeric) instead of binary_label (string)
    # Drop both label columns from features
    label_col = 'label_encoded'
    
    # Remove label columns from features
    cols_to_drop = ['binary_label', 'label_encoded']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    y = df[label_col]

    print(f"✅ Features: {X.shape[1]} columns")
    print(f"✅ Target distribution:\n{y.value_counts()}")

    # Split for internal validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Calculate validation accuracy
    val_acc = clf.score(X_val, y_val)

    # Save model
    joblib.dump(clf, model_out)
    print(f"\n✅ Model trained and saved to {model_out}")
    print(f"📊 Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    print(f"🎯 Validation Accuracy: {val_acc:.4f}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_csv', required=True)
    p.add_argument('--model_out', default='rf_model.pkl')
    args = p.parse_args()
    main(args.data_csv, args.model_out)