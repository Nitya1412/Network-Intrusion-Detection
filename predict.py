# predict.py - predict labels for unlabeled data
import joblib, pandas as pd
import argparse

def main(model_path, input_csv, out_csv):
    clf = joblib.load(model_path)
    df = pd.read_csv(input_csv)
    preds = clf.predict(df)
    df['predicted_label'] = preds
    df.to_csv(out_csv, index=False)
    print(f"✅ Predictions saved to {out_csv}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--input', required=True)
    p.add_argument('--out', default='predictions.csv')
    args = p.parse_args()
    main(args.model, args.input, args.out)
