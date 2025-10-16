# split_data.py - Split dataset into train and test sets
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    # Load your full dataset
    df = pd.read_csv('nsl_kdd_800_15features.csv')
    
    # Split: 80% train (640 samples), 20% test (160 samples)
    train_df, test_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42,
        stratify=df['label_encoded']  # Keep same class distribution
    )
    
    # Save to separate files
    train_df.to_csv('train_data.csv', index=False)
    test_df.to_csv('test_data.csv', index=False)
    
    print("✅ Data split complete!")
    print(f"📊 Training set: {len(train_df)} samples")
    print(f"   - Attacks: {(train_df['label_encoded'] == 0).sum()}")
    print(f"   - Normal: {(train_df['label_encoded'] == 1).sum()}")
    print(f"\n📊 Test set: {len(test_df)} samples")
    print(f"   - Attacks: {(test_df['label_encoded'] == 0).sum()}")
    print(f"   - Normal: {(test_df['label_encoded'] == 1).sum()}")
    print(f"\n✅ Saved: train_data.csv and test_data.csv")

if __name__ == '__main__':
    main()