import numpy as np # numerical ops
import pandas as pd # data handling
from sklearn.model_selection import train_test_split # splitting data
import os # file paths

# Splits dataset into train, validation, and test sets
def split_dataset(dataset_path, output_dir="splits", test_size=0.2, val_size=0.2, random_state=42):
    """
    Splits dataset into training, validation, and test sets.
    
    Inputs:
        dataset_path: Path to the combined dataset CSV
        output_dir: Where to save the split datasets
        test_size: Fraction of data for test set (default: 0.2)
        val_size: Fraction of training data for validation (default: 0.2)
        random_state: For reproducible splits
    """
    # Load dataset
    df = pd.read_csv(dataset_path)
    
    # Split 1: Create test set (20% of data)
    # This set will be used for final evaluation only
    train_val, test = train_test_split(
        df, 
        test_size=test_size,  # 20% for test
        random_state=random_state,  # for reproducible splits
        stratify=df['label']  # keep same ratio of gunshots/non-gunshots
    )
    
    # Split 2: Create validation set (20% of remaining 80%)
    # This gives us roughly 60% train, 20% val, 20% test
    train, val = train_test_split(
        train_val,
        test_size=val_size,  # 20% of remaining 80% for validation
        random_state=random_state,
        stratify=train_val['label']  # keep class balance
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each split with its purpose
    train.to_csv(f"{output_dir}/train.csv", index=False)  # for teaching the model
    val.to_csv(f"{output_dir}/val.csv", index=False)      # for checking during training
    test.to_csv(f"{output_dir}/test.csv", index=False)    # for final evaluation
    
    # Print dataset statistics
    print(f"Total samples: {len(df)}")
    print(f"Training set: {len(train)} samples")    # 60% of data
    print(f"Validation set: {len(val)} samples")    # 20% of data
    print(f"Test set: {len(test)} samples")         # 20% of data
    
    # Show class distribution in each split
    print("\nClass distribution:")
    for split_name, split_df in [("Train", train), ("Val", val), ("Test", test)]:
        print(f"\n{split_name}:")
        print(split_df['label'].value_counts())  # count samples per class

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python data_split.py path_to_dataset.csv")
    else:
        split_dataset(sys.argv[1]) 