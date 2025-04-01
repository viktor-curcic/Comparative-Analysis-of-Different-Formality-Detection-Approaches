from sklearn.model_selection import train_test_split
import os
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(current_dir, "../../data/synthetic_formality_data.csv"))

train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

train_df.to_csv(os.path.join(current_dir, "../../data/processed/train.csv"), index=False)
val_df.to_csv(os.path.join(current_dir, "../../data/processed/val.csv"), index=False)
test_df.to_csv(os.path.join(current_dir, "../../data/processed/test.csv"), index=False)
