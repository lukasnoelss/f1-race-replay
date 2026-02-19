import pandas as pd
import numpy as np

def sanitize_dataset():
    print("🧹 SANITIZING DATASET (Fixing Look-Ahead Bias)...")
    
    df = pd.read_csv('undercut_dataset.csv')
    
    # We need to calculate the average pit duration per TRACK and YEAR (to be safe)
    # or just per Track if we assume pit lanes don't change much.
    # Let's do it per Track.
    
    track_avgs = df[df['Att_Pit_Duration'] < 50].groupby('Track')['Att_Pit_Duration'].mean().to_dict()
    
    def get_avg(track):
        return round(track_avgs.get(track, 24.5), 3)

    # 1. Replace actuals with track averages
    df['Att_Pit_Duration'] = df['Track'].map(get_avg)
    df['Def_Pit_Duration'] = df['Track'].map(get_avg)
    
    # 2. Reset Pit_Delta to 0 (Assumption: Clean stop for both)
    df['Pit_Delta'] = 0.0
    
    # 3. Save it back
    df.to_csv('undercut_dataset.csv', index=False)
    print(f"✅ Success! Corrected {len(df)} samples by assuming standard pit stops.")
    print("Model will now focus on Gaps and Tires instead of mechanical luck.")

if __name__ == "__main__":
    sanitize_dataset()
