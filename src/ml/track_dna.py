import pandas as pd
import joblib
import os

def calculate_track_dna():
    print("CRUNCHING TRACK DNA (100% Data-Driven)...")
    
    csv_path = 'undercut_dataset.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please run data collection first.")
        return

    # 1. Load the historical data
    df = pd.read_csv(csv_path)
    
    # 2. Filter out unknowns
    df = df[df['Outcome'] != 'unknown']
    
    # 3. Create a Success Binary
    df['Success_Binary'] = (df['Outcome'] == 'success').astype(int)
    
    # 4. Group by Track and calculate the mean success rate AND average pit duration
    # This is our objective "Track DNA" profile.
    track_stats = df.groupby('Track').agg({
        'Success_Binary': ['mean', 'count'],
        'Att_Pit_Duration': 'mean' 
    }).reset_index()
    
    # Flatten columns
    track_stats.columns = ['Track', 'Track_Undercut_Index', 'Sample_Size', 'Avg_Pit_Duration']
    
    # 5. Handle low-sample tracks (regress toward the global average)
    global_mean = df['Success_Binary'].mean()
    global_pit_avg = df['Att_Pit_Duration'].mean()

    track_stats['Track_Undercut_Index'] = (
        (track_stats['Track_Undercut_Index'] * track_stats['Sample_Size']) + (global_mean * 5)
    ) / (track_stats['Sample_Size'] + 5)
    
    # 6. Save as a dictionary of dictionaries for the model and tool
    dna_dict = {}
    for _, row in track_stats.iterrows():
        dna_dict[row['Track']] = {
            'index': row['Track_Undercut_Index'],
            'avg_pit': row['Avg_Pit_Duration']
        }
    
    joblib.dump(dna_dict, 'track_dna.pkl')
    
    print("\n--- TRACK DNA REPORT ---")
    print(f"{'Track':30} | Undercut Index | Avg Pit Duration")
    print("-" * 65)
    for track, data in dna_dict.items():
        print(f"{track:30} | {data['index']:.3f}          | {data['avg_pit']:.2f}s")
        
    print(f"\n💾 Saved {len(dna_dict)} track profiles to 'track_dna.pkl'")
    return dna_dict

if __name__ == "__main__":
    calculate_track_dna()
