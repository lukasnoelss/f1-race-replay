import joblib
import pandas as pd
import numpy as np

# 1. Load the "Brain" and the "Feature Memory"
model = joblib.load('undercut_predictor.pkl')
all_features = joblib.load('model_features.pkl')
track_dna = joblib.load('track_dna.pkl')

def predict_undercut():
    print("\n--- 🏎️ F1 STRATEGY PREDICTOR V4 (Standard Stop Mode) ---")
    
    # 1. Track Selection First (to auto-fill pit durations)
    print("\nAvailable Tracks examples: 'Bahrain Grand Prix', 'Monaco Grand Prix', 'Spanish Grand Prix'")
    track_input = input("Enter Track Name: ").strip()
    
    # Lookup DNA
    dna = track_dna.get(track_input, {'index': 0.45, 'avg_pit': 24.5})
    if track_input in track_dna:
        print(f"📍 Track DNA Loaded: Index {dna['index']:.3f}, Avg Pit {dna['avg_pit']:.1f}s")
    else:
        print("⚠️ Track not found. Using global average profiles.")

    # 2. Get tactical inputs
    gap = float(input("Current Gap (e.g., 1.2): "))
    tire_delta = float(input("Tire Age Delta (Attacker - Defender, e.g. -5): "))
    traffic = float(input("Defender Traffic Gap (Clean air = 10): "))
    temp = float(input("Track Temp (e.g., 35.0): "))
    att_h = int(input("Attacker Tire (1=Soft, 3=Hard): "))
    def_h = int(input("Defender Tire (1=Soft, 3=Hard): "))
    progress = float(input("Race Progress (0.0 to 1.0, e.g. 0.5 for mid-race): "))

    # 3. Build the exact DataFrame the model expects
    scenario = pd.DataFrame(0, index=[0], columns=all_features)
    
    scenario['Gap_Before'] = gap
    scenario['Tire_Delta'] = tire_delta
    scenario['Def_Traffic_Gap'] = traffic
    scenario['Att_Pit_Duration'] = dna['avg_pit']
    scenario['Pit_Delta'] = 0.0 # Assuming clean stops for both
    scenario['Track_Temp'] = temp
    scenario['Att_Hardness'] = att_h
    scenario['Def_Hardness'] = def_h
    scenario['Race_Progress'] = progress
    scenario['Track_Undercut_Index'] = dna['index']
    
    # 4. Predict!
    proba = model.predict_proba(scenario)[0][1] 
    
    print(f"\n--- 📊 PREDICTION RESULT ---")
    print(f"Probability of Success: {proba*100:.1f}%")
    
    if proba > 0.65:
        print("✅ BOX NOW! Strong tactical advantage.")
    elif proba > 0.45:
        print("⚠️ MARGINAL. Risky move, depends on perfect execution.")
    else:
        print("❌ STAY OUT. High risk of failure at this track.")

if __name__ == "__main__":
    predict_undercut()