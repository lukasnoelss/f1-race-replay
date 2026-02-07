import joblib
import pandas as pd
import numpy as np

# 1. Load the "Brain" and the "Feature Memory"
model = joblib.load('undercut_predictor.pkl')
all_features = joblib.load('model_features.pkl')

def predict_undercut():
    print("\n--- 🏎️ F1 STRATEGY PREDICTOR V3 (Pro-Level) ---")
    
    # Get inputs
    gap = float(input("Current Gap (e.g., 1.2): "))
    tire_delta = float(input("Tire Age Delta (Attacker - Defender, e.g. -5): "))
    traffic = float(input("Defender Traffic Gap (Clean air = 10): "))
    att_pit = float(input("Estimated Attacker Pit (e.g., 24.5): "))
    def_pit = float(input("Estimated Defender Pit (e.g., 24.5): "))
    temp = float(input("Track Temp (e.g., 35.0): "))
    att_h = int(input("Attacker Tire (1=Soft, 3=Hard): "))
    def_h = int(input("Defender Tire (1=Soft, 3=Hard): "))
    
    # --- Part B Enhancement: Track Selection ---
    print("\nAvailable Tracks examples: 'Bahrain Grand Prix', 'Monaco Grand Prix', 'Spanish Grand Prix'")
    track_input = input("Enter Track Name (or press enter for average): ").strip()
    
    # 2. Build the exact DataFrame the model expects
    # Start with all Zeros
    scenario = pd.DataFrame(0, index=[0], columns=all_features)
    
    # Fill in numeric values
    scenario['Gap_Before'] = gap
    scenario['Tire_Delta'] = tire_delta
    scenario['Def_Traffic_Gap'] = traffic
    scenario['Att_Pit_Duration'] = att_pit
    scenario['Def_Pit_Duration'] = def_pit
    scenario['Pit_Delta'] = att_pit - def_pit # <--- Calculated automatically!
    scenario['Track_Temp'] = temp
    scenario['Att_Hardness'] = att_h
    scenario['Def_Hardness'] = def_h
    
    # One-Hot Encode the chosen track
    track_col = f"Track_{track_input}"
    if track_col in all_features:
        scenario[track_col] = 1
        print(f"📍 Applying track-specific rules for: {track_input}")
    else:
        print("⚠️ Track not found in database. Using general average rules.")

    # 3. Predict!
    proba = model.predict_proba(scenario)[0][1] 
    
    print(f"\n--- 📊 PREDICTION RESULT ---")
    print(f"Probability of Success: {proba*100:.1f}%")
    
    if proba > 0.65:
        print("✅ BOX NOW! Strong tactical advantage.")
    elif proba > 0.45:
        print("⚠️ MARGINAL. risky move, depends on driver skill.")
    else:
        print("❌ STAY OUT. Defender has the upper hand.")

if __name__ == "__main__":
    predict_undercut()