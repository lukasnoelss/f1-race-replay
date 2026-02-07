import joblib
import pandas as pd

# 1. Load the trained "Brain"
model = joblib.load('undercut_predictor.pkl')

def predict_undercut():
    print("--- 🏎️ F1 STRATEGY PREDICTOR ---")
    
    # Get inputs from the user
    gap = float(input("Current Gap (e.g., 1.2): "))
    tire_delta = float(input("Tire Age Delta (Attacker - Defender, e.g. -5): "))
    traffic = float(input("Defender Traffic Gap (Clean air = 10): "))
    att_pit = float(input("Estimated Attacker Pit (e.g., 24.5): "))
    def_pit = float(input("Estimated Defender Pit (e.g., 24.5): "))
    temp = float(input("Track Temp (e.g., 35.0): "))
    att_h = int(input("Attacker Tire (1=Soft, 3=Hard): "))
    def_h = int(input("Defender Tire (1=Soft, 3=Hard): "))

    # Prepare data for model
    scenario = [[gap, tire_delta, traffic, att_pit, def_pit, temp, att_h, def_h]]
    
    # Get the prediction
    # model.predict_proba gives us the % chance (e.g. 0.82)
    proba = model.predict_proba(scenario)[0][1] 
    
    print(f"\n--- 📊 PREDICTION RESULT ---")
    print(f"Probability of Success: {proba*100:.1f}%")
    
    if proba > 0.65:
        print("✅ BOX NOW! High probability of success.")
    elif proba > 0.45:
        print("⚠️ MARGINAL. risky move.")
    else:
        print("❌ STAY OUT. Likely to fail.")

if __name__ == "__main__":
    predict_undercut()