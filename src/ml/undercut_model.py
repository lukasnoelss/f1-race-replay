import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Load the data
df = pd.read_csv('undercut_dataset.csv')


features = [
    'Gap_Before', 
    'Tire_Delta', 
    'Def_Traffic_Gap', 
    'Att_Pit_Duration', 
    'Def_Pit_Duration', 
    'Track_Temp', 
    'Att_Hardness', 
    'Def_Hardness'
]

df = df[df['Outcome'] != 'unknown']
df = df.dropna(subset=features + ['Outcome'])
df['Target'] = df['Outcome'].map({'success': 1, 'fail': 0})
X = df[features]

y = df['Target']

print("Features (X) head:")
print(X.head())
print("\nTarget (y) head:")
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Total samples: {len(X)}")
print(f"Training set size (Study material): {len(X_train)}")
print(f"Testing set size (Final Exam): {len(X_test)}")

# Random Forest 
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n Training Complete!")
print(f" Model Accuracy: {accuracy * 100:.1f}%")

# Feature Importance
print("\n--- THE STRATEGIST'S KEY CLUES (Feature Importance) ---")
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print(importances)

test_scenario = [[0.8, -10, 5.0, 24.5, 24.5, 30.0, 2, 2]]
prediction = model.predict(test_scenario)
result = "SUCCESS" if prediction[0] == 1 else "FAIL"
print(f"\nPREDICTION TEST:")
print(f"Scenario: Gap 0.8s, Attacker tires 10 laps newer")
print(f"Model says: {result}")

joblib.dump(model, 'undercut_predictor.pkl')
print("\n💾 Model saved as 'undercut_predictor.pkl'")