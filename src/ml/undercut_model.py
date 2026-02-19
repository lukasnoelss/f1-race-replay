import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Load the data and DNA
df = pd.read_csv('undercut_dataset.csv')
track_dna = joblib.load('track_dna.pkl')

# Inject Track DNA Features
df['Track_Undercut_Index'] = df['Track'].map(lambda x: track_dna.get(x, {}).get('index', 0.5))
# Note: Avg_Pit_Duration is already in the CSV as Att_Pit_Duration since we sanitized it!

features = [
    'Gap_Before', 
    'Tire_Delta', 
    'Def_Traffic_Gap', 
    'Att_Pit_Duration', 
    'Pit_Delta', # Now 0.0 in sanitized data
    'Track_Temp', 
    'Att_Hardness', 
    'Def_Hardness',
    'Race_Progress',
    'Track_Undercut_Index'
]

df = df[df['Outcome'] != 'unknown']
df = df.dropna(subset=features + ['Outcome'])
df['Target'] = df['Outcome'].map({'success': 1, 'fail': 0})

X = df[features]
y = df['Target']
all_features = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Total samples: {len(X)}")
print(f"Training set size (Study material): {len(X_train)}")
print(f"Testing set size (Final Exam): {len(X_test)}")

# Using the reliable V3 Random Forest brain
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

print("Running Automated V3.5 Random Forest Tuning...")
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

model = grid_search.best_estimator_
print(f"\nBest Settings Found: {grid_search.best_params_}")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n Training Complete!")
print(f" Model Accuracy: {accuracy * 100:.1f}%")

# Feature Importance
print("\n--- THE STRATEGIST'S KEY CLUES (Feature Importance) ---")
importances = pd.Series(model.feature_importances_, index=all_features).sort_values(ascending=False)
print(importances.head(10)) # Show top 10

# 6. Test Scenario (Building a realistic "Pro" input)
print(f"\nPREDICTION TEST:")
# To test, we create a DataFrame with 1 row so the columns match exactly
test_sample = pd.DataFrame(0, index=[0], columns=all_features)
test_sample['Gap_Before'] = 0.8
test_sample['Tire_Delta'] = -10
test_sample['Def_Traffic_Gap'] = 5.0

# Bahrain DNA Lookup
bahrain_dna = track_dna.get('Bahrain Grand Prix', {'index': 0.5, 'avg_pit': 24.5})

test_sample['Att_Pit_Duration'] = bahrain_dna['avg_pit']
test_sample['Pit_Delta'] = 0.0
test_sample['Track_Temp'] = 30.0
test_sample['Att_Hardness'] = 2
test_sample['Def_Hardness'] = 2
test_sample['Race_Progress'] = 0.5 
test_sample['Track_Undercut_Index'] = bahrain_dna['index']

prediction = model.predict(test_sample)
result = "SUCCESS" if prediction[0] == 1 else "FAIL"
print(f"Scenario: Gap 0.8s, 50% through race (Bahrain)")
print(f"Model says: {result}")

joblib.dump(model, 'undercut_predictor.pkl')
# Also save the feature names so the predictor tool knows what to do!
joblib.dump(all_features, 'model_features.pkl')
print("\n💾 Model saved as 'undercut_predictor.pkl'")
print("💾 Feature list saved as 'model_features.pkl'")