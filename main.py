import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import ADASYN # Advanced SMOTE

# 1. LOAD DATA
base_path = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_path, 'heart_combined.csv')
df = pd.read_csv(file_path)

# 2. ENCODING
df_encoded = pd.get_dummies(df, columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'], drop_first=True)
X = df_encoded.drop('HeartDisease', axis=1)
y = df_encoded['HeartDisease']

# 3. FEATURE SELECTION (The "Secret" to 95%+)
selector = SelectKBest(score_func=f_classif, k=12)
X_selected = selector.fit_transform(X, y)

# 4. DATA TRANSFORMATION (Normalizing the Bell Curve)
pt = PowerTransformer()
X_transformed = pt.fit_transform(X_selected)

# 5. BALANCING (ADASYN is better than SMOTE for 90%+)
ada = ADASYN(random_state=42)
X_res, y_res = ada.fit_resample(X_transformed, y)

# 6. SPLIT
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.15, random_state=42, stratify=y_res)

# 7. ELITE STACKING
base_models = [
    ('rf', RandomForestClassifier(n_estimators=500, max_depth=12, random_state=42)),
    ('xgb', xgb.XGBClassifier(n_estimators=500, learning_rate=0.03, max_depth=8, gamma=0.3, subsample=0.8, eval_metric='logloss'))
]

final_model = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(),
    cv=10 # Higher CV for more stable results
)

print("Running Elite Precision Training...")
final_model.fit(X_train, y_train)

# 8. RESULTS
y_pred = final_model.predict(X_test)
print(f"\n--- PRECISION RESEARCH RESULTS ---")
print(f"Final Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(classification_report(y_test, y_pred))