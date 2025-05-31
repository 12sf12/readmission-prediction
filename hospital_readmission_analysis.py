# hospital_readmission_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, ConfusionMatrixDisplay

# Load dataset
df = pd.read_csv("Assignment_Data.csv")

# Initial cleanup
df.drop(columns=["patient_id"], inplace=True)

# Convert categorical features to category dtype
categorical_columns = ["gender", "diagnosis_code", "medication_type"]
for col in categorical_columns:
    df[col] = df[col].astype("category")

# Encode categorical features
df_encoded = pd.get_dummies(df.drop(columns=["discharge_note"]), drop_first=True)

# Separate target and features
X = df_encoded.drop("readmitted_30_days", axis=1)
y = df_encoded["readmitted_30_days"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Initialize and train model
model = RandomForestClassifier(class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Evaluation
roc_auc = roc_auc_score(y_test, y_proba)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Display results
print("\nClassification Report:\n", report)
print("ROC AUC:", roc_auc)
print("F1 Score:", f1)

# Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot()
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# Feature Importance
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop Features:\n", importances.head(10))

# Save feature importance plot
plt.figure(figsize=(10, 6))
importances.head(10).plot(kind='barh')
plt.title("Top 10 Important Features")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()
