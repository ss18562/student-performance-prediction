import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("=" * 60)
print("STUDENT PERFORMANCE PREDICTION MODEL")
print("=" * 60)

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("data/student_performance.csv")

print("\nDataset Loaded Successfully")
print(f"Dataset Shape: {df.shape}")

# ==========================================
# FEATURES & TARGET
# ==========================================

X = df.drop("Performance", axis=1)
y = df["Performance"]

# ==========================================
# ENCODE CATEGORICAL FEATURES
# ==========================================

encoders = {}

categorical_cols = X.select_dtypes(include=["object"]).columns

for col in categorical_cols:

    le = LabelEncoder()

    X[col] = le.fit_transform(X[col].astype(str))

    encoders[col] = le

print("\nEncoded Columns:")
for col in categorical_cols:
    print(f"✓ {col}")

# ==========================================
# ENCODE TARGET
# ==========================================

target_encoder = LabelEncoder()

y = target_encoder.fit_transform(y)

print("\nTarget Classes:")
print(target_encoder.classes_)

# ==========================================
# CHECK DATA TYPES
# ==========================================

print("\nFeature Data Types:")
print(X.dtypes)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples :", len(X_test))

# ==========================================
# RANDOM FOREST MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

# ==========================================
# TRAIN MODEL
# ==========================================

print("\nTraining Model...")

model.fit(X_train, y_train)

print("Training Complete!")

# ==========================================
# PREDICTIONS
# ==========================================

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n" + "=" * 60)
print(f"MODEL ACCURACY : {accuracy * 100:.2f}%")
print("=" * 60)

# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        predictions,
        target_names=target_encoder.classes_
    )
)

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop Important Features:")
print("-" * 40)

for _, row in importance_df.iterrows():
    print(
        f"{row['Feature']:<25} "
        f"{row['Importance']:.4f}"
    )

# ==========================================
# SAVE FILES
# ==========================================

joblib.dump(
    model,
    "models/student_model.pkl"
)

joblib.dump(
    encoders,
    "models/encoders.pkl"
)

joblib.dump(
    target_encoder,
    "models/target_encoder.pkl"
)

print("\n" + "=" * 60)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 60)

print("\nCreated Files:")
print("✓ models/student_model.pkl")
print("✓ models/encoders.pkl")
print("✓ models/target_encoder.pkl")