# ============================================================
# TASK-2: PREDICTIVE ANALYSIS USING MACHINE LEARNING
# Dataset: Titanic Survival Dataset
# Model: Logistic Regression
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------
print("Loading Dataset...")

df = pd.read_csv("Titanic-Dataset.csv")

print("\nDataset Loaded Successfully!")
print(df.head())

# ------------------------------------------------------------
# Dataset Information
# ------------------------------------------------------------
print("\nDataset Shape:", df.shape)

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# ------------------------------------------------------------
# Missing Values
# ------------------------------------------------------------
print("\nMissing Values:")
print(df.isnull().sum())

# ------------------------------------------------------------
# Data Preprocessing
# ------------------------------------------------------------
print("\nPerforming Data Preprocessing...")

# Fill missing values
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# Convert categorical values to numerical values
df["Sex"] = df["Sex"].map({
    "male": 0,
    "female": 1
})

df["Embarked"] = df["Embarked"].map({
    "S": 0,
    "C": 1,
    "Q": 2
})

print("Preprocessing Completed!")

# ------------------------------------------------------------
# Feature Selection
# ------------------------------------------------------------
print("\nSelecting Features...")

X = df[
    [
        "Pclass",
        "Sex",
        "Age",
        "Fare",
        "Embarked"
    ]
]

y = df["Survived"]

print("\nSelected Features:")
print(X.head())

# ------------------------------------------------------------
# Train-Test Split
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# ------------------------------------------------------------
# Model Training
# ------------------------------------------------------------
print("\nTraining Logistic Regression Model...")

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("Model Training Completed!")

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------
print("\nGenerating Predictions...")

y_pred = model.predict(X_test)

# ------------------------------------------------------------
# Model Evaluation
# ------------------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------
print("\nGenerating Visualization...")

plt.figure(figsize=(6, 4))

df["Survived"].value_counts().plot(
    kind="bar"
)

plt.title("Titanic Survival Distribution")
plt.xlabel("Survived")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig("survival_distribution.png")

plt.show()

# ------------------------------------------------------------
# Feature Importance
# ------------------------------------------------------------
print("\nFeature Importance:")

importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

print(importance)

# ------------------------------------------------------------
# Insights
# ------------------------------------------------------------
print("\nINSIGHTS")
print("1. Logistic Regression was used to predict passenger survival.")
print("2. Features such as Passenger Class, Age, Fare, Sex, and Embarked location influenced predictions.")
print("3. Female passengers generally had higher survival rates.")
print("4. Passenger Class significantly affected survival probability.")
print("5. The model achieved good prediction accuracy.")
print("6. Machine Learning successfully identified patterns and predicted outcomes.")

print("\nTASK COMPLETED SUCCESSFULLY!")
