import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

Data_set = 'Telco-Customer-Churn.csv'
df = pd.read_csv(Data_set)
print('Dataset loaded. Rows, columns:', df.shape)

print('\nColumns:\n', df.columns.tolist())
print('\nSample rows:')
print(df.head())

# 4. Simple cleaning & preprocessing

# Convert TotalCharges to numeric if present (some dataset versions have spaces)
if 'TotalCharges' in df.columns:
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan), errors='coerce')

# Fill numeric NaNs with median
for c in df.select_dtypes(include=[np.number]).columns:
    df[c] = df[c].fillna(df[c].median())

# Fill categorical NaNs with mode
for c in df.select_dtypes(include=['object']).columns:
    if df[c].isnull().any():
        df[c] = df[c].fillna(df[c].mode().iloc[0])

# Convert 'Yes'/'No' style columns to 1/0 where appropriate
yes_no_map = lambda x: 1 if str(x).strip().lower().startswith('y') else 0
for c in df.select_dtypes(include=['object']).columns:
    uniques = df[c].dropna().unique()
    if len(uniques) == 2 and set(map(str.lower, map(str, uniques))).issubset({'yes','no','y','n'}):
        df[c] = df[c].apply(yes_no_map)

# Convert Churn target to 0/1 if present
if 'Churn' in df.columns:
    # If already numeric, keep as is
    if df['Churn'].dtype != 'int64' and df['Churn'].dtype != 'float64':
        df['Churn'] = df['Churn'].apply(yes_no_map)

# Create a simple tenure_group for insight
if 'tenure' in df.columns:
    df['tenure_group'] = pd.cut(df['tenure'], bins=[-1,6,12,24,48,72], labels=['0-6','7-12','13-24','25-48','49-72'])

print('\nAfter cleaning, sample rows:')
print(df.head())

# 5. Visualizations (simple & pastel)
# --------------------
print('\nGenerating visualizations...')

# Churn count

if 'Churn' in df.columns:
    plt.figure()
    ax = sns.countplot(data=df, x='Churn')
    
    for container in ax.containers:
        ax.bar_label(container)

    plt.title('Churn count (0 = No, 1 = Yes)')
    plt.tight_layout()
    plt.show()

# Churn by Contract type
if 'Contract' in df.columns and 'Churn' in df.columns:
    plt.figure()
    sns.countplot(data=df, x='Contract', hue='Churn')
    plt.title('Churn by Contract type')
    plt.tight_layout()
    plt.show()

# MonthlyCharges distribution by Churn (if present)
if 'MonthlyCharges' in df.columns and 'Churn' in df.columns:
    plt.figure()
    # Use jittered violin-like plot replaced with kde for simplicity
    sns.kdeplot(df[df['Churn']==0]['MonthlyCharges'], label='No churn', fill=True)
    sns.kdeplot(df[df['Churn']==1]['MonthlyCharges'], label='Churn', fill=True)
    plt.title('MonthlyCharges distribution by Churn')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Correlation heatmap for numeric features (small)
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(num_cols) > 1:
    plt.figure(figsize=(10,6))
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues', vmin=-1, vmax=1)
    plt.title('Correlation matrix (numeric features)')
    plt.tight_layout()
    plt.show()

# 6. Prepare data for ML

TARGET = 'Churn'
if TARGET not in df.columns:
    raise ValueError("Target column 'Churn' not found in dataset after preprocessing.")

# Keep a copy of raw data for output
df_raw = df.copy()

# Drop customer ID if present (not used for training)
drop_cols = [col for col in df.columns if col.lower() in ['customerid','id']]

df_features = df.drop(columns=drop_cols + [TARGET], errors='ignore')

# One-hot encode categorical variables
X = pd.get_dummies(df_features, drop_first=True)
y = df[TARGET].values

# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print('\nTraining set shape:', X_train.shape, 'Test set shape:', X_test.shape)

# 7. Train three models

# 1) Logistic Regression
print("\nTraining Logistic Regression...")
lr = LogisticRegression(max_iter=1000, solver='liblinear')
lr.fit(X_train, y_train)
preds_lr = lr.predict(X_test)
acc_lr = accuracy_score(y_test, preds_lr)
print(f"Logistic Regression accuracy: {acc_lr:.3f}")
print(classification_report(y_test, preds_lr))
print("Confusion matrix:\n", confusion_matrix(y_test, preds_lr))

# 2) Random Forest
print("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
preds_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, preds_rf)
print(f"Random Forest accuracy: {acc_rf:.3f}")
print(classification_report(y_test, preds_rf))
print("Confusion matrix:\n", confusion_matrix(y_test, preds_rf))

# 3) Decision Tree
print("\nTraining Decision Tree...")
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
preds_dt = dt.predict(X_test)
acc_dt = accuracy_score(y_test, preds_dt)
print(f"Decision Tree accuracy: {acc_dt:.3f}")
print(classification_report(y_test, preds_dt))
print("Confusion matrix:\n", confusion_matrix(y_test, preds_dt))

# Choose best model by accuracy
best_model = lr
best_name = "LogisticRegression"
best_acc = acc_lr

if acc_rf > best_acc:
    best_model = rf
    best_name = "RandomForest"
    best_acc = acc_rf

if acc_dt > best_acc:
    best_model = dt
    best_name = "DecisionTree"
    best_acc = acc_dt

print(f"\nBest model by accuracy: {best_name} (accuracy = {best_acc:.3f})")

# 8. Final predictions on full data & save CSV (no probabilities, just predictions)

print("\nGenerating final predictions on full dataset...")
X_all = X.copy()
preds_all = best_model.predict(X_all)           
output = df_raw.copy()
output['Churn_Pred'] = preds_all

output_file = 'churn_predictions.csv'
output.to_csv(output_file, index=False)
print(f"Saved predictions to {output_file}")

























