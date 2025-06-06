import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score



# Read a .jsonl (JSON Lines) file
df_ember = pd.read_json('JSON/train_features_1.jsonl', lines=True)
df_ember1 = pd.read_json('JSON/train_features_0.jsonl', lines=True)
df_ember2 = pd.read_json('JSON/train_features_2.jsonl', lines=True)
df_ember3 = pd.read_json('JSON/train_features_3.jsonl', lines=True)
df_ember4 = pd.read_json('JSON/train_features_4.jsonl', lines=True)
df_ember5 = pd.read_json('JSON/train_features_5.jsonl', lines=True)


# df_ember_test = pd.read_json('JSON/test_features.jsonl', lines=True)
df_combined = pd.concat([df_ember, df_ember1, df_ember2, df_ember3,df_ember4,df_ember5], ignore_index=True, )


df_combined = df_combined[df_combined['label'] != -1]
# df_ember_test = df_ember_test[df_ember_test['label'] != -1]
# print(df_combined.columns)
# print(df_combined["label"].unique())
# print(df_ember.columns)

categorical_cols = df_combined.select_dtypes(include=['object', 'category']).columns
# Label Encoder
for col in categorical_cols:
    df_combined[col] = LabelEncoder().fit_transform(df_combined[col].astype(str))

# df = pd.get_dummies(df_ember, columns=categorical_cols)

# Normalize
columns_to_normalize = [col for col in df_combined.columns if col not in ['label']]


scaler = MinMaxScaler()
df_combined[columns_to_normalize] = scaler.fit_transform(df_combined[columns_to_normalize])


X = df_combined.drop(columns=["label"]).values.astype(np.float32)
y = df_combined["label"].values.astype(np.int64)

# print(df_ember["label"].unique())
# print(df_ember_test["label"].unique())


# Separate features and labels for training set
# X_train = df_combined.drop(columns=["label"]).values.astype(np.float32)
# y_train = df_combined["label"].values.astype(np.int64)

# # Separate features and labels for testing set
# X_test = df_ember_test.drop(columns='label').values.astype(np.float32)
# y_test = df_ember_test['label'].values.astype(np.int64)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)


# === Train model ===
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
print("Training Random Forest model...")
clf.fit(X_train, y_train)

# === Evaluate ===
y_pred = clf.predict(X_test)
# Calculer l'accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_pred)
print(f"ROC AUC: {roc_auc:.4f}")

# PR AUC (also called Average Precision Score)
pr_auc = average_precision_score(y_test, y_pred)
print(f"PR AUC: {pr_auc:.4f}")

print("................................................................................")


xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    use_label_encoder=False,
    n_estimators=200,
    max_depth=10,
    learning_rate=0.1,
    tree_method="gpu_hist",  # Utiliser "gpu_hist" si un GPU est disponible
    random_state=42,
    n_jobs=-1
)

print("Training XGBoost model...")
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)

# Calculer l'accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_pred)
print(f"ROC AUC: {roc_auc:.4f}")

# PR AUC (also called Average Precision Score)
pr_auc = average_precision_score(y_test, y_pred)
print(f"PR AUC: {pr_auc:.4f}")


# from sklearn.inspection import permutation_importance

# # After training your model
# result = permutation_importance(xgb_model, X_train, y_train, n_repeats=10, random_state=42)
# feature_names = df_combined.columns.drop("label")  # or your actual feature column names


# # Show top features
# importances = result.importances_mean
# feature_ranking = sorted(zip(importances, feature_names), reverse=True)

# for score, name in feature_ranking[:10]:
#     print(f"{name}: {score:.4f}")



print("................................................................................")


from lightgbm import LGBMClassifier
model = LGBMClassifier()
print("Training LightGBM model...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Calculer l'accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
roc_auc = roc_auc_score(y_test, y_pred)
print(f"ROC AUC: {roc_auc:.4f}")

# PR AUC (also called Average Precision Score)
pr_auc = average_precision_score(y_test, y_pred)
print(f"PR AUC: {pr_auc:.4f}")

print("................................................................................")

from catboost import CatBoostClassifier
model_catboost = CatBoostClassifier(verbose=0)

print("Training Catboost model...")
model_catboost.fit(X_train, y_train)

y_pred = model_catboost.predict(X_test)

# Calculer l'accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
roc_auc = roc_auc_score(y_test, y_pred)
print(f"ROC AUC: {roc_auc:.4f}")

# PR AUC (also called Average Precision Score)
pr_auc = average_precision_score(y_test, y_pred)
print(f"PR AUC: {pr_auc:.4f}")

# from sklearn.inspection import permutation_importance

# results = permutation_importance(model_catboost, X_test, y_test, scoring='accuracy')
# feat_importance = pd.Series(results.importances_mean, index=feature_names).sort_values(ascending=False)
# print(feat_importance.head(10))
print("................................................................................")

