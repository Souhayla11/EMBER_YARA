import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score




# Read a .jsonl (JSON Lines) file
df_ember0 = pd.read_json('JSON/train_features_1.jsonl', lines=True)
df_ember1 = pd.read_json('JSON/train_features_0.jsonl', lines=True)
df_ember2 = pd.read_json('JSON/train_features_2.jsonl', lines=True)
df_ember3 = pd.read_json('JSON/train_features_3.jsonl', lines=True)
df_ember4 = pd.read_json('JSON/train_features_4.jsonl', lines=True)
df_ember5 = pd.read_json('JSON/train_features_5.jsonl', lines=True)
df_ember_test = pd.read_json('JSON/test_features.jsonl', lines=True)

df_ember = pd.concat([df_ember0, df_ember1, df_ember2, df_ember3,df_ember4,df_ember5], ignore_index=True, )
df_ember.rename(columns={'avclass': 'family', 'strings': 'string_stats', 'imports':'strings'}, inplace=True)
df_ember = df_ember[df_ember['label'] != -1]


df_yara = pd.read_csv('yara_dataset.csv')

# Find missing columns in each dataset
missing_in_json = set(df_yara.columns) - set(df_ember.columns)
missing_in_csv = set(df_ember.columns) - set(df_yara.columns)

# Add missing columns with 0 values
for col in missing_in_json:
    df_ember[col] = 0  

for col in missing_in_csv:
    df_yara[col] = 0

df_ember = df_ember[df_yara.columns]  # Now both have the same column order


combined_df = pd.concat([df_ember, df_yara], ignore_index=True).fillna(0)

family_to_class = {name: idx for idx, name in enumerate(combined_df['family'].unique())}
combined_df['class'] = combined_df['family'].map(family_to_class)

# print(combined_df[['family', 'class']].sort_values('class'))
# print(combined_df.shape)
# print(combined_df['label'].unique())
# print(combined_df[['family', 'class']].drop_duplicates())


categorical_cols = combined_df.select_dtypes(include=['object', 'category']).columns

# Label Encoder
for col in categorical_cols:
    combined_df[col] = LabelEncoder().fit_transform(combined_df[col].astype(str))


# df = pd.get_dummies(df_ember, columns=categorical_cols)

# Normalize
columns_to_normalize = [col for col in combined_df.columns if col not in ['label']]
scaler = MinMaxScaler()
combined_df[columns_to_normalize] = scaler.fit_transform(combined_df[columns_to_normalize])

X = combined_df.drop(columns=["label"]).values.astype(np.float32)
y = combined_df["label"].values.astype(np.int64)

# # Separate features and labels for training set
# X_train = combined_df.drop(columns=["label"]).values.astype(np.float32)
# y_train = combined_df["label"].values.astype(np.int64)

# # Separate features and labels for testing set
# X_test = df_ember_test.drop(columns='label').values.astype(np.float32)
# y_test = df_ember_test['label'].values.astype(np.int64)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)

print("Training Random Forest model...")
# === Train model ===
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
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

print("................................................................................")