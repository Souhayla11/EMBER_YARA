import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression





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

# X = X.reshape(X.shape[0], 1, X.shape[1])  # LSTM input shape: (batch_size, sequence_length, input_size)

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
rf_scores = clf.predict(X_test)


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
print("Accuracy scores for each fold:", scores)
print(f"Mean accuracy: {scores.mean():.4f}")
# Calculer l'accuracy
# accuracy = accuracy_score(y_test, rf_scores)
# print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, rf_scores, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_scores))

roc_auc = roc_auc_score(y_test, rf_scores)
print(f"ROC AUC: {roc_auc:.4f}")
# PR AUC (also called Average Precision Score)
pr_auc = average_precision_score(y_test, rf_scores)
print(f"PR AUC: {pr_auc:.4f}")

cm = confusion_matrix(y_test, rf_scores)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

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

xgb_scores = xgb_model.predict(X_test)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(xgb_model, X, y, cv=cv, scoring='accuracy')
print("Accuracy scores for each fold:", scores)
print(f"Mean accuracy: {scores.mean():.4f}")

# Calculer l'accuracy
# accuracy = accuracy_score(y_test, xgb_scores)
# print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, xgb_scores, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_test, xgb_scores))

roc_auc = roc_auc_score(y_test, xgb_scores)
print(f"ROC AUC: {roc_auc:.4f}")
# PR AUC (also called Average Precision Score)
pr_auc = average_precision_score(y_test, xgb_scores)
print(f"PR AUC: {pr_auc:.4f}")

cm = confusion_matrix(y_test, xgb_scores)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()


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

gbm_scores = model.predict(X_test) 

# Calculer l'accuracy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
print("Accuracy scores for each fold:", scores)
print(f"Mean accuracy: {scores.mean():.4f}")

# accuracy = accuracy_score(y_test, gbm_scores)
# print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, gbm_scores, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_test, gbm_scores))

roc_auc = roc_auc_score(y_test, gbm_scores)
print(f"ROC AUC: {roc_auc:.4f}")
# PR AUC (also called Average Precision Score)
pr_auc = average_precision_score(y_test, gbm_scores)
print(f"PR AUC: {pr_auc:.4f}")

cm = confusion_matrix(y_test, gbm_scores)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

print("................................................................................")

from catboost import CatBoostClassifier
model_catboost = CatBoostClassifier(verbose=0)

print("Training Catboost model...")
model_catboost.fit(X_train, y_train)

cat_scores = model_catboost.predict(X_test)

# Calculer l'accuracy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model_catboost, X, y, cv=cv, scoring='accuracy')
print("Accuracy scores for each fold:", scores)
print(f"Mean accuracy: {scores.mean():.4f}")

# accuracy = accuracy_score(y_test, cat_scores)
# print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, cat_scores, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_test, cat_scores))

roc_auc = roc_auc_score(y_test, cat_scores)
print(f"ROC AUC: {roc_auc:.4f}")
# PR AUC (also called Average Precision Score)
pr_auc = average_precision_score(y_test, cat_scores)
print(f"PR AUC: {pr_auc:.4f}")

cm = confusion_matrix(y_test, cat_scores)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# from sklearn.inspection import permutation_importance

# results = permutation_importance(model_catboost, X_test, y_test, scoring='accuracy')
# feat_importance = pd.Series(results.importances_mean, index=feature_names).sort_values(ascending=False)
# print(feat_importance.head(10))
print("................................................................................")

# === Train SVM Model ===
print("Training Logistic Regression...")

reg_model = LogisticRegression(max_iter=1000)  # max_iter is important to ensure convergence
reg_model.fit(X_train, y_train)

reg_scores = reg_model.predict(X_test)

# Calculer l'accuracy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(reg_model, X, y, cv=cv, scoring='accuracy')
print("Accuracy scores for each fold:", scores)
print(f"Mean accuracy: {scores.mean():.4f}")

# accuracy = accuracy_score(y_test, reg_scores)
# print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, reg_scores, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_test, reg_scores))

roc_auc = roc_auc_score(y_test, reg_scores)
print(f"ROC AUC: {roc_auc:.4f}")
# PR AUC (also called Average Precision Score)
pr_auc = average_precision_score(y_test, reg_scores)
print(f"PR AUC: {pr_auc:.4f}")

cm = confusion_matrix(y_test, reg_scores)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()
print("................................................................................")


# # === Train SVM Model ===
# print("Training SVM...")
# svm = SVC(kernel="rbf", probability=True, random_state=42)
# svm.fit(X_train, y_train)

# # === Predict ===
# y_pred = svm.predict(X_test)
# y_proba = svm.predict_proba(X_test)[:, 1]

# # === Evaluation ===
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# reg_scores = cross_val_score(svm, X, y, cv=cv, scoring='accuracy')
# print("Accuracy scores for each fold:", scores)
# print(f"Mean accuracy: {scores.mean():.4f}")

# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.4f}")
# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, y_pred))
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# roc_auc = roc_auc_score(y_test, y_proba)
# print(f"\nROC AUC Score: {roc_auc:.4f}")
# precision, recall, _ = precision_recall_curve(y_test, y_proba)
# pr_auc = auc(recall, precision)
# print(f"PR AUC Score: {pr_auc:.4f}")

# cm = confusion_matrix(y_test, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# disp.plot(cmap='Blues')
# plt.title("Confusion Matrix")
# plt.show()

# print("................................................................................")


# X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])  # LSTM input shape: (batch_size, sequence_length, input_size)

# class MalwareDataset(Dataset):
#     def __init__(self, X, y): 
#         self.X = torch.tensor(X, dtype=torch.float32)
#         self.y = torch.tensor(y, dtype=torch.long)

#     def __len__(self):
#         return len(self.X)

#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]

# train_data = MalwareDataset(X_train, y_train)
# test_data = MalwareDataset(X_test, y_test)

# train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_data, batch_size=32)


# class MalwareLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(MalwareLSTM, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, num_classes)

#     def forward(self, x):
#         h0 = torch.zeros(1, x.size(0), 64)
#         c0 = torch.zeros(1, x.size(0), 64)
#         out, _ = self.lstm(x, (h0, c0))
#         out = self.fc(out[:, -1, :])
#         return out


# model = MalwareLSTM(input_size=X.shape[2], hidden_size=64, num_classes=len(np.unique(y)))

# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# for epoch in range(20):
#     model.train()
#     total_loss = 0

#     for batch_X, batch_y in train_loader:
#         optimizer.zero_grad()
#         output = model(batch_X)
#         loss = criterion(output, batch_y)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# from sklearn.metrics import classification_report, precision_recall_fscore_support

# model.eval()
# correct, total = 0, 0
# all_preds = []
# all_targets = []

# with torch.no_grad():
#     for inputs, targets in test_loader:
#         outputs = model(inputs)
#         _, predicted = torch.max(outputs.data, 1)
#         total += targets.size(0)
#         correct += (predicted == targets).sum().item()

#         all_preds.extend(predicted.cpu().numpy())
#         all_targets.extend(targets.cpu().numpy())

# # Print accuracy
# print(f"Test Accuracy: {100 * correct / total:.2f}%")

# # Calculate precision, recall, f1-score
# # print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))

# precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')
# print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")


#######################################################Cross-Validation###############################################
# Define models
models = {
    "Random Forest": clf,
    "XGBoost":xgb_model ,
    "LightGBM": model,
    "CATBoost": model_catboost,
    "Logistic Regression": reg_model,
    # "SVM": svm,
    # "LSTM": ,
}

# Collect CV scores
model_scores = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
k =5
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    model_scores[name] = scores

# Plot
plt.figure(figsize=(10, 6))

for name, scores in model_scores.items():
    plt.plot(range(1, k+1), scores, marker='o', label=name)

plt.title("Cross-Validation Accuracy Across Folds")
plt.xlabel("Fold Number")
plt.ylabel("Accuracy")
plt.ylim(0.7, 1.00)
plt.xticks(range(1, k+1))
# plt.grid(True)
plt.legend()
plt.show()
