import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb

# Define root directory
root_dir = '/Users/meng-hsuehsung/Downloads/simulated_data/'
all_sheets = {}

# Detect directories and files, file name as labels
sheet_number = 1

for folder_name in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder_name)
    if os.path.isdir(folder_path):  # Detect if directory or not
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path) and file_name.endswith('.csv'):  # Detect if csv file or not
                try:
                    df = pd.read_csv(file_path) # read total file, should notice OOM issue
                    label = file_name.split('.')[0]  # Use file name（extension name not included）as labels
                    sheet_name = f"{label}_Sheet{sheet_number:02d}"
                    df['label'] = label  # Add label column
                    all_sheets[sheet_name] = df
                    sheet_number += 1
                except Exception as e:
                    print(f"讀取檔案失敗: {file_path}, 錯誤: {e}")
            else:
                print(f"跳過無效檔案: {file_path}")

# Combine all sheet as Excel
output_path = os.path.join(root_dir, 'combined_data_with_labels.xlsx')
with pd.ExcelWriter(output_path) as writer:
    for sheet_name, df in all_sheets.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"所有數據已保存到 {output_path}")

# 分類器定義及訓練
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=2000),
    'k-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machine': SVC(),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# label_mapping generating automatically
unique_labels = set()
for df in all_sheets.values():
    unique_labels.update(df['label'].unique())
label_mapping = {label: idx for idx, label in enumerate(sorted(unique_labels))}

X_train_total = pd.DataFrame()
X_test_total = pd.DataFrame()
y_train_total = pd.Series(dtype='int')
y_test_total = pd.Series(dtype='int')
for sheet_name, df in all_sheets.items():
    df['label'] = df['label'].map(label_mapping)
    X = df.drop(columns=['label'])
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    X_train_total = pd.concat([X_train_total, X_train], ignore_index=True)
    X_test_total = pd.concat([X_test_total, X_test], ignore_index=True)
    y_train_total = pd.concat([y_train_total, y_train], ignore_index=True)
    y_test_total = pd.concat([y_test_total, y_test], ignore_index=True)

# Model evaluate
best_model = None
best_accuracy = 0
for name, clf in classifiers.items():
    clf.fit(X_train_total, y_train_total)
    y_pred = clf.predict(X_test_total)
    accuracy = accuracy_score(y_test_total, y_pred)
    print(f"{name} Accuracy: {accuracy:.4f}")
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = clf

# Predict testing data
test_data_path = os.path.join(root_dir, 'testing.csv')
test_data = pd.read_csv(test_data_path)
test_pred = best_model.predict(test_data)
reverse_label_mapping = {v: k for k, v in label_mapping.items()}
predicted_class = reverse_label_mapping[test_pred[0]]
print(f"testing.csv 的預測屬於類別: {predicted_class}")
