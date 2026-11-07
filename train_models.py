import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# 1. Loading and Cleaning Dataset
df = pd.read_csv('adult_income_classification.csv')
# The adult dataset often using '?' for missing values - Replacing and dropping them.
df = df.replace('?', pd.NA).dropna()

# 2. Preprocessing
target = 'income'
X = df.drop(columns=[target])
y = LabelEncoder().fit_transform(df[target]) # 0 for <=50K, 1 for >50K

# One-hot encode categorical features
X = pd.get_dummies(X, drop_first=True)

# Scale features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 3. Train-Test Split (Sampled for SVM training speed and Streamlit upload limits)
# will use 5000 samples overall to keep it fast which also satisfy assignment requirement.
X_subset, _, y_subset, _ = train_test_split(X_scaled, y, train_size=5000, random_state=42, stratify=y)
X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, test_size=0.2, random_state=42)

# 4. Saving Test Data for Streamlit App
test_data = X_test.copy()
test_data['income_target'] = y_test
test_data.to_csv('test_data.csv', index=False)
print("Saved test_data.csv successfully.")

# 5. Training and Saving the Models
os.makedirs('model', exist_ok=True)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'Support Vector Machine': SVC(probability=True, random_state=42) # probability=True required for AUC
}

for name, clf in models.items():
    print(f"Training {name}...")
    clf.fit(X_train, y_train)
    filename = f"model/{name.replace(' ', '_').lower()}.pkl"
    joblib.dump(clf, filename)
    print(f"Saved {filename}")

print("All models trained and saved!")