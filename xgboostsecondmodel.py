import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import numpy as np


dfcleanclasses = pd.read_csv('myData_cleanedclasses.csv')
dfcleanclasses['Recommended Course'] = dfcleanclasses['Recommended Course'].str.lower()

# Get the value counts of each class
class_counts = dfcleanclasses['Recommended Course'].value_counts()

# Filter classes that appear only once
rare_classes = class_counts[class_counts == 1]

# Show which classes have only 1 sample
print("Classes with only 1 sample:")
print(rare_classes)

# Count how many rows in the dataset belong to those rare classes
num_rare_rows = dfcleanclasses['Recommended Course'].isin(rare_classes.index).sum()
print(f"\nNumber of rows with only 1-sample classes: {num_rare_rows}")

# Step 1: Parse multi-label features
dfcleanclasses['Skills'] = dfcleanclasses['Skills'].apply(lambda lst: [i.strip().lower() for i in lst if isinstance(i, str)])
dfcleanclasses['Interests'] = dfcleanclasses['Interests'].apply(lambda lst: [i.strip().lower() for i in lst if isinstance(i, str)])
dfcleanclasses['Extracurriculars'] = dfcleanclasses['Extracurriculars'].apply(lambda lst: [i.strip().lower() for i in lst if isinstance(i, str)])

# Step 2: One-hot encode multi-label features
mlb_skills = MultiLabelBinarizer()
mlb_interests = MultiLabelBinarizer()
mlb_extras = MultiLabelBinarizer()

skills_encoded = pd.DataFrame(mlb_skills.fit_transform(dfcleanclasses['Skills']), columns=mlb_skills.classes_)
interests_encoded = pd.DataFrame(mlb_interests.fit_transform(dfcleanclasses['Interests']), columns=mlb_interests.classes_)
extras_encoded = pd.DataFrame(mlb_extras.fit_transform(dfcleanclasses['Extracurriculars']), columns=mlb_extras.classes_)

# Step 3: Extract and scale numerical features
numerical_features = ['Bodily Activeness', 'Logical Reasoning', 'STEM Avg_score',
                      'Humanities Avg_score', 'Languages Avg_score', 'Intrapersonal Score',
                      'Interpersonal Score', 'Grade_3 Score', 'Grade_6 Score',
                      'Grade_9 Score', 'KCSE Score']

df_numerical = dfcleanclasses[numerical_features].copy()

# Combine all features
X = pd.concat([skills_encoded, interests_encoded, extras_encoded, df_numerical], axis=1)
y = dfcleanclasses['Recommended Course']

# Scale numerical features
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])

# Initialize encoder
label_encoder = LabelEncoder()

# Fit and transform target column to numeric labels
y_encoded = label_encoder.fit_transform(dfcleanclasses['Recommended Course'])


# For decoding predictions later
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))


# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=90, stratify=y_encoded, random_state=42)


# Step 5: Handle class imbalance
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)


# Step 6: Train XGBoost model
xgb_model = XGBClassifier(
    objective='multi:softmax',  # for multi-class classification
    num_class=len(np.unique(y_encoded)),  # number of classes
    learning_rate=0.1,
    max_depth=6,
    n_estimators=100,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='mlogloss',
    random_state=42
)

# Convert inputs to appropriate types
X_train_np = X_train.values  # Use .values instead of .to_numpy()
y_train_np = np.array(y_train, dtype=np.int32)  # Ensure y_train is int32
sample_weights_np = np.array(sample_weights, dtype=np.float32)  # Ensure weights are float32

# Fit the model
xgb_model.fit(
    X_train_np,
    y_train_np,
    sample_weight=sample_weights_np
)

# Step 7: Evaluate
X_test_np = X_test.values  # Convert test data as well
y_pred = xgb_model.predict(X_test_np)
print(classification_report(y_test, y_pred, zero_division=0))