import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier  # Import Gradient Boosting

from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE  # Import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline  # Import Imbalanced Pipeline to combine SMOTE
import pickle
import matplotlib.pyplot as plt
import io
import base64

# Load the dataset
df = pd.read_csv('deposit term dirty null.csv', low_memory=False)

# Drop unnecessary columns
df = df.drop(['Id', 'BankId', 'Year', 'first_name', 'last_name', 'email'], axis=1, errors='ignore')

# Drop rows with missing categorical data
df.dropna(subset=['housing', 'default', 'month', 'contact', 'job'], inplace=True)

# Convert 'duration' to numeric
df['duration'] = pd.to_numeric(df['duration'], errors='coerce')


# Fill missing values for numeric columns with median
df['age'] = df['age'].fillna(df['age'].mean())
df['balance'] = df['balance'].fillna(df['balance'].mean())
df['duration'] = df['duration'].fillna(df['duration'].mean())
df['pdays'] = df['pdays'].fillna(df['pdays'].mean())

# Replace negative balances with the mean of positive balances
mean_balance = df[df['balance'] >= 0]['balance'].mean()
df['balance'] = df['balance'].apply(lambda x: mean_balance if x < 0 else x)

#Replace negative age with median age
age = df[df['age'] >= 0]['age'].median()
df['age'] = df['age'].apply(lambda x: age if x < 0 else x)

# Function to cap outliers
def cap_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.clip(df[column], lower_bound, upper_bound)  # Cap values
    return df

# Cap outliers for selected columns
df = cap_outliers(df, 'age')
df = cap_outliers(df, 'balance')
df = cap_outliers(df, 'day')
df = cap_outliers(df, 'campaign')

# Drop 'pdays' and 'previous' columns
df = df.drop(['pdays', 'previous'], axis=1)

# 1. Data preparation
# Assuming data set is already loaded in df
# Feature-target split
X = df.drop('y', axis=1) #Features
y = df['y'].apply(lambda x: 1 if x == 'yes' else 0)  # Binary encoding for target

# 2. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer

#  Data Standardization and Label Encoding
def preprocess_data(X, categorical_features, numerical_features):
    # Label encoding for categorical features and scaling numerical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features), #Scaling for numerical features
            ('cat', OrdinalEncoder(), categorical_features) #Label encoding for categorical features
        ]
    )
    return preprocessor

# Automatically identify categorical and numerical columns
categorical_features = [col for col in X.columns if X[col].dtype == 'object']  # Automatically get categorical features
numerical_features = [col for col in X.columns if X[col].dtype in ['int64', 'int32','float64']]  # Automatically get numerical features

# Apply preprocessing
preprocessor = preprocess_data(X, categorical_features, numerical_features)

# Transform the data (X_train and X_test)
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

#Apply SMOTE to the training data to balance the classes
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_preprocessed,y_train)
X_test_resampled, y_test_resampled = smote.fit_resample(X_test_preprocessed,y_test)




# Define the SMOTE and Gradient Boosting pipeline
model_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),  # First, preprocess data
    ('smote', SMOTE(random_state=42)),  # Then apply SMOTE to the resampled data
    ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))  # Apply Gradient Boosting
])

# Train the model
model_pipeline.fit(X_train, y_train)

# Predict probabilities for class 1 (yes)
#y_proba = model_pipeline.predict_proba(X_test)[:, 1]

# Make predictions and evaluate accuracy
y_pred = model_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Gradient Boosting Model Accuracy: {accuracy * 100:.2f}%")

# Save the model and preprocessor
with open('gradient_boosting_model.pkl', 'wb') as model_file:
    pickle.dump(model_pipeline, model_file)

print("Gradient Boosting Model saved!")

