import React, { useState } from 'react';
import { Copy, CheckCircle, ExternalLink } from 'lucide-react';

const MLExamSolution = () => {
  const [copiedSection, setCopiedSection] = useState('');

  const copyToClipboard = (text, section) => {
    navigator.clipboard.writeText(text);
    setCopiedSection(section);
    setTimeout(() => setCopiedSection(''), 2000);
  };

  const colabCode = `# ML Final Exam - Diabetes Prediction System
# Dataset: Pima Indians Diabetes Database

# Install required libraries
!pip install gradio scikit-learn pandas numpy seaborn matplotlib

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import gradio as gr
import joblib

# ========================================
# 1. DATA LOADING (5 Marks)
# ========================================
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, names=columns)

print("First few rows:")
print(df.head())
print(f"\\nDataset Shape: {df.shape}")

# ========================================
# 2. DATA PREPROCESSING (10 Marks)
# ========================================
print("\\n=== DATA PREPROCESSING ===")

# Step 1: Handle zero values as missing
print("\\n1. Handling zero values as missing:")
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_cols:
    df[col] = df[col].replace(0, np.nan)
print(f"Missing values after replacing zeros:\\n{df.isnull().sum()}")

# Step 2: Impute missing values with median
print("\\n2. Imputing missing values with median:")
for col in zero_cols:
    df[col].fillna(df[col].median(), inplace=True)
print(f"Missing values after imputation:\\n{df.isnull().sum()}")

# Step 3: Feature engineering - Create age groups
print("\\n3. Creating age groups:")
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 30, 50, 100], labels=[0, 1, 2])
df['AgeGroup'] = df['AgeGroup'].astype(int)
print(f"Age groups created: {df['AgeGroup'].unique()}")

# Step 4: Feature engineering - BMI categories
print("\\n4. Creating BMI categories:")
df['BMI_Category'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3])
df['BMI_Category'] = df['BMI_Category'].astype(int)
print(f"BMI categories created: {df['BMI_Category'].unique()}")

# Step 5: Outlier detection and handling using IQR
print("\\n5. Handling outliers using IQR method:")
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    return df

for col in ['Insulin', 'SkinThickness']:
    df = remove_outliers_iqr(df, col)
print("Outliers handled for Insulin and SkinThickness")

# Separate features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\\nTraining set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# ========================================
# 3. PIPELINE CREATION (10 Marks)
# ========================================
print("\\n=== PIPELINE CREATION ===")
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])
print("Pipeline created with StandardScaler and RandomForestClassifier")

# ========================================
# 4. PRIMARY MODEL SELECTION (5 Marks)
# ========================================
print("\\n=== PRIMARY MODEL SELECTION ===")
print("Model Selected: Random Forest Classifier")
print("Justification:")
print("- Handles non-linear relationships well")
print("- Robust to outliers")
print("- Provides feature importance")
print("- Works well with medical datasets")
print("- No assumption about data distribution")

# ========================================
# 5. MODEL TRAINING (10 Marks)
# ========================================
print("\\n=== MODEL TRAINING ===")
pipeline.fit(X_train, y_train)
print("Model trained successfully on training data")

# ========================================
# 6. CROSS-VALIDATION (10 Marks)
# ========================================
print("\\n=== CROSS-VALIDATION ===")
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Average CV Score: {cv_scores.mean():.4f}")
print(f"Standard Deviation: {cv_scores.std():.4f}")

# ========================================
# 7. HYPERPARAMETER TUNING (10 Marks)
# ========================================
print("\\n=== HYPERPARAMETER TUNING ===")
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [10, 20, 30, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print(f"\\nBest Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")

# ========================================
# 8. BEST MODEL SELECTION (10 Marks)
# ========================================
print("\\n=== BEST MODEL SELECTION ===")
best_model = grid_search.best_estimator_
print("Best model selected from GridSearchCV")
print(f"Best parameters: {grid_search.best_params_}")

# ========================================
# 9. MODEL PERFORMANCE EVALUATION (10 Marks)
# ========================================
print("\\n=== MODEL PERFORMANCE EVALUATION ===")
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

print("\\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the model
joblib.dump(best_model, 'diabetes_model.pkl')
print("\\nModel saved as 'diabetes_model.pkl'")

# ========================================
# 10. WEB INTERFACE WITH GRADIO (10 Marks)
# ========================================
print("\\n=== CREATING GRADIO INTERFACE ===")

def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age):
    # Create age group
    if age <= 30:
        age_group = 0
    elif age <= 50:
        age_group = 1
    else:
        age_group = 2
    
    # Create BMI category
    if bmi < 18.5:
        bmi_category = 0
    elif bmi < 25:
        bmi_category = 1
    elif bmi < 30:
        bmi_category = 2
    else:
        bmi_category = 3
    
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                           insulin, bmi, dpf, age, age_group, bmi_category]])
    
    prediction = best_model.predict(input_data)[0]
    probability = best_model.predict_proba(input_data)[0]
    
    result = "Diabetic" if prediction == 1 else "Non-Diabetic"
    confidence = probability[1] if prediction == 1 else probability[0]
    
    return f"Prediction: {result}\\nConfidence: {confidence:.2%}"

interface = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Number(label="Pregnancies", value=0),
        gr.Number(label="Glucose Level", value=120),
        gr.Number(label="Blood Pressure", value=70),
        gr.Number(label="Skin Thickness", value=20),
        gr.Number(label="Insulin", value=80),
        gr.Number(label="BMI", value=25),
        gr.Number(label="Diabetes Pedigree Function", value=0.5),
        gr.Number(label="Age", value=30)
    ],
    outputs=gr.Textbox(label="Result"),
    title="Diabetes Prediction System",
    description="Enter patient details to predict diabetes risk"
)

# Launch the interface
interface.launch(share=True)

# ========================================
# 11. DEPLOYMENT TO HUGGING FACE (10 Marks)
# Note: Create app.py with the deployment code below
# ========================================`;

  const appPyCode = `import gradio as gr
import joblib
import numpy as np

# Load the trained model
model = joblib.load('diabetes_model.pkl')

def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age):
    # Create age group
    if age <= 30:
        age_group = 0
    elif age <= 50:
        age_group = 1
    else:
        age_group = 2
    
    # Create BMI category
    if bmi < 18.5:
        bmi_category = 0
    elif bmi < 25:
        bmi_category = 1
    elif bmi < 30:
        bmi_category = 2
    else:
        bmi_category = 3
    
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                           insulin, bmi, dpf, age, age_group, bmi_category]])
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    result = "Diabetic" if prediction == 1 else "Non-Diabetic"
    confidence = probability[1] if prediction == 1 else probability[0]
    
    return f"Prediction: {result}\\nConfidence: {confidence:.2%}"

interface = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Number(label="Pregnancies", value=0),
        gr.Number(label="Glucose Level", value=120),
        gr.Number(label="Blood Pressure", value=70),
        gr.Number(label="Skin Thickness", value=20),
        gr.Number(label="Insulin", value=80),
        gr.Number(label="BMI", value=25),
        gr.Number(label="Diabetes Pedigree Function", value=0.5),
        gr.Number(label="Age", value=30)
    ],
    outputs=gr.Textbox(label="Result"),
    title="Diabetes Prediction System",
    description="Enter patient details to predict diabetes risk"
)

if __name__ == "__main__":
    interface.launch()`;

  const requirementsCode = `gradio==4.12.0
scikit-learn==1.3.2
pandas==2.1.4
numpy==1.26.2
joblib==1.3.2`;

  const readmeCode = `---
title: Diabetes Prediction System
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.12.0
app_file: app.py
pinned: false
---

# Diabetes Prediction System

ML Final Exam Project - Predicts diabetes using patient data.

## Dataset
Pima Indians Diabetes Database

## Model
Random Forest Classifier with hyperparameter tuning`;

  const deploymentSteps = [
    {
      title: "Create GitHub Repository",
      steps: [
        "Go to github.com and create a new public repository",
        "Name it 'ml-diabetes-prediction'",
        "Clone the repository to your local machine",
        "Add the following files:"
      ],
      files: [
        { name: "diabetes_model.pkl", desc: "Trained model (download from Colab)" },
        { name: "app.py", desc: "Gradio application code" },
        { name: "requirements.txt", desc: "Python dependencies" },
        { name: "README.md", desc: "Project documentation" }
      ]
    },
    {
      title: "Deploy to Hugging Face",
      steps: [
        "Go to huggingface.co/spaces",
        "Click 'Create new Space'",
        "Name: 'diabetes-prediction'",
        "License: MIT",
        "Select SDK: Gradio",
        "Create Space",
        "Upload all files from your GitHub repo",
        "Wait for deployment (2-3 minutes)",
        "Test the public URL in incognito mode"
      ]
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
      <div className="max-w-6xl mx-auto">
        <div className="bg-white rounded-xl shadow-2xl p-8 mb-6">
          <h1 className="text-4xl font-bold text-indigo-900 mb-2">ML Final Exam Solution</h1>
          <p className="text-gray-600 mb-4">Diabetes Prediction System - Complete Implementation</p>
          <div className="flex items-center gap-2 text-sm text-indigo-600">
            <CheckCircle size={16} />
            <span>All 11 tasks completed (100 marks)</span>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-xl p-6 mb-6">
          <h2 className="text-2xl font-bold text-gray-800 mb-4">📋 Task Checklist</h2>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            {[
              "1. Data Loading (5)",
              "2. Preprocessing (10)",
              "3. Pipeline (10)",
              "4. Model Selection (5)",
              "5. Training (10)",
              "6. Cross-Validation (10)",
              "7. Tuning (10)",
              "8. Best Model (10)",
              "9. Evaluation (10)",
              "10. Gradio UI (10)",
              "11. Deployment (10)"
            ].map((task, idx) => (
              <div key={idx} className="flex items-center gap-2 p-2 bg-green-50 rounded border border-green-200">
                <CheckCircle size={16} className="text-green-600" />
                <span className="text-sm text-gray-700">{task}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-xl p-6 mb-6">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-2xl font-bold text-gray-800">📓 Complete Colab Code</h2>
            <button
              onClick={() => copyToClipboard(colabCode, 'colab')}
              className="flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition"
            >
              {copiedSection === 'colab' ? <CheckCircle size={16} /> : <Copy size={16} />}
              {copiedSection === 'colab' ? 'Copied!' : 'Copy Code'}
            </button>
          </div>
          <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto max-h-96 overflow-y-auto">
            <pre className="text-sm">{colabCode}</pre>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-xl p-6 mb-6">
          <h2 className="text-2xl font-bold text-gray-800 mb-4">🚀 Deployment Files</h2>
          
          <div className="mb-6">
            <div className="flex justify-between items-center mb-2">
              <h3 className="text-lg font-semibold text-gray-700">app.py</h3>
              <button
                onClick={() => copyToClipboard(appPyCode, 'app')}
                className="flex items-center gap-2 px-3 py-1 bg-indigo-600 text-white rounded hover:bg-indigo-700 transition text-sm"
              >
                {copiedSection === 'app' ? <CheckCircle size={14} /> : <Copy size={14} />}
                Copy
              </button>
            </div>
            <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto max-h-64 overflow-y-auto">
              <pre className="text-sm">{appPyCode}</pre>
            </div>
          </div>

          <div className="mb-6">
            <div className="flex justify-between items-center mb-2">
              <h3 className="text-lg font-semibold text-gray-700">requirements.txt</h3>
              <button
                onClick={() => copyToClipboard(requirementsCode, 'req')}
                className="flex items-center gap-2 px-3 py-1 bg-indigo-600 text-white rounded hover:bg-indigo-700 transition text-sm"
              >
                {copiedSection === 'req' ? <CheckCircle size={14} /> : <Copy size={14} />}
                Copy
              </button>
            </div>
            <div className="bg-gray-900 text-gray-100 p-4 rounded-lg">
              <pre className="text-sm">{requirementsCode}</pre>
            </div>
          </div>

          <div>
            <div className="flex justify-between items-center mb-2">
              <h3 className="text-lg font-semibold text-gray-700">README.md</h3>
              <button
                onClick={() => copyToClipboard(readmeCode, 'readme')}
                className="flex items-center gap-2 px-3 py-1 bg-indigo-600 text-white rounded hover:bg-indigo-700 transition text-sm"
              >
                {copiedSection === 'readme' ? <CheckCircle size={14} /> : <Copy size={14} />}
                Copy
              </button>
            </div>
            <div className="bg-gray-900 text-gray-100 p-4 rounded-lg">
              <pre className="text-sm">{readmeCode}</pre>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-xl p-6">
          <h2 className="text-2xl font-bold text-gray-800 mb-4">📦 Deployment Instructions</h2>
          {deploymentSteps.map((section, idx) => (
            <div key={idx} className="mb-6">
              <h3 className="text-xl font-semibold text-indigo-700 mb-3">{section.title}</h3>
              <ol className="list-decimal list-inside space-y-2 ml-2">
                {section.steps.map((step, stepIdx) => (
                  <li key={stepIdx} className="text-gray-700">{step}</li>
                ))}
              </ol>
              {section.files && (
                <div className="mt-3 ml-6 space-y-1">
                  {section.files.map((file, fileIdx) => (
                    <div key={fileIdx} className="flex items-start gap-2">
                      <span className="text-indigo-600 font-mono text-sm">•</span>
                      <span className="text-gray-700">
                        <span className="font-semibold">{file.name}</span> - {file.desc}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>

        <div className="bg-gradient-to-r from-indigo-600 to-purple-600 rounded-xl shadow-xl p-6 mt-6 text-white">
          <h2 className="text-2xl font-bold mb-3">✅ Submission Checklist</h2>
          <div className="space-y-2">
            <div className="flex items-center gap-3">
              <CheckCircle size={20} />
              <span>GitHub repository with all code and files (public)</span>
            </div>
            <div className="flex items-center gap-3">
              <CheckCircle size={20} />
              <span>Google Colab notebook with complete implementation</span>
            </div>
            <div className="flex items-center gap-3">
              <CheckCircle size={20} />
              <span>Hugging Face Space deployment (tested in incognito)</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MLExamSolution;