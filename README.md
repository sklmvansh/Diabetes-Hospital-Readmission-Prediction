Diabetes Hospital Readmission Prediction

Author: Cedric Izabayo
Date: October 2025
Institution: African Leadership University (ALU)

📋 Project Overview

This project implements and compares traditional machine learning approaches with deep learning architectures for predicting 30-day hospital readmission risk in diabetic patients. Early identification of high-risk patients enables targeted interventions and can significantly reduce healthcare costs.

Problem Statement

Hospital readmissions represent a significant burden on healthcare systems, with an estimated cost of $41 billion annually in the United States alone. For diabetic patients, the 30-day readmission rate ranges from 14-22%, substantially higher than the general population.

This project aims to:

Predict which diabetic patients are at high risk of readmission within 30 days
Compare traditional ML models vs. deep learning approaches
Identify key features most predictive of early readmission
Balance model complexity with performance for practical healthcare deployment
📊 Dataset

Source: Diabetes 130-US Hospitals for Years 1999-2008
Provider: UCI Machine Learning Repository
Citation: Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). Diabetes 130-US Hospitals for Years 1999-2008 [Dataset]. https://doi.org/10.24432/C5230J

Dataset Characteristics

Samples: 100,000+ hospital encounters
Features: 50+ clinical and demographic variables
Target: 30-day readmission status (binary classification)
Time Period: 1999-2008
Hospitals: 130 US hospitals and integrated delivery networks
🎯 Key Features

Clinical Variables

Patient demographics (age, gender, race)
Admission details (type, source, discharge disposition)
Medical history (diagnoses, procedures, medications)
Laboratory results and vital signs
Length of hospital stay
Number of procedures and medications
Engineered Features

Medication complexity indicators
Procedure intensity metrics
Age-based risk stratification
Polypharmacy flags
🛠️ Technologies Used

Core Libraries

Python 3.x
Pandas & NumPy - Data manipulation and analysis
Scikit-learn - Traditional ML algorithms and preprocessing
TensorFlow/Keras - Deep learning framework
Imbalanced-learn - Handling class imbalance (SMOTE)
Matplotlib & Seaborn - Data visualization
Machine Learning Models

Traditional ML

Logistic Regression
Random Forest Classifier
Gradient Boosting Classifier
Support Vector Machine (SVM)
K-Nearest Neighbors
Naive Bayes
Deep Learning

Sequential Neural Networks
Functional API Neural Networks
Custom architectures with dropout and batch normalization
📁 Project Structure

alu-diabetes-readmission-prediction/
├── README.md
├── Summative Assignment Model Training and Evaluation.ipynb
├── Cedric Izabayo Model Training Evaluation.pdf
└── requirements.txt (to be added)
🚀 Getting Started

Prerequisites

Python 3.8+
pip or conda package manager
Installation

Clone the repository:
git clone https://github.com/izabayo7/alu-diabetes-readmission-prediction.git
cd alu-diabetes-readmission-prediction
Install required packages:
pip install pandas numpy scikit-learn tensorflow imbalanced-learn matplotlib seaborn ucimlrepo
Open the Jupyter notebook:
jupyter notebook "Summative Assignment Model Training and Evaluation.ipynb"
📈 Methodology

1. Data Preprocessing

Target Transformation: Convert 3-class to binary (readmitted <30 days vs. not)
Missing Value Imputation: Median for numeric, mode for categorical
Feature Engineering: Create derived clinical indicators
Encoding: One-hot encoding for low cardinality, label encoding for high cardinality
Scaling: RobustScaler for outlier resistance
Train-Validation-Test Split: Stratified 60-20-20 split
2. Handling Class Imbalance

Technique: SMOTE (Synthetic Minority Over-sampling Technique)
Application: Applied only to training data to prevent leakage
Goal: Balance minority class (readmitted patients) to improve recall
3. Model Training & Evaluation

Cross-validation with stratified K-fold
Hyperparameter tuning with GridSearchCV
Early stopping and learning rate scheduling for deep learning
Comprehensive metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC
4. Model Comparison

Performance benchmarking across all models
Feature importance analysis
Confusion matrix visualization
ROC and Precision-Recall curves
📊 Key Results

Results will be documented here after running the full analysis.

Performance Metrics

Best performing model
ROC-AUC scores
F1-scores for positive class
Feature importance rankings
Clinical Insights

Most predictive features
Risk stratification patterns
Model interpretability for healthcare providers
🔍 Key Questions Addressed

How do traditional ML models compare to deep learning for tabular healthcare data?
Which features are most predictive of early readmission?
What is the optimal balance between model complexity and performance?
How can class imbalance be effectively addressed in this medical context?
💡 Clinical Applications

Early identification of high-risk patients enables:

Enhanced discharge planning and patient education
Early post-discharge follow-up appointments
Medication reconciliation and adherence monitoring
Home health services for vulnerable populations
Resource allocation optimization
📖 References

Jencks, S. F., Williams, M. V., & Coleman, E. A. (2009). Rehospitalizations among patients in the Medicare fee-for-service program. New England Journal of Medicine.

Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). Diabetes 130-US Hospitals for Years 1999-2008. UCI Machine Learning Repository.

Medicare Hospital Readmissions Reduction Program (HRRP) - Centers for Medicare & Medicaid Services.

👤 Author
Vansh Malik
