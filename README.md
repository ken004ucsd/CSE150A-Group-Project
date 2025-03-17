# CSE150A-Group-Project

Predicting Diabetes Outcome Using (FILL THIS OUT LATER ONCE YOU HAVE YOUR MODEL)
==============================================

Authors:
- Noah Danan
- Kenny Nguyen
- Yuliana Chavez

PART 1 . PEAS Analysis
=============

Performance Measure (P)
-----------------------
Our AI agent is evaluated based on:

- **Accuracy**: How well it evaluates whether a patient will tend to develop a positive case (1) or negative case (0) for diabetes later in life.
- **Precision (Positive Predictive Value)**: Precision measures the proportion of true positive predictions (correctly identified cases of diabetes) out of all the predicted positive cases (both true positives and false positives). In other words, it answers the question: "Of all the patients predicted to have diabetes, how many actually have it?"
- **Computational Efficiency**: The model should be efficient in training and inference.

Environment (E)
---------------
The model operates in a **supervised learning environment**, the system will interact with a dataset that contains medical and lifestyle factors (e.g., age, BMI, glucose levels) and the outcome (whether the patient has diabetes or not). Each patient is categorized as either **positive (1)** or **negative (0).**

Actuators (A)
-------------
The agent’s primary actuator is **predicting** the positive or negative risk factor for a patient to develop diabetes based on their medical and demographic features.

Sensors (S)
-----------
- The model **receives** inputs on medical and personal attributes.

Agent Type
----------
Our AI agent is a **goal-based agent** because it aims to maximize classification accuracy. It can also be considered **utility-based** since it assigns probabilities to predictions.

Probabilistic Modeling
----------------------
The agent fits into **probabilistic modeling** via:

- **Naïve Bayes classifier**: Uses **Bayes’ Theorem** with the assumption that words in a review are independent.
- **TF-IDF**: Weights words based on their frequency across all reviews.
- **Laplace Smoothing**: Handles unseen words in test data. ( FILL THIS OUT LATER ONCE YOU HAVE YOUR MODEL)


PART 2. Data Exploration
=============

Dataset
-----------------------
- The dataset we are using is the Kaggle Diabete's dataset (https://www.kaggle.com/datasets/mathchi/diabetes-data-set/data). This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases.

It is important to note that all patients here are females at least 21 years old of Pima Indian heritage.

The relevant varibles to highlight in this dataset are: 

Pregnancies: Number of times pregnant  
Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test  
BloodPressure: Diastolic blood pressure (mm Hg)  
SkinThickness: Triceps skin fold thickness (mm)  
Insulin: 2-Hour serum insulin (mu U/ml)  
BMI: Body mass index (weight in kg/(height in m)^2)  
DiabetesPedigreeFunction: Diabetes pedigree function  
Age: Age (years)  
Outcome: Class variable (0 or 1)  

All of these are relevant medical predictors that contribute to a patients risk factor in developing diabetes in life.
The key features to predict diabetes are typically those related to glucose levels, insulin, BMI, age, and blood pressure. However, all features should be considered as they may have some impact on the prediction:

Glucose: High glucose levels are strongly correlated with diabetes.  
BMI: Being overweight or obese (higher BMI) is a significant risk factor.  
Age: Older individuals are more likely to develop diabetes.  
BloodPressure: High blood pressure often accompanies diabetes.  
Insulin: Higher insulin levels are associated with diabetes.  
Pregnancies and SkinThickness: These may have secondary effects but can still contribute useful information about a patient’s overall health.  
The Outcome column is the target variable we are trying to predict.  








