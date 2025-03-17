# CSE150A-Group-Project

Predicting Diabetes Outcome Using (FILL THIS OUT LATER ONCE YOU HAVE YOUR MODEL)
==============================================

Authors:
- Noah Danan
- Kenny Nguyen
- Yuliana Chavez

PEAS Analysis
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


Data Exploration
=============












