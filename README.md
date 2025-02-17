# CSE150A-Group-Project

IMDB Sentiment Classification Using Naïve Bayes
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

- **Accuracy**: How well it classifies movie reviews as "positive" or "negative."
- **Precision & Recall**: Ensuring it does not over-classify positive reviews when they're actually negative.
- **Computational Efficiency**: The model should be efficient in training and inference.

Environment (E)
---------------
The model operates in a **supervised learning environment**, using labeled IMDB movie reviews. Each review is categorized as either **positive (1)** or **negative (0).**

Actuators (A)
-------------
The agent’s primary actuator is **predicting** the sentiment of a given review.

Sensors (S)
-----------
- The model **receives** input text reviews.
- **Processes** text using the `TfidfVectorizer` (to extract relevant features).
- Uses `MultinomialNB` for classification.

Agent Type
----------
Our AI agent is a **goal-based agent** because it aims to maximize classification accuracy. It can also be considered **utility-based** since it assigns probabilities to predictions.

Probabilistic Modeling
----------------------
The agent fits into **probabilistic modeling** via:

- **Naïve Bayes classifier**: Uses **Bayes’ Theorem** with the assumption that words in a review are independent.
- **TF-IDF**: Weights words based on their frequency across all reviews.
- **Laplace Smoothing**: Handles unseen words in test data.







Project Overview:
This project classifies IMDB movie reviews as positive or negative using Naïve Bayes with TF-IDF text vectorization.

Dataset:
- Source: IMDB movie reviews dataset.
- Size: 50,000 labeled reviews.

Model Performance:
| Metric        | Score  |
|--------------|--------|
| Accuracy     | 88.5%  |
| Precision    | 87.9%  |
| Recall       | 88.1%  |

Conclusion:
- **Strengths**:
- Performs well (~88% accuracy) using Naïve Bayes.
- Fast training time, efficient for large datasets.
- Handles text preprocessing well using TF-IDF.

- **Limitations**:
- Assumes word independence (may not capture meaning/context well).
- Vulnerable to word order loss (doesn’t detect sarcasm or negation well).

- **Potential Improvements**:
- Use Word Embeddings (e.g., Word2Vec, BERT) for better word relationships.
- Try Deep Learning (LSTMs or Transformers).
- Hyperparameter tuning (optimize smoothing factors in Naïve Bayes).
- Expand preprocessing (lemmatization, bigrams, sentiment-aware techniques).

Next Steps:
- Implement logistic regression or random forest for comparison.
- Introduce pretrained embeddings (e.g., GloVe, FastText).
- Analyze error cases where the model fails.
