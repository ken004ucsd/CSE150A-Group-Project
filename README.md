# CSE150A-Group-Project

IMDB Sentiment Classification Using Naïve Bayes
==============================================

Authors:
- Noah Danan
- Kenny Nguyen
- Yuliana Chavez

[Link to Jupyter Notebook](https://github.com/ken004ucsd/CSE150A-Group-Project/blob/milestone2/Milestone2.ipynb)

UPDATE:
=============

## 1. Training Phase
- Given a dataset with text documents and labels (e.g., spam or not spam), the model:
  1. **Tokenizes** text into words (features).
  2. **Counts word frequencies** per class.
  3. **Estimates probabilities** using **Conditional Probability Tables (CPTs):**

    $$
    P(w|C) = \frac{\text{count}(w, C) + \alpha}{\sum_{\text{all } w'} (\text{count}(w', C) + \alpha)}
    $$
Multinomial Naïve Bayes is a probabilistic model used for text classification. It assumes that words in a document are conditionally independent given the sentiment and calculates the probability of a document belonging to a class based on word frequencies. The model estimates these probabilities using training data and applies Laplace smoothing to handle unseen words.

The classifier calculates conditional probabilities for each word given a sentiment (positive or negative). It assumes that the probability of a review's sentiment is the product of the probabilities of individual words appearing in that review. This simplifies text classification and means the model does not account for word order.

For training, the dataset consists of IMDB movie reviews labeled as positive or negative. The input features are text reviews, and the target variable is sentiment (1 for positive, 0 for negative). Preprocessing includes removing stopwords and applying TF-IDF vectorization.

Our units are vectorized using TF-IDF. TF-IDF converts text into numerical features by assigning weights to words based on their importance. Term frequency (TF) measures how often a word appears in a document, while inverse document frequency (IDF) reduces the weight of common words that appear in many documents. This ensures that words more relevant to sentiment classification have higher weights.

Our preprocessing process converts all text to lowercase (lowercase=True) to ensure consistency. We remove English stop words (stop_words='english') to eliminate common words that do not provide meaningful contributions. Additionally, we apply a regex pattern (token_pattern=r'[a-zA-Z]{2,}') to extract words with at least two letters, effectively ignoring numbers and single-character tokens.

For feature extraction, we use fit_transform(X_train), which learns a vocabulary from X_train, computes TF-IDF scores for words, and converts X_train into a sparse matrix where each row represents a document and each column corresponds to a word’s TF-IDF score. Next, we apply transform(X_test), which uses the same vocabulary from X_train to transform X_test into TF-IDF feature vectors. Our X_train_tfidf and X_test_tfidf are now numerical feature matrices, where each row represents a document and each column corresponds to a word from the learned vocabulary with its TF-IDF weight.

During training, the model learns word probability distributions for each sentiment. When predicting sentiment, it calculates the probability of a review belonging to each class and assigns the one with the highest probability.

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
Our sentiment analysis model performs well, achieving approximately 88% accuracy using the Naïve Bayes classifier. It is computationally efficient, making it suitable for large datasets, and effectively handles text preprocessing using TF-IDF vectorization. However, the model has some limitations, particularly due to the assumption of word independence, which can result in loss of meaning and context. Additionally, it struggles with word order, making it less effective at detecting sarcasm or negation in reviews.

So far, we have only worked on the first feature predicting sentiment from text reviews. However, we need to develop the second feature, which involves analyzing review ratings to enhance sentiment classification. Incorporating numerical ratings can provide additional context and improve sentiment predictions, especially for ambiguous or mixed reviews.

For our next submission, we plan to enhance the model by experimenting with n-gram models (bigrams, trigrams) to capture contextual word relationships and improve sentiment classification. Additionally, we will explore Logistic Regression or Neural Networks for better generalization, as they can provide more nuanced decision boundaries compared to Naïve Bayes. To further refine performance, we will incorporate word embeddings such as Word2Vec and optimize TF-IDF vectorization to improve feature representation. These enhancements aim to create a more robust and accurate sentiment analysis model capable of handling complex language patterns in reviews.








