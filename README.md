# CSE150A-Group-Project

IMDB Sentiment Classification Using Naïve Bayes
==============================================

Authors:
- Noah Danan
- Kenny Nguyen
- Yuliana Chavez

Link To Notebook
[Link to Jupyter Notebook](https://github.com/ken004ucsd/CSE150A-Group_Project/blob/milestone2/Milestone2.ipynb)

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








