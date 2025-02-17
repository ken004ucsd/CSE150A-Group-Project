# CSE150A-Group-Project

# CSE150A-Group-Project

IMDB Sentiment Classification Using Naïve Bayes
==============================================

Authors:
- Noah Danan
- Kenny Nguyen
- Yuliana Chavez

Project Overview:
This project classifies IMDB movie reviews as positive or negative using Naïve Bayes with TF-IDF text vectorization.

Dataset:
- Source: IMDB movie reviews dataset.
- Size: 50,000 labeled reviews.

Setup Instructions:
1. Install dependencies:
pip install pandas scikit-learn numpy

markdown
Copy
Edit
2. Clone the repository:
git clone <repo-link> cd <repo-folder>

markdown
Copy
Edit
3. Run `train.py` to train the model:
python train.py

markdown
Copy
Edit
4. Run `predict.py` to test on new reviews.

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
