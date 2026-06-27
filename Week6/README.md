# ICP-E2ABBE74-2026-REPO
## AI/ML Self-Learning Internship — InternCareerPath
**Intern:** Nishkarsh Jandial
**Portal ID:** ICP-E2ABBE74-2026
**Duration:** 6 Weeks
**Program:** Machine Learning (AI) Self-Learning Internship

---

## Repository Structure

```
ICP-E2ABBE74-2026-REPO/
├── Week1/
│   ├── Project_Selection_Document.md
│   └── 00_Environment_Setup.ipynb
├── Week2/
│   └── Project1_Binary_Classification.ipynb
├── Week3/
│   └── Project1_Completion_Evaluation_Tuning.ipynb
├── Week4/
│   └── Project2_Sentiment_Analysis_Development.ipynb
├── Week5/
│   └── Project2_Sentiment_Analysis_Completion_LSTM.ipynb
└── Week6/
    └── README.md
```

---

## Project 1: Binary Classification Model
**Dataset:** Breast Cancer Wisconsin (sklearn built-in)
**Goal:** Predict whether a tumor is malignant or benign

### Week 2 — Development
- Loaded and explored the dataset (569 samples, 30 features, no missing values)
- Performed EDA: class distribution, correlation heatmap
- Train/test split (80/20) with stratification and StandardScaler
- Trained baseline models: Logistic Regression and Decision Tree

### Week 3 — Completion
- Evaluated both models: confusion matrix, precision, recall, F1-score, ROC-AUC
- Plotted ROC curves for both models
- Visualized top 10 feature importances from Decision Tree
- Applied GridSearchCV hyperparameter tuning on both models
- **Best Model:** Logistic Regression (tuned) — highest accuracy and ROC-AUC

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression (baseline) | ~95% | ~99% |
| Decision Tree (baseline) | ~93% | ~93% |
| Logistic Regression (tuned) | ~97% | ~99% |
| Decision Tree (tuned) | ~95% | ~95% |

---

## Project 2: Sentiment Analysis
**Dataset:** IMDB Movie Reviews (50,000 reviews)
**Goal:** Classify movie reviews as positive or negative

### Week 4 — Development
- Loaded IMDB dataset (25k positive, 25k negative — perfectly balanced)
- Text preprocessing: lowercasing, HTML tag removal, punctuation removal
- TF-IDF vectorization with bigrams (1,2) and 10,000 features
- Trained baseline models: Logistic Regression and Naive Bayes
- Visualized top predictive words for positive and negative sentiment

### Week 5 — Completion
- Tokenized and padded sequences for deep learning (MAX_LEN=200)
- Built and trained a Bidirectional LSTM model using TensorFlow/Keras
- Applied EarlyStopping to prevent overfitting
- Plotted training/validation accuracy and loss curves
- Evaluated all models on test set and compared performance
- Tested model on custom reviews with confidence scores
- **Best Model:** Bidirectional LSTM — captures sequential context in text

| Model | Accuracy |
|-------|----------|
| Naive Bayes (TF-IDF) | ~85% |
| Logistic Regression (TF-IDF) | ~89% |
| Bidirectional LSTM | ~90%+ |

---

## Tools and Technologies
- Python 3.11
- scikit-learn, pandas, numpy
- TensorFlow / Keras
- matplotlib, seaborn
- HuggingFace Datasets
- Jupyter Notebook / Google Colab
- GitHub

---

## Key Learnings
- Clean data often matters more than model complexity
- Simple models (Logistic Regression) can outperform complex ones on structured data
- Deep learning (LSTM) shines on sequential/text data where word order matters
- Hyperparameter tuning with GridSearchCV consistently improves model performance
- EarlyStopping is essential to prevent overfitting in neural networks

