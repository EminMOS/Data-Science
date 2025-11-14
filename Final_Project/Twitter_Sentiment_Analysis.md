# Twitter Sentiment Analysis Using Classical & Neural Models

## 1. Introduction
This project studies sentiment classification on Twitter using the Kaggle "Twitter Sentiment Dataset" (Saurabh Shahane, 2021). The dataset contains cleaned tweets (`clean_text`) and sentiment labels in `category` with values -1 (negative), 0 (neutral), +1 (positive). The goal is to compare classical machine learning models against a text-oriented neural model (CNN–LSTM hybrid) and identify which approach is best for multiclass sentiment classification in terms of accuracy and robust F1 (macro).

## 2. Research Questions
1. Which model achieves the best overall and per-class performance for predicting sentiment?
2. How do TF-IDF and Word2Vec features compare when used with classical models?
3. Does a CNN–LSTM hybrid outperform classical ensembles (Voting Classifier) on this dataset?

## 3. Dataset
- Source: Kaggle — Twitter Sentiment Dataset (Saurabh Shahane, 2021).
- Columns: `clean_text` (string), `category` (int: -1, 0, 1).
- Size: (162980, 2)

## 4. Methodology
1. EDA & preprocessing: class distribution, text length, missing data handling, tokenization, stopword removal.
2. Feature pipelines:
    - TF-IDF vectorizer (for baseline classical models).
    - Word2Vec embeddings (gensim): average tweet vectors for classical models.
    - Keras Tokenizer + padding for NN (embedding layer + CNN–LSTM).
3. Models:
    - Baselines: Decision Tree, KNN, Logistic Regression.
    - Ensemble: Voting Classifier (hard or soft) combining best performing classical models.
    - Neural: CNN–LSTM hybrid (Embedding → Conv1D → MaxPool → LSTM → Dense).
4. Hyperparameter tuning: GridSearchCV for classical models; manual / Keras Tuner for NN if time permits.
5. Evaluation: accuracy, precision, recall, F1 (macro & per class), confusion matrix, ROC-AUC (one-vs-rest).

## 5. Results
*Add tables / charts:* compare models by accuracy and F1, confusion matrices, example SHAP/feature importance analysis for top classical model.

## 6. Conclusion & Future Work
- Summarize findings. Discuss limitations (dataset size, class balance), and propose extensions (transformers, domain-specific fine-tuning).

## 7. Reproducibility
- `requirements.txt` provided.
- Scripts:
    - `twitter_sentiment_main.py` — end-to-end pipeline (EDA, preprocessing, training, evaluation).
- How to run: `python Twitter_Sentiment_Analysis.py --data_path Twitter_Data.csv`

## 8. Acknowledgements
HUSSEIN, SHERIF (2021), “Twitter Sentiments Dataset”, Mendeley Data, V1, doi:10.17632/z9zw7nt5h2.1
