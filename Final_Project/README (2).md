# Twitter Sentiment Analysis Project

This project implements sentiment classification on Twitter data using both classical machine learning models and a CNN-LSTM neural network hybrid.

## 📁 Project Structure

- `twitter_sentiment_main.py` - Main pipeline script that implements the entire analysis
- `Twitter_Data.csv` - Dataset with cleaned tweets and sentiment labels
- `Twitter_Sentiment_Analysis.md` - Project specification document
- `requirements.txt` - Python dependencies

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Analysis Pipeline

**Basic usage (assuming Twitter_Data.csv is in the same directory):**
```bash
python twitter_sentiment_main.py
```

**With custom data path:**
```bash
python twitter_sentiment_main.py --data_path /path/to/Twitter_Data.csv
```

**With custom random state for reproducibility:**
```bash
python twitter_sentiment_main.py --random_state 123
```

## 📊 What the Pipeline Does

1. **Exploratory Data Analysis (EDA)**
   - Analyzes sentiment distribution
   - Examines text length statistics
   - Creates visualizations

2. **Data Preprocessing**
   - Cleans text (removes URLs, mentions, hashtags)
   - Removes stopwords
   - Tokenizes text

3. **Feature Engineering**
   - **TF-IDF Features**: Creates term frequency-inverse document frequency vectors
   - **Word2Vec Features**: Generates word embeddings and averages them per tweet
   - **Neural Network Features**: Uses Keras tokenizer and padding for sequence data

4. **Model Training**
   - **Classical Models**: Decision Tree, KNN, Logistic Regression (with GridSearchCV)
   - **Ensemble**: Voting Classifier combining best models
   - **Neural Network**: CNN-LSTM hybrid architecture

5. **Evaluation**
   - Accuracy, Precision, Recall, F1-Score (macro and per-class)
   - Confusion matrices
   - ROC curves
   - Model comparison table

## 📈 Output Files

After running the pipeline, you'll get:

- **Visualizations**:
  - `eda_visualization.png` - EDA charts
  - `nn_training_history.png` - Neural network training curves
  - `confusion_matrix_*.png` - Confusion matrix for each model
  - `roc_curves.png` - ROC curves comparison
  - `model_comparison.png` - Bar charts comparing models

- **Data Files**:
  - `model_comparison.csv` - Table comparing all models
  - `best_model.h5` - Best neural network weights

- **Saved Models** (in `models/` directory):
  - `tfidf_vectorizer.pkl` - TF-IDF vectorizer
  - `word2vec_model.bin` - Word2Vec model
  - `nn_tokenizer.pkl` - Neural network tokenizer
  - Individual model files for each classifier

## 🎯 Research Questions Addressed

1. Which model achieves the best overall and per-class performance?
2. How do TF-IDF and Word2Vec features compare?
3. Does the CNN-LSTM hybrid outperform classical ensembles?

## 📝 Notes

- The script automatically downloads required NLTK data (punkt tokenizer and stopwords)
- Training time varies based on dataset size and hardware (expect 10-30 minutes for full pipeline)
- Neural network training uses early stopping to prevent overfitting
- All models are saved for later use/deployment

## 🔧 Troubleshooting

If you encounter memory issues with large datasets:
- Reduce `max_features` in TF-IDF vectorizer
- Decrease `batch_size` in neural network training
- Use a smaller subset of data for initial testing

## 📚 Citation

Dataset source: HUSSEIN, SHERIF (2021), "Twitter Sentiments Dataset", Mendeley Data, V1, doi:10.17632/z9zw7nt5h2.1
