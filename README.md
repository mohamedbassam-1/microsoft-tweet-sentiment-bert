# 🤖 Microsoft Twitter Sentiment Analysis using BERT

An end-to-end Natural Language Processing (NLP) pipeline designed to classify the sentiment of tweets targeting **Microsoft**. This project utilizes a fine-tuned **BERT (Bidirectional Encoder Representations from Transformers)** model, addresses class imbalances using **SMOTE**, and evaluates model performance with comprehensive visualization tools.

## 🚀 Overview

The project is designed to automatically categorize public sentiment around Microsoft by processing large, raw Twitter datasets. By leveraging transformer-based NLP models along with classical machine learning balancing techniques, the system achieves robust text classification across positive, negative, neutral, and irrelevant sentiments.

## ✨ Features

* **Targeted Ingestion:** Filters massive tweet datasets strictly to target the Microsoft entity.
* **Robust Text Preprocessing:** Strips URLs, mentions, hashtags, digits, and punctuation to clean raw strings.
* **Text Lemmatization:** Normalizes text data using NLTK's `WordNetLemmatizer` and eliminates common English stop words.
* **Imbalance Handling:** Implements Synthetic Minority Over-sampling Technique (SMOTE) with a CountVectorizer baseline.
* **Transformer Tokenization:** Encodes sequences efficiently using the `BertTokenizer` from `bert-base-uncased`.
* **Deep Learning Fine-Tuning:** Fine-tunes `BertForSequenceClassification` using Hugging Face's advanced `Trainer`.
* **Performance Evaluation:** Dynamically builds confusion matrices, classification report heatmaps, and multiclass ROC curves.

## 🛠️ Technologies Used

* Python
* PyTorch
* Transformers (BERT)
* Hugging Face Datasets
* Imbalanced-learn (SMOTE)
* Scikit-Learn
* NLTK
* Matplotlib & Seaborn

## ⚙️ System Workflow

1. **Data Selection:** Import `twitter_training.csv` and isolate rows where the entity is "Microsoft".
2. **Data Cleaning:** Lowercase text, clear special characters, and remove stop words while lemmatizing terms.
3. **Data Balancing:** Tokenize with vector configurations and run SMOTE to address minor class gaps.
4. **Tokenization:** Run the clean strings through the BERT tokenizer with a fixed maximum length of 128.
5. **Model Fine-Tuning:** Feed tokens into `BertForSequenceClassification` via the unified `Trainer` API.
6. **Evaluation Plots:** Map target validation probabilities onto confusion matrices and multidimensional ROC curves.
