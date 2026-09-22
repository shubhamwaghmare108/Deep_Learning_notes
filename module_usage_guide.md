# Deep Learning Notes – Module Usage Guide

This project covers core machine learning, deep learning, and NLP topics. The modules below match the imports found across the notebooks and Python files in this workspace.

## 1. Core data and scientific computing

### numpy
Use: vector/matrix math, arrays, reshaping, broadcasting, and numerical operations.
Typical tasks in this repo:
- preprocessing arrays for neural networks
- handling embeddings and sequences
- numerical calculations for RNN/LSTM examples

### pandas
Use: tabular data loading and transformation.
Typical tasks in this repo:
- reading CSV/Excel datasets
- cleaning feature columns
- preparing data for modeling and analysis

### matplotlib
Use: charting and visualization.
Typical tasks in this repo:
- plotting training loss and accuracy
- visualizing model performance
- drawing charts for results analysis

### seaborn
Use: statistical plotting and quick data visualization.
Typical tasks in this repo:
- correlation heatmaps
- visual summaries of dataset patterns
- cleaner plots for exploratory analysis

## 2. Machine learning and feature engineering

### scikit-learn
Use: preprocessing, vectorization, model training, and evaluation.
Typical tasks in this repo:
- CountVectorizer and TfidfVectorizer
- train/test splitting
- cosine similarity and metric evaluation
- classification pipelines for text tasks

## 3. NLP and text processing

### nltk
Use: tokenization, stemming, lemmatization, PoS tagging, stopword removal, and corpus access.
Typical tasks in this repo:
- word_tokenize and sent_tokenize
- stopwords and part-of-speech tagging
- stemming and lemmatization utilities
- movie reviews and text cleaning exercises

### spacy
Use: industrial-strength NLP processing.
Typical tasks in this repo:
- tokenization and sentence parsing
- named entity recognition
- part-of-speech tagging
- dependency parsing and visualization

### gensim
Use: word embeddings and topic modeling.
Typical tasks in this repo:
- Word2Vec training
- FastText embeddings
- text similarity and semantic representation

### transformers
Use: Hugging Face pretrained models and tokenizers.
Typical tasks in this repo:
- BERT-like tokenization
- transformer-based embeddings
- text encoding pipelines

### sentence-transformers
Use: sentence-level embeddings and semantic similarity.
Typical tasks in this repo:
- sentence embeddings
- retrieval and similarity tasks

## 4. Deep learning

### torch
Use: PyTorch neural network development.
Typical tasks in this repo:
- custom deep learning models
- embedding layers and transformers
- training loops and optimization

### tensorflow
Use: TensorFlow/Keras-based deep learning workflows.
Typical tasks in this repo:
- LSTM/GRU training
- embedding layers
- model building with Sequential API
- dataset preprocessing and fitting

### keras
Use: high-level neural network API commonly bundled with TensorFlow.
Typical tasks in this repo:
- Embedding layer definition
- Dense, LSTM, Dropout layers
- model compilation and training

## 5. Web scraping and data collection

### beautifulsoup4
Use: HTML parsing and extraction.
Typical tasks in this repo:
- scrubbing raw webpage content
- extracting text from HTML documents
- preparing web scraped data for NLP

### requests
Use: HTTP requests for downloading data or making API calls.
Typical tasks in this repo:
- fetching online datasets
- retrieving web content
- interacting with APIs in data collection workflows

## 6. Databases and external integrations

### pymysql
Use: MySQL database connectivity.
Typical tasks in this repo:
- fetching stock or time-series data from a database
- loading stored datasets into notebook workflows

## 7. Vision and multimodal helpers

### mtcnn
Use: face detection and alignment.
Typical tasks in this repo:
- MTCNN-based face detection workflows
- image preprocessing for detection tasks

## Suggested setup

1. Create a virtual environment:
   python -m venv .venv
2. Activate it:
   - Windows: .venv\Scripts\activate
   - macOS/Linux: source .venv/bin/activate
3. Install dependencies:
   pip install -r requirements.txt
4. Download NLP data if needed:
   python -m nltk.downloader punkt stopwords wordnet omw-1.4 averaged_perceptron_tagger
   python -m spacy download en_core_web_sm

## Files in this workspace that use these modules
- [03.1. text_preprocessing .py](03.1.%20text_preprocessing%20.py)
- [04. text_representation .py](04.%20text_representation%20.py)
- [NLP_Processing.py](NLP_Processing.py)
- [28. LSTM & GRU.ipynb](28.%20LSTM%20&%20GRU.ipynb)
- [embedding.ipynb](embedding.ipynb)
- [Fasttext.ipynb](Fasttext.ipynb)
- [hugging_face.ipynb](hugging_face.ipynb)
- [Stock price using lstm.ipynb](Stock%20price%20using%20lstm.ipynb)

## Notes

The project is a learning-focused notebook set, so you may not need every package installed at once. In practice, you can install the full set for a complete environment, or install a subset based on the notebook you are running.
