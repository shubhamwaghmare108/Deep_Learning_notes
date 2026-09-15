# Text Representation Techniques - Complete Guide

---

## 1. BAG OF WORDS (BoW)

**Kya Hai:** Har word ko count karna aur vector banana. Word order ignore hota hai, sirf frequency matter karti hai.

**Kab Use Karein:** Simple text classification tasks (spam detection, sentiment analysis) jahan computational resources limited hain.

**Scenario:** Email spam filter banana ho jahan "free", "win", "prize" jaise words ki frequency se spam detect karna hai.

### Example 1: NLTK Implementation
```python
from nltk.tokenize import word_tokenize
from collections import Counter

# Sample documents
doc1 = "I love machine learning"
doc2 = "I love deep learning"
doc3 = "Machine learning is awesome"

# Tokenize all documents
all_docs = [doc1, doc2, doc3]
tokenized = [word_tokenize(doc.lower()) for doc in all_docs]

# Create vocabulary (all unique words)
vocab = set()
for tokens in tokenized:
    vocab.update(tokens)

print("Vocabulary:", sorted(vocab))

# Create BoW vectors
bow_vectors = []
for tokens in tokenized:
    word_counts = Counter(tokens)
    # Create vector based on vocabulary
    vector = [word_counts.get(word, 0) for word in sorted(vocab)]
    bow_vectors.append(vector)

# Display results
for i, (doc, vector) in enumerate(zip(all_docs, bow_vectors)):
    print(f"\nDocument {i+1}: {doc}")
    print(f"BoW Vector: {vector}")

# Output shows word frequency for each document
```

### Example 2: spaCy with Custom BoW
```python
import spacy
from collections import Counter

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Documents
docs = [
    "Natural language processing is fun",
    "Machine learning is powerful",
    "Natural language is important"
]

# Process with spaCy and create BoW
processed_docs = [nlp(doc.lower()) for doc in docs]

# Build vocabulary (excluding stopwords and punctuation)
vocab = set()
for doc in processed_docs:
    vocab.update([token.text for token in doc if not token.is_stop and not token.is_punct])

print("Vocabulary:", sorted(vocab))

# Create BoW representation
for i, doc in enumerate(processed_docs):
    # Count words (excluding stopwords)
    tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
    word_freq = Counter(tokens)
    
    print(f"\nDocument {i+1}: {docs[i]}")
    print(f"Word Frequencies: {dict(word_freq)}")
```

### Example 3: scikit-learn CountVectorizer (Industry Standard)
```python
from sklearn.feature_extraction.text import CountVectorizer

# Sample documents
corpus = [
    "I love machine learning",
    "I love deep learning", 
    "Machine learning is awesome"
]

# Create BoW using CountVectorizer
vectorizer = CountVectorizer()

# Fit and transform documents
bow_matrix = vectorizer.fit_transform(corpus)

# Get feature names (vocabulary)
feature_names = vectorizer.get_feature_names_out()

print("Vocabulary:", feature_names)
print("\nBoW Matrix (sparse):")
print(bow_matrix.toarray())

# Show representation for each document
for i, doc in enumerate(corpus):
    print(f"\nDocument {i+1}: {doc}")
    print(f"Vector: {bow_matrix.toarray()[i]}")
```

### Example 4: spaCy with Lemmatization for Better BoW
```python
import spacy
from collections import Counter

# Load model
nlp = spacy.load('en_core_web_sm')

# Documents with different word forms
docs = [
    "The cats are running quickly",
    "A cat runs fast",
    "Running cats run faster"
]

# Process and lemmatize
lemmatized_docs = []
for doc in docs:
    processed = nlp(doc.lower())
    # Get lemmas, exclude stopwords
    lemmas = [token.lemma_ for token in processed 
             if not token.is_stop and not token.is_punct and token.is_alpha]
    lemmatized_docs.append(lemmas)

# Build vocabulary from lemmas
vocab = set()
for lemmas in lemmatized_docs:
    vocab.update(lemmas)

print("Lemmatized Vocabulary:", sorted(vocab))

# Create BoW from lemmas
for i, (doc, lemmas) in enumerate(zip(docs, lemmatized_docs)):
    bow = Counter(lemmas)
    print(f"\nOriginal: {doc}")
    print(f"Lemmas: {lemmas}")
    print(f"BoW: {dict(bow)}")
```

### Example 5: Hugging Face Tokenizer with Manual BoW
```python
from transformers import AutoTokenizer
from collections import Counter

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Documents
docs = [
    "Natural language processing with transformers",
    "Machine learning and deep learning",
    "Natural language understanding is important"
]

# Tokenize all documents
tokenized_docs = [tokenizer.tokenize(doc.lower()) for doc in docs]

# Build vocabulary (excluding special tokens)
vocab = set()
for tokens in tokenized_docs:
    vocab.update([t for t in tokens if not t.startswith('[') and not t.startswith('#')])

print("Vocabulary size:", len(vocab))
print("Sample vocab:", sorted(list(vocab))[:10])

# Create BoW representation
for i, (doc, tokens) in enumerate(zip(docs, tokenized_docs)):
    # Count tokens
    token_freq = Counter([t for t in tokens if not t.startswith('[')])
    print(f"\nDocument {i+1}: {doc}")
    print(f"Token counts: {dict(token_freq)}")
```

---

## 2. TF-IDF (Term Frequency - Inverse Document Frequency)

**Kya Hai:** BoW ka improved version jo rare words ko zyada importance deta hai. Common words (jo har document mein hain) ka weight kam hota hai.

**Kab Use Karein:** Document similarity, search engines, information retrieval - jahan important/unique words identify karne hain.

**Scenario:** Resume screening mein "Python", "AWS" jaise specific skills ko highlight karna, "the", "is" jaise common words ko ignore karna.

### Example 1: NLTK + Manual TF-IDF Calculation
```python
from nltk.tokenize import word_tokenize
import math
from collections import Counter

# Sample corpus
corpus = [
    "machine learning is great",
    "deep learning is powerful",
    "machine learning and deep learning"
]

# Tokenize all documents
tokenized_docs = [word_tokenize(doc.lower()) for doc in corpus]

# Calculate TF (Term Frequency)
def calculate_tf(doc):
    word_count = Counter(doc)
    total_words = len(doc)
    tf = {word: count/total_words for word, count in word_count.items()}
    return tf

# Calculate IDF (Inverse Document Frequency)
def calculate_idf(tokenized_docs):
    num_docs = len(tokenized_docs)
    idf = {}
    
    # Get all unique words
    all_words = set()
    for doc in tokenized_docs:
        all_words.update(doc)
    
    # Calculate IDF for each word
    for word in all_words:
        # Count documents containing this word
        doc_count = sum(1 for doc in tokenized_docs if word in doc)
        idf[word] = math.log(num_docs / doc_count)
    
    return idf

# Calculate TF-IDF
idf_scores = calculate_idf(tokenized_docs)

for i, (doc, tokens) in enumerate(zip(corpus, tokenized_docs)):
    tf = calculate_tf(tokens)
    tfidf = {word: tf[word] * idf_scores[word] for word in tf}
    
    print(f"\nDocument {i+1}: {doc}")
    print("TF-IDF scores:")
    for word, score in sorted(tfidf.items(), key=lambda x: x[1], reverse=True):
        print(f"  {word}: {score:.4f}")
```

### Example 2: spaCy Preprocessing + sklearn TF-IDF
```python
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

# Load spaCy
nlp = spacy.load('en_core_web_sm')

# Documents
docs = [
    "Python is a programming language",
    "Java is a programming language",
    "Python and Java are popular programming languages"
]

# Preprocess with spaCy (lemmatize, remove stopwords)
def preprocess(text):
    doc = nlp(text.lower())
    # Keep only lemmas of alphabetic tokens, excluding stopwords
    tokens = [token.lemma_ for token in doc 
             if token.is_alpha and not token.is_stop]
    return ' '.join(tokens)

# Preprocess all documents
processed_docs = [preprocess(doc) for doc in docs]

print("Preprocessed documents:")
for orig, proc in zip(docs, processed_docs):
    print(f"Original: {orig}")
    print(f"Processed: {proc}\n")

# Calculate TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(processed_docs)

# Display results
feature_names = tfidf_vectorizer.get_feature_names_out()
print("TF-IDF Matrix:")
for i, doc in enumerate(processed_docs):
    print(f"\nDocument {i+1}: {docs[i]}")
    # Get non-zero TF-IDF scores
    scores = tfidf_matrix[i].toarray()[0]
    word_scores = [(feature_names[j], scores[j]) for j in range(len(scores)) if scores[j] > 0]
    word_scores.sort(key=lambda x: x[1], reverse=True)
    for word, score in word_scores:
        print(f"  {word}: {score:.4f}")
```

### Example 3: sklearn TF-IDF with Custom Parameters
```python
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Sample documents
documents = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning is a subset of machine learning",
    "Neural networks are used in deep learning",
    "Artificial intelligence includes machine learning and deep learning"
]

# Create TF-IDF vectorizer with custom parameters
tfidf = TfidfVectorizer(
    max_features=10,        # Keep top 10 features
    min_df=1,               # Minimum document frequency
    max_df=0.8,             # Maximum document frequency (ignore too common words)
    ngram_range=(1, 1),     # Use unigrams only
    stop_words='english'    # Remove English stopwords
)

# Fit and transform
tfidf_matrix = tfidf.fit_transform(documents)

# Create DataFrame for better visualization
df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=tfidf.get_feature_names_out()
)

print("TF-IDF DataFrame:")
print(df)

# Show most important words per document
print("\nMost important words per document:")
for i in range(len(documents)):
    print(f"\nDocument {i+1}: {documents[i]}")
    row = df.iloc[i]
    top_words = row.nlargest(3)
    for word, score in top_words.items():
        if score > 0:
            print(f"  {word}: {score:.4f}")
```

### Example 4: spaCy + TF-IDF for Document Similarity
```python
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy
nlp = spacy.load('en_core_web_sm')

# Documents to compare
docs = [
    "Machine learning is transforming healthcare industry",
    "Artificial intelligence is revolutionizing medical diagnosis",
    "Python is the best programming language for data science",
    "Deep learning models improve medical image analysis"
]

# Preprocess
def preprocess_spacy(text):
    doc = nlp(text.lower())
    return ' '.join([token.lemma_ for token in doc 
                    if token.is_alpha and not token.is_stop])

processed = [preprocess_spacy(d) for d in docs]

# Calculate TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed)

# Calculate cosine similarity
similarity_matrix = cosine_similarity(tfidf_matrix)

print("Document Similarity Matrix:")
print(similarity_matrix)

# Find most similar documents
print("\nMost similar document pairs:")
for i in range(len(docs)):
    for j in range(i+1, len(docs)):
        print(f"Doc {i+1} & Doc {j+1}: {similarity_matrix[i][j]:.4f}")
        print(f"  Doc {i+1}: {docs[i]}")
        print(f"  Doc {j+1}: {docs[j]}\n")
```

### Example 5: Hugging Face Tokenizer + Custom TF-IDF
```python
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Documents
docs = [
    "Natural language processing enables computers to understand human language",
    "Machine translation is an application of natural language processing",
    "Sentiment analysis uses natural language processing techniques"
]

# Tokenize with Hugging Face (get clean tokens)
def tokenize_clean(text):
    tokens = tokenizer.tokenize(text.lower())
    # Remove special tokens and subword markers
    clean_tokens = [t.replace('##', '') for t in tokens if not t.startswith('[')]
    return ' '.join(clean_tokens)

# Process all documents
tokenized_docs = [tokenize_clean(doc) for doc in docs]

print("Tokenized documents:")
for orig, tok in zip(docs, tokenized_docs):
    print(f"Original: {orig}")
    print(f"Tokenized: {tok}\n")

# Apply TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(tokenized_docs)

# Get top terms per document
feature_names = vectorizer.get_feature_names_out()

print("\nTop TF-IDF terms per document:")
for i, doc in enumerate(docs):
    print(f"\nDocument {i+1}: {doc}")
    tfidf_scores = tfidf_matrix[i].toarray()[0]
    # Get indices of top 5 scores
    top_indices = np.argsort(tfidf_scores)[-5:][::-1]
    for idx in top_indices:
        if tfidf_scores[idx] > 0:
            print(f"  {feature_names[idx]}: {tfidf_scores[idx]:.4f}")
```

---

## 3. N-GRAMS (Unigrams, Bigrams, Trigrams)

**Kya Hai:** Consecutive words ka sequence capture karna. Unigram=1 word, Bigram=2 words, Trigram=3 words. Context aur word combinations samajhne ke liye.

**Kab Use Karein:** Phrase detection, named entity recognition, jahan word order important hai (e.g., "not good" vs "good").

**Scenario:** Sentiment analysis mein "not bad" ek positive phrase hai jo unigrams se miss ho jayega, bigrams se catch hoga.

### Example 1: NLTK N-grams Generation
```python
from nltk import ngrams
from nltk.tokenize import word_tokenize

# Sample text
text = "Natural language processing is very interesting"

# Tokenize
tokens = word_tokenize(text.lower())

# Generate unigrams (1-word)
unigrams = list(ngrams(tokens, 1))
print("Unigrams:", unigrams)

# Generate bigrams (2-word sequences)
bigrams = list(ngrams(tokens, 2))
print("\nBigrams:", bigrams)

# Generate trigrams (3-word sequences)
trigrams = list(ngrams(tokens, 3))
print("\nTrigrams:", trigrams)

# More readable format
print("\n--- Readable Format ---")
print("Unigrams:", [' '.join(gram) for gram in unigrams])
print("Bigrams:", [' '.join(gram) for gram in bigrams])
print("Trigrams:", [' '.join(gram) for gram in trigrams])
```

### Example 2: spaCy N-grams with Custom Function
```python
import spacy

# Load model
nlp = spacy.load('en_core_web_sm')

# Text
text = "Machine learning models learn from data"

# Process with spaCy
doc = nlp(text.lower())

# Function to generate n-grams
def generate_ngrams(doc, n):
    ngrams_list = []
    tokens = [token.text for token in doc if not token.is_punct]
    
    for i in range(len(tokens) - n + 1):
        ngram = ' '.join(tokens[i:i+n])
        ngrams_list.append(ngram)
    
    return ngrams_list

# Generate different n-grams
unigrams = generate_ngrams(doc, 1)
bigrams = generate_ngrams(doc, 2)
trigrams = generate_ngrams(doc, 3)

print("Original text:", text)
print("\nUnigrams:", unigrams)
print("Bigrams:", bigrams)
print("Trigrams:", trigrams)

# Count frequencies
from collections import Counter
print("\nBigram frequencies:")
print(Counter(bigrams))
```

### Example 3: sklearn CountVectorizer with N-grams
```python
from sklearn.feature_extraction.text import CountVectorizer

# Sample documents
corpus = [
    "not good at all",
    "very good product",
    "not bad but not great"
]

# Unigrams only
unigram_vectorizer = CountVectorizer(ngram_range=(1, 1))
unigram_matrix = unigram_vectorizer.fit_transform(corpus)

print("Unigrams vocabulary:", unigram_vectorizer.get_feature_names_out())
print("Unigram matrix:\n", unigram_matrix.toarray())

# Bigrams only
bigram_vectorizer = CountVectorizer(ngram_range=(2, 2))
bigram_matrix = bigram_vectorizer.fit_transform(corpus)

print("\nBigrams vocabulary:", bigram_vectorizer.get_feature_names_out())
print("Bigram matrix:\n", bigram_matrix.toarray())

# Combined: Unigrams + Bigrams
combined_vectorizer = CountVectorizer(ngram_range=(1, 2))
combined_matrix = combined_vectorizer.fit_transform(corpus)

print("\nCombined vocabulary:", combined_vectorizer.get_feature_names_out())
print("Combined matrix:\n", combined_matrix.toarray())

# See how "not good" is captured as bigram
print("\nNotice 'not good' is captured as a single feature in bigrams!")
```

### Example 4: spaCy with Lemmatized N-grams
```python
import spacy
from collections import Counter

# Load spaCy
nlp = spacy.load('en_core_web_sm')

# Documents
docs = [
    "The running dogs are fast",
    "A dog runs quickly",
    "Dogs run faster than cats"
]

# Generate lemmatized bigrams
def get_lemma_bigrams(text):
    doc = nlp(text.lower())
    # Get lemmas, exclude stopwords and punctuation
    lemmas = [token.lemma_ for token in doc 
             if not token.is_stop and not token.is_punct and token.is_alpha]
    
    # Create bigrams
    bigrams = []
    for i in range(len(lemmas) - 1):
        bigrams.append(f"{lemmas[i]} {lemmas[i+1]}")
    
    return bigrams

# Process all documents
all_bigrams = []
for doc in docs:
    bigrams = get_lemma_bigrams(doc)
    all_bigrams.extend(bigrams)
    print(f"Text: {doc}")
    print(f"Lemmatized bigrams: {bigrams}\n")

# Count most common lemmatized bigrams
print("Most common lemmatized bigrams:")
print(Counter(all_bigrams).most_common())
```

### Example 5: Hugging Face Tokenizer with N-grams
```python
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import CountVectorizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Documents
docs = [
    "natural language processing is important",
    "machine learning and deep learning are related",
    "natural language understanding requires machine learning"
]

# Tokenize with Hugging Face
def tokenize_for_ngrams(text):
    tokens = tokenizer.tokenize(text.lower())
    # Remove special tokens and join
    clean_tokens = [t.replace('##', '') for t in tokens if not t.startswith('[')]
    return ' '.join(clean_tokens)

# Process documents
tokenized_docs = [tokenize_for_ngrams(doc) for doc in docs]

print("Tokenized documents:")
for orig, tok in zip(docs, tokenized_docs):
    print(f"Original: {orig}")
    print(f"Tokenized: {tok}\n")

# Extract bigrams and trigrams using sklearn
vectorizer = CountVectorizer(ngram_range=(2, 3), max_features=10)
ngram_matrix = vectorizer.fit_transform(tokenized_docs)

print("Top N-grams (bigrams + trigrams):")
feature_names = vectorizer.get_feature_names_out()
print(feature_names)

# Show n-grams per document
print("\nN-gram counts per document:")
for i, doc in enumerate(docs):
    print(f"\nDocument {i+1}: {doc}")
    counts = ngram_matrix[i].toarray()[0]
    for j, count in enumerate(counts):
        if count > 0:
            print(f"  '{feature_names[j]}': {count}")
```

---

## 4. WORD2VEC - CBOW & Skip-gram

**Kya Hai:** Words ko dense vectors mein convert karna jahan similar meaning wale words close hote hain. CBOW context se word predict karta, Skip-gram word se context predict karta.

**Kab Use Karein:** Semantic similarity, word analogies, document clustering - jahan word meaning important hai (king - man + woman = queen).

**Scenario:** Recommendation system mein similar products dhundhna based on description similarity, ya customer queries mein synonyms automatically handle karna.

### Example 1: Gensim Word2Vec Training (NLTK tokenization)
```python
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Sample corpus (in real world, use much larger corpus)
sentences = [
    "machine learning is a subset of artificial intelligence",
    "deep learning is a type of machine learning",
    "neural networks are used in deep learning",
    "natural language processing uses machine learning",
    "computer vision also uses deep learning techniques"
]

# Tokenize sentences
tokenized_sentences = [word_tokenize(sent.lower()) for sent in sentences]

print("Tokenized sentences:")
for sent in tokenized_sentences:
    print(sent)

# Train Word2Vec model (Skip-gram)
model_skipgram = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=100,      # Dimension of word vectors
    window=5,             # Context window size
    min_count=1,          # Minimum word frequency
    sg=1,                 # 1 = Skip-gram, 0 = CBOW
    workers=4
)

# Train CBOW model for comparison
model_cbow = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=0,                 # CBOW
    workers=4
)

# Get word vector
word = "learning"
vector = model_skipgram.wv[word]
print(f"\nVector for '{word}' (first 10 dimensions):")
print(vector[:10])

# Find similar words
similar_words = model_skipgram.wv.most_similar('learning', topn=3)
print(f"\nWords similar to 'learning':")
for word, score in similar_words:
    print(f"  {word}: {score:.4f}")

# Word analogy (if vocabulary is large enough)
# result = model.wv.most_similar(positive=['king', 'woman'], negative=['man'])
```

### Example 2: spaCy Pre-trained Word Vectors
```python
import spacy
import numpy as np

# Load spaCy model with word vectors
nlp = spacy.load('en_core_web_md')  # Medium model with vectors

# Sample words
words = ["king", "queen", "man", "woman", "learning", "education"]

print("Word vectors (spaCy pre-trained):")
for word in words:
    token = nlp(word)[0]
    # Check if word has vector
    if token.has_vector:
        print(f"\n{word}:")
        print(f"  Vector shape: {token.vector.shape}")
        print(f"  First 5 dimensions: {token.vector[:5]}")
    else:
        print(f"\n{word}: No vector available")

# Calculate similarity
word1 = nlp("king")
word2 = nlp("queen")
word3 = nlp("apple")

similarity_kq = word1.similarity(word2)
similarity_ka = word1.similarity(word3)

print(f"\nSimilarity between 'king' and 'queen': {similarity_kq:.4f}")
print(f"Similarity between 'king' and 'apple': {similarity_ka:.4f}")

# Find similar words using spaCy
text = "machine learning deep neural network"
doc = nlp(text)

for token in doc:
    if token.has_vector:
        # Get most similar words from vocabulary
        similar = sorted(
            nlp.vocab,
            key=lambda w: token.similarity(nlp(w.text)[0]) if w.has_vector else 0,
            reverse=True
        )[:5]
        print(f"\nWords similar to '{token.text}':")
        for w in similar:
            if w.text != token.text and w.has_vector:
                print(f"  {w.text}")
```

### Example 3: Gensim Pre-trained Word2Vec (Google News)
```python
# Note: Download pre-trained model first
# from gensim.models import KeyedVectors
# model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# For demonstration, we'll train a small model
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Larger corpus for better training
corpus = [
    "the king sits on the throne",
    "the queen sits beside the king",
    "the prince is the son of king and queen",
    "the princess is the daughter of king and queen",
    "man and woman are equal",
    "a boy grows into a man",
    "a girl grows into a woman",
    "machine learning requires mathematics",
    "deep learning is subset of machine learning",
    "neural networks power deep learning"
]

# Tokenize
tokenized = [word_tokenize(sent.lower()) for sent in corpus]

# Train Word2Vec
model = Word2Vec(
    tokenized,
    vector_size=50,
    window=5,
    min_count=1,
    sg=1,  # Skip-gram
    epochs=100
)

# Word analogies
print("Word Analogies:")
try:
    # king - man + woman = queen
    result = model.wv.most_similar(
        positive=['king', 'woman'],
        negative=['man'],
        topn=1
    )
    print(f"king - man + woman = {result[0][0]}")
except:
    print("Not enough training data for analogy")

# Similarity scores
print(f"\nSimilarity between 'king' and 'queen': {model.wv.similarity('king', 'queen'):.4f}")
print(f"Similarity between 'king' and 'machine': {model.wv.similarity('king', 'machine'):.4f}")

# Most similar words
print("\nWords most similar to 'learning':")
for word, score in model.wv.most_similar('learning', topn=3):
    print(f"  {word}: {score:.4f}")
```

### Example 4: spaCy Document Similarity using Word Vectors
```python
import spacy

# Load model with vectors
nlp = spacy.load('en_core_web_md')

# Documents to compare
docs_text = [
    "Machine learning is a branch of artificial intelligence",
    "AI and machine learning are transforming industries",
    "I love eating pizza and pasta",
    "Deep learning uses neural networks for pattern recognition"
]

# Process documents
docs = [nlp(text) for text in docs_text]

# Calculate document similarities
print("Document Similarity Matrix:")
print("(based on average word vectors)\n")

for i, doc1 in enumerate(docs):
    for j, doc2 in enumerate(docs):
        if i <= j:
            similarity = doc1.similarity(doc2)
            print(f"Doc{i+1} & Doc{j+1}: {similarity:.4f}")
            if i != j:
                print(f"  Doc{i+1}: {docs_text[i]}")
                print(f"  Doc{j+1}: {docs_text[j]}")

# Notice: ML-related documents have higher similarity
# while pizza document has low similarity with ML docs
```

### Example 5: Hugging Face Word2Vec-like Embeddings
```python
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Load model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to get word embedding
def get_word_embedding(word):
    # Tokenize
    inputs = tokenizer(word, return_tensors='pt')
    
    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use last hidden state, average over tokens
    # (word might be split into subwords)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
    
    return embedding.numpy()

# Get embeddings for words
words = ['king', 'queen', 'man', 'woman']

embeddings = {}
for word in words:
    embeddings[word] = get_word_embedding(word)
    print(f"Embedding shape for '{word}': {embeddings[word].shape}")

# Calculate cosine similarity
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

print("\nWord Similarities (BERT embeddings):")
print(f"king vs queen: {cosine_similarity(embeddings['king'], embeddings['queen']):.4f}")
print(f"king vs man: {cosine_similarity(embeddings['king'], embeddings['man']):.4f}")
print(f"queen vs woman: {cosine_similarity(embeddings['queen'], embeddings['woman']):.4f}")

# Note: BERT embeddings are contextual, so single word embeddings
# are less meaningful than in Word2Vec. Context matters!
```

---

## 5. GloVe (Global Vectors for Word Representation)

**Kya Hai:** Word2Vec ka competitor, global word co-occurrence statistics use karta hai. Pre-trained vectors available hain (Wikipedia, Twitter trained).

**Kab Use Karein:** Jab Word2Vec alternative chahiye ya domain-specific pre-trained vectors use karne hain (Twitter text ke liye Twitter GloVe).

**Scenario:** Social media sentiment analysis ke liye Twitter-trained GloVe vectors use karna jo informal language better samajhte hain.

### Example 1: Loading Pre-trained GloVe (Manual)
```python
import numpy as np
from nltk.tokenize import word_tokenize

# Function to load GloVe vectors from file
# Download from: https://nlp.stanford.edu/projects/glove/
def load_glove_vectors(glove_file):
    """Load pre-trained GloVe vectors"""
    embeddings = {}
    
    # For demo, we'll create dummy embeddings
    # In real scenario, download and load actual GloVe file
    print("Note: In production, download actual GloVe file")
    print("Creating dummy embeddings for demonstration...")
    
    # Dummy embeddings (in real case, load from file)
    words = ['king', 'queen', 'man', 'woman', 'learning', 'machine', 'deep', 'neural']
    for word in words:
        embeddings[word] = np.random.randn(50)  # 50-dim vectors
    
    return embeddings

# Load GloVe embeddings
glove_embeddings = load_glove_vectors('glove.6B.50d.txt')

print(f"Loaded {len(glove_embeddings)} word vectors")
print(f"Vector dimension: {len(list(glove_embeddings.values())[0])}")

# Get embedding for a word
if 'learning' in glove_embeddings:
    vector = glove_embeddings['learning']
    print(f"\nGloVe vector for 'learning' (first 10 dims):")
    print(vector[:10])

# Calculate similarity
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

if 'king' in glove_embeddings and 'queen' in glove_embeddings:
    sim = cosine_similarity(glove_embeddings['king'], glove_embeddings['queen'])
    print(f"\nSimilarity between 'king' and 'queen': {sim:.4f}")

# Real code would look like:
"""
with open('glove.6B.50d.txt', 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        embeddings[word] = vector
"""
```

### Example 2: spaCy with GloVe-like Pre-trained Vectors
```python
import spacy
import numpy as np

# Load spaCy with vectors (trained similarly to GloVe)
nlp = spacy.load('en_core_web_lg')  # Large model with better vectors

# Example words
words = ['king', 'queen', 'prince', 'princess', 'computer', 'laptop']

print("Word Vector Information (spaCy):")
for word in words:
    doc = nlp(word)
    token = doc[0]
    
    if token.has_vector:
        print(f"\n{word}:")
        print(f"  Vector shape: {token.vector.shape}")
        print(f"  L2 norm: {token.vector_norm:.2f}")

# Word similarity matrix
print("\n\nWord Similarity Matrix:")
print("      ", end="")
for w in words[:4]:
    print(f"{w:10}", end="")
print()

for word1 in words[:4]:
    print(f"{word1:6}", end="")
    for word2 in words[:4]:
        sim = nlp(word1).similarity(nlp(word2))
        print(f"{sim:10.4f}", end="")
    print()

# Find most dissimilar word (odd one out)
tokens = [nlp(word)[0] for word in ['king', 'queen', 'prince', 'computer']]
target = nlp('royal')[0]

print("\n\nSimilarity with 'royal':")
for token in tokens:
    sim = token.similarity(target)
    print(f"  {token.text}: {sim:.4f}")
```

### Example 3: Using GloVe with Keras/TensorFlow
```python
import numpy as np

# Simulate loading GloVe embeddings for Keras
def create_embedding_matrix(word_index, embeddings_dict, embedding_dim=50):
    """
    Create embedding matrix for Keras Embedding layer
    word_index: dictionary mapping words to indices
    embeddings_dict: pre-loaded GloVe vectors
    """
    vocab_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    
    for word, idx in word_index.items():
        embedding_vector = embeddings_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[idx] = embedding_vector
    
    return embedding_matrix

# Example vocabulary from your corpus
word_index = {
    'machine': 1,
    'learning': 2,
    'deep': 3,
    'neural': 4,
    'network': 5
}

# Simulate GloVe embeddings
glove_dict = {
    'machine': np.random.randn(50),
    'learning': np.random.randn(50),
    'deep': np.random.randn(50),
    'neural': np.random.randn(50),
    'network': np.random.randn(50)
}

# Create embedding matrix
embedding_matrix = create_embedding_matrix(word_index, glove_dict, 50)

print(f"Embedding matrix shape: {embedding_matrix.shape}")
print(f"Vocabulary size: {len(word_index)}")
print(f"Embedding dimension: 50")

# This matrix can be used in Keras:
"""
from tensorflow.keras.layers import Embedding

embedding_layer = Embedding(
    input_dim=len(word_index) + 1,
    output_dim=50,
    weights=[embedding_matrix],
    trainable=False  # Freeze GloVe weights
)
"""

print("\nSample embedding for 'learning' (first 10 dims):")
print(embedding_matrix[word_index['learning']][:10])
```

### Example 4: spaCy Sentence Embeddings using Word Vectors
```python
import spacy
import numpy as np

# Load model
nlp = spacy.load('en_core_web_md')

# Sentences
sentences = [
    "Machine learning is powerful",
    "Artificial intelligence is amazing",
    "I love pizza and pasta",
    "Deep learning uses neural networks"
]

# Process sentences
docs = [nlp(sent) for sent in sentences]

# Get sentence embeddings (average of word vectors)
print("Sentence Embeddings (averaged word vectors):")
for i, (sent, doc) in enumerate(zip(sentences, docs)):
    # Doc.vector is average of token vectors
    print(f"\nSentence {i+1}: {sent}")
    print(f"  Embedding shape: {doc.vector.shape}")
    print(f"  First 5 dimensions: {doc.vector[:5]}")

# Calculate sentence similarities
print("\n\nSentence Similarity Matrix:")
for i in range(len(docs)):
    for j in range(i+1, len(docs)):
        sim = docs[i].similarity(docs[j])
        print(f"\nSent{i+1} vs Sent{j+1}: {sim:.4f}")
        print(f"  {sentences[i]}")
        print(f"  {sentences[j]}")
```

### Example 5: Hugging Face with GloVe-style Static Embeddings
```python
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Load model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Get static-like embeddings (from embedding layer only)
def get_static_embedding(word):
    """Get embedding from BERT's embedding layer (token embeddings only)"""
    # Tokenize
    token_id = tokenizer.encode(word, add_special_tokens=False)[0]
    
    # Get embedding from embedding layer (similar to GloVe)
    with torch.no_grad():
        embedding = model.embeddings.word_embeddings.weight[token_id]
    
    return embedding.numpy()

# Words to embed
words = ['machine', 'learning', 'computer', 'science', 'data']

print("Static Token Embeddings (BERT embedding layer):")
embeddings = {}
for word in words:
    embeddings[word] = get_static_embedding(word)
    print(f"{word}: shape {embeddings[word].shape}")

# Calculate similarities
def cosine_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

print("\nWord Similarities:")
print(f"machine & learning: {cosine_sim(embeddings['machine'], embeddings['learning']):.4f}")
print(f"machine & computer: {cosine_sim(embeddings['machine'], embeddings['computer']):.4f}")
print(f"computer & science: {cosine_sim(embeddings['computer'], embeddings['science']):.4f}")

# Note: These are static embeddings from BERT's embedding layer
# In actual BERT, context changes these embeddings dynamically
```

---

## 6. FASTTEXT

**Kya Hai:** Word2Vec ka improved version jo subword information use karta hai. Unknown/misspelled words ko bhi handle kar sakta kyunki character n-grams use karta.

**Kab Use Karein:** Morphologically rich languages, noisy text (typos), OOV (out-of-vocabulary) words handle karne ke liye.

**Scenario:** Medical text mein rare terms hain jo training data mein nahi the, FastText subword se approximate embedding bana dega.

### Example 1: Gensim FastText Training
```python
from gensim.models import FastText
from nltk.tokenize import word_tokenize

# Training corpus
sentences = [
    "machine learning algorithms learn from data",
    "deep learning is a subset of machine learning",
    "neural networks power deep learning models",
    "natural language processing uses machine learning",
    "computer vision applies deep learning techniques",
    "reinforcement learning is another branch",
    "unsupervised learning finds patterns in unlabeled data"
]

# Tokenize
tokenized = [word_tokenize(sent.lower()) for sent in sentences]

print("Training FastText model...")

# Train FastText model
model = FastText(
    sentences=tokenized,
    vector_size=100,      # Embedding dimension
    window=5,             # Context window
    min_count=1,          # Minimum word frequency
    sg=1,                 # Skip-gram (1) or CBOW (0)
    min_n=3,              # Min character n-gram length
    max_n=6,              # Max character n-gram length
    epochs=10
)

print("Training complete!")

# Get vector for known word
known_word = "learning"
vector = model.wv[known_word]
print(f"\nVector for '{known_word}' (first 10 dims):")
print(vector[:10])

# Get vector for OOV word (not in training)
# FastText can generate vector using subword information
oov_word = "learnings"  # Plural form not in training
vector_oov = model.wv[oov_word]
print(f"\nVector for OOV word '{oov_word}' (first 10 dims):")
print(vector_oov[:10])

# Check similarity even with typo/variant
print(f"\nSimilarity between 'learning' and 'learnings': {model.wv.similarity('learning', 'learnings'):.4f}")

# Most similar words
print(f"\nWords similar to 'learning':")
for word, score in model.wv.most_similar('learning', topn=3):
    print(f"  {word}: {score:.4f}")
```

### Example 2: spaCy-wrapped FastText (Third-party)
```python
import spacy
# Note: This requires spacy-fasttext package
# pip install spacy-fasttext

# For demonstration, we'll show how it would work
print("FastText with spaCy (conceptual example):")
print("Install: pip install spacy-fasttext")

# In actual implementation:
"""
import spacy
from spacy_fasttext import FastText

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('fasttext', config={'model_path': 'path/to/fasttext.bin'})

# Process text
doc = nlp("This is a sample text")

# Get FastText embeddings
for token in doc:
    fasttext_vector = token._.fasttext_vector
"""

# Alternative: Using Gensim's FastText with spaCy preprocessing
from gensim.models import FastText as GensimFastText

nlp = spacy.load('en_core_web_sm')

# Documents
texts = [
    "running runners run",
    "quickly quick quicker",
    "learning learned learner"
]

# Tokenize with spaCy
tokenized = []
for text in texts:
    doc = nlp(text.lower())
    tokens = [token.text for token in doc if not token.is_punct]
    tokenized.append(tokens)

# Train FastText
fasttext_model = GensimFastText(
    sentences=tokenized,
    vector_size=50,
    window=3,
    min_count=1,
    epochs=20,
    min_n=2,
    max_n=5
)

# Test with word forms
base_word = "run"
variants = ["run", "running", "runner", "runs"]

print("\nFastText handling of word variants:")
for word in variants:
    try:
        vector = fasttext_model.wv[word]
        print(f"{word}: vector available (shape: {vector.shape})")
        
        # Similarity to base word
        if word != base_word:
            sim = fasttext_model.wv.similarity(base_word, word)
            print(f"  Similarity to '{base_word}': {sim:.4f}")
    except KeyError:
        print(f"{word}: not in vocabulary (FastText should still work!)")
```

### Example 3: Pre-trained FastText Vectors
```python
# Downloading and using pre-trained FastText
# Download from: https://fasttext.cc/docs/en/english-vectors.html

from gensim.models import FastText
from gensim.models.fasttext import load_facebook_model
import numpy as np

# For demonstration, simulate pre-trained usage
print("Using Pre-trained FastText Vectors:")
print("Download from: https://fasttext.cc/docs/en/english-vectors.html")

# Simulated functionality (actual code shown below)
def simulate_pretrained_fasttext():
    """
    In real scenario, load pre-trained model like:
    model = load_facebook_model('cc.en.300.bin')
    """
    
    # Demonstrate key features
    print("\nKey FastText Features:")
    print("1. Handles OOV words using subword info")
    print("2. Works with misspellings and rare words")
    print("3. Morphologically aware")
    
    # Example words including OOV
    words = [
        "coronavirus",     # Might be OOV in old models
        "antibiotic",      # In vocabulary
        "antibiotics",     # Variant
        "runningg",        # Typo
        "COVID-19"         # Recent term
    ]
    
    print("\nFastText can generate embeddings for:")
    for word in words:
        print(f"  - {word}")
    
    return None

simulate_pretrained_fasttext()

# Actual loading code (when you have the model file):
"""
from gensim.models.fasttext import load_facebook_model

# Load pre-trained model
model = load_facebook_model('cc.en.300.bin')

# Get vector for any word (including OOV)
vector = model.wv['uncommonword']

# Find similar words
similar = model.wv.most_similar('computer')

# Handle misspellings
typo_vector = model.wv['computr']  # Still works!
"""
```

### Example 4: spaCy + FastText for Domain-Specific Text
```python
import spacy
from gensim.models import FastText

# Load spaCy for preprocessing
nlp = spacy.load('en_core_web_sm')

# Domain-specific corpus (medical example)
medical_texts = [
    "Patient diagnosed with hypertension and diabetes",
    "Prescribed antihypertensive medication for blood pressure",
    "Diabetic patient needs insulin therapy",
    "Hypertensive crisis requires immediate treatment",
    "Antidiabetic drugs control glucose levels"
]

# Preprocess with spaCy
def preprocess_medical(text):
    doc = nlp(text.lower())
    # Keep medical terms, remove stopwords
    tokens = [token.text for token in doc 
             if not token.is_stop and not token.is_punct and token.is_alpha]
    return tokens

# Tokenize all texts
tokenized_medical = [preprocess_medical(text) for text in medical_texts]

print("Training domain-specific FastText...")

# Train FastText on medical corpus
medical_fasttext = FastText(
    sentences=tokenized_medical,
    vector_size=100,
    window=5,
    min_count=1,
    epochs=50,  # More epochs for small corpus
    min_n=3,
    max_n=6
)

# Test with medical terms and variants
medical_terms = [
    "hypertension",
    "hypertensive",  # Variant
    "antihypertensive",  # Compound
    "hypertensin"  # Typo/variant
]

print("\nMedical term embeddings:")
for term in medical_terms:
    try:
        vector = medical_fasttext.wv[term]
        print(f"\n{term}:")
        print(f"  Vector available: Yes")
        
        # Find similar terms
        similar = medical_fasttext.wv.most_similar(term, topn=2)
        print(f"  Similar terms:")
        for word, score in similar:
            print(f"    {word}: {score:.4f}")
    except KeyError:
        print(f"{term}: Error (should not happen with FastText!)")

# Similarity between related terms
print("\n\nTerm Similarities:")
print(f"hypertension vs hypertensive: {medical_fasttext.wv.similarity('hypertension', 'hypertensive'):.4f}")
print(f"diabetes vs diabetic: {medical_fasttext.wv.similarity('diabetes', 'diabetic'):.4f}")
```

### Example 5: Hugging Face Tokenizer + FastText Approach
```python
from transformers import AutoTokenizer
from gensim.models import FastText
import numpy as np

# Load tokenizer for preprocessing
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Corpus
texts = [
    "transformers revolutionize natural language processing",
    "bert model uses bidirectional encoding",
    "gpt models generate coherent text",
    "roberta improves on bert architecture"
]

# Tokenize with Hugging Face
def tokenize_with_hf(text):
    tokens = tokenizer.tokenize(text.lower())
    # Clean special tokens and subword markers
    clean = [t.replace('##', '') for t in tokens if not t.startswith('[')]
    return clean

# Process all texts
tokenized = [tokenize_with_hf(text) for text in texts]

print("Tokenized with Hugging Face:")
for orig, tok in zip(texts, tokenized):
    print(f"Original: {orig}")
    print(f"Tokens: {tok}\n")

# Train FastText on HF-tokenized data
print("Training FastText on HF-tokenized corpus...")

fasttext = FastText(
    sentences=tokenized,
    vector_size=100,
    window=5,
    min_count=1,
    epochs=30,
    min_n=2,
    max_n=5
)

# Test with technical terms
tech_terms = ['transformer', 'transformers', 'bert', 'gpt', 'roberta']

print("\nTechnical term embeddings:")
for term in tech_terms:
    vector = fasttext.wv[term]
    print(f"{term}: shape {vector.shape}")

# Find similar terms
print("\n\nSimilar to 'transformer':")
for word, score in fasttext.wv.most_similar('transformer', topn=3):
    print(f"  {word}: {score:.4f}")

# Handle variant/typo
typo = "transformr"
print(f"\n\nFastText handles typo '{typo}':")
typo_vector = fasttext.wv[typo]
print(f"Vector generated: {typo_vector.shape}")
print(f"Similarity to 'transformer': {fasttext.wv.similarity('transformer', typo):.4f}")
```

---

**(Continuing in next message due to length...)**

Let me continue with remaining techniques:
    

# Text Representation Techniques - Part 2

---

## 7. ONE-HOT ENCODING

**Kya Hai:** Har word ko binary vector mein convert karna jahan sirf ek position 1 hai baaki 0. Vocabulary size = vector size. Sparse representation hai.

**Kab Use Karein:** Small vocabulary wale simple tasks, categorical features encode karne ke liye. Deep learning mein starting layer ke liye.

**Scenario:** 10-20 categories (product types, sentiment labels) ko encode karna jahan ordering matter nahi karta.

### Example 1: NLTK + Manual One-Hot Encoding
```python
from nltk.tokenize import word_tokenize
import numpy as np

# Sample sentences
sentences = [
    "I love machine learning",
    "I love deep learning",
    "Machine learning is great"
]

# Tokenize and build vocabulary
all_tokens = []
for sent in sentences:
    tokens = word_tokenize(sent.lower())
    all_tokens.extend(tokens)

# Create vocabulary (unique words)
vocab = sorted(set(all_tokens))
print("Vocabulary:", vocab)
print(f"Vocabulary size: {len(vocab)}")

# Create word to index mapping
word_to_idx = {word: idx for idx, word in enumerate(vocab)}

# Function to create one-hot vector
def one_hot_encode(word, vocab_size, word_to_idx):
    vector = np.zeros(vocab_size)
    if word in word_to_idx:
        vector[word_to_idx[word]] = 1
    return vector

# Encode some words
words_to_encode = ['love', 'machine', 'learning']

print("\nOne-Hot Encoded Vectors:")
for word in words_to_encode:
    vector = one_hot_encode(word, len(vocab), word_to_idx)
    print(f"\n{word}:")
    print(f"  Vector: {vector}")
    print(f"  Index of 1: {np.argmax(vector)}")

# Encode entire sentence
sentence = "I love learning"
tokens = word_tokenize(sentence.lower())

print(f"\n\nEncoding sentence: '{sentence}'")
sentence_vectors = []
for token in tokens:
    if token in word_to_idx:
        vec = one_hot_encode(token, len(vocab), word_to_idx)
        sentence_vectors.append(vec)
        print(f"{token}: position {word_to_idx[token]}")

# Stack vectors
sentence_matrix = np.array(sentence_vectors)
print(f"\nSentence matrix shape: {sentence_matrix.shape}")
```

### Example 2: spaCy Preprocessing + One-Hot
```python
import spacy
import numpy as np

# Load spaCy
nlp = spacy.load('en_core_web_sm')

# Documents
docs = [
    "Natural language processing",
    "Machine learning algorithms",
    "Deep neural networks"
]

# Process and build vocabulary
all_lemmas = []
for doc_text in docs:
    doc = nlp(doc_text.lower())
    lemmas = [token.lemma_ for token in doc if token.is_alpha]
    all_lemmas.extend(lemmas)

# Create vocabulary from lemmas
vocab = sorted(set(all_lemmas))
vocab_size = len(vocab)
word_to_idx = {word: idx for idx, word in enumerate(vocab)}

print(f"Vocabulary: {vocab}")
print(f"Vocabulary size: {vocab_size}")

# One-hot encoding function
def create_one_hot(word, vocab_size, word_to_idx):
    vector = np.zeros(vocab_size)
    if word in word_to_idx:
        vector[word_to_idx[word]] = 1
    return vector

# Encode lemmatized documents
print("\n\nOne-Hot Encoded Documents:")
for doc_text in docs:
    doc = nlp(doc_text.lower())
    lemmas = [token.lemma_ for token in doc if token.is_alpha]
    
    print(f"\nOriginal: {doc_text}")
    print(f"Lemmas: {lemmas}")
    
    # Create one-hot matrix for document
    doc_matrix = np.array([create_one_hot(lemma, vocab_size, word_to_idx) 
                           for lemma in lemmas])
    print(f"Matrix shape: {doc_matrix.shape}")
    print(f"Non-zero positions: {[word_to_idx[lemma] for lemma in lemmas]}")
```

### Example 3: Keras/TensorFlow One-Hot Encoding
```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
import numpy as np

# Sample texts
texts = [
    "machine learning is powerful",
    "deep learning is amazing",
    "natural language processing"
]

# Create tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

# Get vocabulary
vocab_size = len(tokenizer.word_index) + 1  # +1 for padding
print(f"Vocabulary size: {vocab_size}")
print(f"Word index: {tokenizer.word_index}")

# Convert texts to sequences
sequences = tokenizer.texts_to_sequences(texts)
print(f"\nSequences: {sequences}")

# One-hot encode each word in sequences
print("\n\nOne-Hot Encoded:")
for i, (text, seq) in enumerate(zip(texts, sequences)):
    print(f"\nText {i+1}: {text}")
    print(f"Sequence: {seq}")
    
    # One-hot encode the sequence
    one_hot = to_categorical(seq, num_classes=vocab_size)
    print(f"One-hot shape: {one_hot.shape}")
    print(f"First word '{text.split()[0]}' one-hot:")
    print(one_hot[0])

# Alternative: using tokenizer's texts_to_matrix with 'binary' mode
one_hot_matrix = tokenizer.texts_to_matrix(texts, mode='binary')
print(f"\n\nTexts to Matrix (binary mode):")
print(f"Shape: {one_hot_matrix.shape}")
print(f"Matrix:\n{one_hot_matrix}")
```

### Example 4: spaCy + scikit-learn OneHotEncoder
```python
import spacy
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Load spaCy
nlp = spacy.load('en_core_web_sm')

# Texts
texts = [
    "cat dog bird",
    "dog bird fish",
    "cat fish"
]

# Process with spaCy
processed = []
for text in texts:
    doc = nlp(text.lower())
    tokens = [token.text for token in doc]
    processed.append(tokens)

# Flatten all tokens for vocabulary
all_tokens = [token for doc in processed for token in doc]
unique_tokens = sorted(set(all_tokens))

print(f"Vocabulary: {unique_tokens}")

# Map tokens to indices
token_to_idx = {token: idx for idx, token in enumerate(unique_tokens)}

# Convert to integer indices for sklearn
X = []
for doc in processed:
    doc_indices = [[token_to_idx[token]] for token in doc]
    X.extend(doc_indices)

X = np.array(X)

# Create OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
one_hot_encoded = encoder.fit_transform(X)

print(f"\n\nOne-Hot Encoded Shape: {one_hot_encoded.shape}")
print(f"First few vectors:\n{one_hot_encoded[:5]}")

# Decode back
print("\n\nDecoding back to words:")
for i in range(min(5, len(one_hot_encoded))):
    idx = np.argmax(one_hot_encoded[i])
    word = unique_tokens[idx]
    print(f"Vector {i}: {word}")
```

### Example 5: Hugging Face Tokenizer to One-Hot
```python
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Texts
texts = [
    "hello world",
    "machine learning",
    "natural language processing"
]

# Tokenize
print("Tokenization:")
for text in texts:
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    print(f"\nText: {text}")
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")

# Convert token IDs to one-hot
def ids_to_onehot(token_ids, vocab_size):
    """Convert token IDs to one-hot vectors"""
    token_ids_tensor = torch.LongTensor(token_ids)
    one_hot = F.one_hot(token_ids_tensor, num_classes=vocab_size)
    return one_hot

# Get vocab size
vocab_size = tokenizer.vocab_size
print(f"\n\nVocabulary size: {vocab_size}")

# Encode first text to one-hot
text = "hello world"
token_ids = tokenizer.encode(text, add_special_tokens=False)
one_hot_vectors = ids_to_onehot(token_ids, vocab_size)

print(f"\n\nOne-Hot Encoding for '{text}':")
print(f"Shape: {one_hot_vectors.shape}")
print(f"Each vector has {vocab_size} dimensions")
print(f"\nFirst token 'hello' (ID: {token_ids[0]}):")
print(f"  Non-zero position: {token_ids[0]}")
print(f"  Verification: {one_hot_vectors[0][token_ids[0]].item()}")
```

---

## 8. COUNT VECTORIZATION

**Kya Hai:** Bag of Words ka implementation - har word ki frequency count karke sparse matrix banana. Similar to BoW but optimized implementation.

**Kab Use Karein:** Text classification, clustering jahan simple frequency-based features chahiye. Production-ready implementation ke liye.

**Scenario:** Spam detection mein words ki frequency matter karti hai - zyada "free", "win" = spam likely.

### Example 1: NLTK Tokenization + Manual Count Vector
```python
from nltk.tokenize import word_tokenize
from collections import Counter
import numpy as np

# Documents
docs = [
    "I love machine learning and deep learning",
    "I love natural language processing",
    "Machine learning is subset of artificial intelligence"
]

# Tokenize all
tokenized_docs = [word_tokenize(doc.lower()) for doc in docs]

# Build vocabulary
vocab = set()
for tokens in tokenized_docs:
    vocab.update(tokens)
vocab = sorted(vocab)

print(f"Vocabulary: {vocab}")
print(f"Vocabulary size: {len(vocab)}")

# Create word to index mapping
word_to_idx = {word: idx for idx, word in enumerate(vocab)}

# Create count vectors
count_vectors = []
for tokens in tokenized_docs:
    # Count word frequencies
    word_counts = Counter(tokens)
    
    # Create vector based on vocabulary
    vector = np.zeros(len(vocab))
    for word, count in word_counts.items():
        if word in word_to_idx:
            vector[word_to_idx[word]] = count
    
    count_vectors.append(vector)

# Display results
print("\n\nCount Vectors:")
for i, (doc, vector) in enumerate(zip(docs, count_vectors)):
    print(f"\nDocument {i+1}: {doc}")
    print(f"Vector: {vector}")
    
    # Show non-zero counts
    non_zero_words = [(vocab[idx], int(count)) 
                      for idx, count in enumerate(vector) if count > 0]
    print(f"Word counts: {non_zero_words}")
```

### Example 2: spaCy Preprocessing + Count Vectorization
```python
import spacy
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# Load spaCy
nlp = spacy.load('en_core_web_sm')

# Documents
docs = [
    "The quick brown fox jumps over the lazy dog",
    "The dog was lazy and slept all day",
    "The fox was quick and clever"
]

# Preprocess with spaCy (lemmatization, remove stopwords)
def preprocess_spacy(text):
    doc = nlp(text.lower())
    # Get lemmas, exclude stopwords and punctuation
    lemmas = [token.lemma_ for token in doc 
             if not token.is_stop and not token.is_punct and token.is_alpha]
    return ' '.join(lemmas)

# Preprocess all documents
processed_docs = [preprocess_spacy(doc) for doc in docs]

print("Preprocessed documents:")
for orig, proc in zip(docs, processed_docs):
    print(f"Original: {orig}")
    print(f"Processed: {proc}\n")

# Apply Count Vectorization
vectorizer = CountVectorizer()
count_matrix = vectorizer.fit_transform(processed_docs)

# Get feature names
feature_names = vectorizer.get_feature_names_out()

# Convert to DataFrame for better visualization
df = pd.DataFrame(
    count_matrix.toarray(),
    columns=feature_names
)

print("Count Vector Matrix:")
print(df)

# Show counts for each document
print("\n\nWord counts per document:")
for i, (orig_doc, proc_doc) in enumerate(zip(docs, processed_docs)):
    print(f"\nDocument {i+1}: {orig_doc}")
    counts = count_matrix[i].toarray()[0]
    word_counts = [(feature_names[j], int(counts[j])) 
                   for j in range(len(counts)) if counts[j] > 0]
    print(f"Counts: {word_counts}")
```

### Example 3: sklearn CountVectorizer with N-grams
```python
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Sample corpus
corpus = [
    "not good at all",
    "very good product",
    "not bad experience"
]

print("=" * 50)
print("UNIGRAMS ONLY")
print("=" * 50)

# Unigrams only
cv_unigram = CountVectorizer(ngram_range=(1, 1))
counts_unigram = cv_unigram.fit_transform(corpus)

print(f"\nVocabulary: {cv_unigram.get_feature_names_out()}")
print(f"\nCount matrix:")
print(counts_unigram.toarray())

print("\n" + "=" * 50)
print("BIGRAMS ONLY")
print("=" * 50)

# Bigrams only
cv_bigram = CountVectorizer(ngram_range=(2, 2))
counts_bigram = cv_bigram.fit_transform(corpus)

print(f"\nVocabulary: {cv_bigram.get_feature_names_out()}")
print(f"\nCount matrix:")
print(counts_bigram.toarray())

print("\n" + "=" * 50)
print("UNIGRAMS + BIGRAMS")
print("=" * 50)

# Combined: Unigrams + Bigrams
cv_combined = CountVectorizer(ngram_range=(1, 2))
counts_combined = cv_combined.fit_transform(corpus)

print(f"\nVocabulary: {cv_combined.get_feature_names_out()}")
print(f"\nCount matrix:")
print(counts_combined.toarray())

# Analyze specific document
print("\n\nAnalyzing: 'not good at all'")
doc_idx = 0
features = cv_combined.get_feature_names_out()
counts = counts_combined[doc_idx].toarray()[0]

print("Features with counts:")
for feature, count in zip(features, counts):
    if count > 0:
        print(f"  '{feature}': {count}")
```

### Example 4: spaCy + Custom Count Vectorization with POS Tags
```python
import spacy
from collections import Counter
import numpy as np

# Load spaCy
nlp = spacy.load('en_core_web_sm')

# Documents
docs = [
    "The cat quickly ran across the street",
    "Dogs run faster than cats",
    "The quick brown fox jumps"
]

# Process with spaCy
processed_docs = [nlp(doc.lower()) for doc in docs]

# Extract only nouns and verbs
def extract_pos_tokens(doc, pos_tags=['NOUN', 'VERB']):
    """Extract tokens with specific POS tags"""
    return [token.lemma_ for token in doc if token.pos_ in pos_tags]

# Get tokens for each document
filtered_docs = [extract_pos_tokens(doc) for doc in processed_docs]

print("Filtered tokens (nouns and verbs only):")
for orig, filtered in zip(docs, filtered_docs):
    print(f"\nOriginal: {orig}")
    print(f"Filtered: {filtered}")

# Build vocabulary
vocab = set()
for tokens in filtered_docs:
    vocab.update(tokens)
vocab = sorted(vocab)

print(f"\n\nVocabulary: {vocab}")

# Create count vectors
word_to_idx = {word: idx for idx, word in enumerate(vocab)}

count_vectors = []
for tokens in filtered_docs:
    word_counts = Counter(tokens)
    vector = np.zeros(len(vocab))
    for word, count in word_counts.items():
        vector[word_to_idx[word]] = count
    count_vectors.append(vector)

# Display
print("\n\nCount vectors:")
for i, (doc, vector) in enumerate(zip(docs, count_vectors)):
    print(f"\nDocument {i+1}: {doc}")
    print(f"Vector: {vector}")
```

### Example 5: Hugging Face Tokenizer + Count Matrix
```python
from transformers import AutoTokenizer
from collections import Counter
import numpy as np

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Documents
docs = [
    "machine learning and deep learning are related",
    "natural language processing uses machine learning",
    "deep learning is a subset of machine learning"
]

# Tokenize all documents
tokenized_docs = [tokenizer.tokenize(doc.lower()) for doc in docs]

print("Tokenized documents:")
for orig, tokens in zip(docs, tokenized_docs):
    print(f"\nOriginal: {orig}")
    print(f"Tokens: {tokens}")

# Build vocabulary (exclude special tokens and subword markers)
vocab = set()
for tokens in tokenized_docs:
    clean_tokens = [t.replace('##', '') for t in tokens 
                   if not t.startswith('[')]
    vocab.update(clean_tokens)

vocab = sorted(vocab)
print(f"\n\nVocabulary size: {len(vocab)}")
print(f"Vocabulary: {vocab}")

# Create count vectors
word_to_idx = {word: idx for idx, word in enumerate(vocab)}

count_matrix = []
for tokens in tokenized_docs:
    # Clean tokens
    clean_tokens = [t.replace('##', '') for t in tokens 
                   if not t.startswith('[')]
    
    # Count frequencies
    token_counts = Counter(clean_tokens)
    
    # Create vector
    vector = np.zeros(len(vocab))
    for token, count in token_counts.items():
        if token in word_to_idx:
            vector[word_to_idx[token]] = count
    
    count_matrix.append(vector)

count_matrix = np.array(count_matrix)

print(f"\n\nCount Matrix Shape: {count_matrix.shape}")
print(f"Matrix:\n{count_matrix}")

# Show most frequent tokens across all documents
all_counts = Counter()
for tokens in tokenized_docs:
    clean = [t.replace('##', '') for t in tokens if not t.startswith('[')]
    all_counts.update(clean)

print("\n\nMost common tokens:")
for token, count in all_counts.most_common(5):
    print(f"  {token}: {count}")
```

---

## 9. BERT EMBEDDINGS (Contextual Embeddings)

**Kya Hai:** Context-aware embeddings - same word different contexts mein different vectors. State-of-the-art representation for modern NLP.

**Kab Use Karein:** Modern NLP tasks - sentiment analysis, NER, question answering. Jab highest accuracy chahiye aur computational resources available hain.

**Scenario:** "bank" word sentence context ke according different meaning (river bank vs financial bank) ke liye different embeddings.

### Example 1: Simple BERT Embeddings Extraction (NLTK tokenization)
```python
from transformers import BertTokenizer, BertModel
import torch
from nltk.tokenize import sent_tokenize

# Load BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Sample text
text = "Machine learning is transforming the world. Natural language processing is a subset of machine learning."

# Split into sentences
sentences = sent_tokenize(text)

print("Extracting BERT embeddings for sentences:")

# Get embeddings for each sentence
for sent in sentences:
    print(f"\nSentence: {sent}")
    
    # Tokenize and encode
    inputs = tokenizer(sent, return_tensors='pt', padding=True, truncation=True)
    
    # Get embeddings (no gradient calculation needed)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Last hidden state contains token embeddings
    last_hidden_state = outputs.last_hidden_state  # Shape: (batch, seq_len, hidden_size)
    
    print(f"  Embedding shape: {last_hidden_state.shape}")
    print(f"  Hidden size: {last_hidden_state.shape[-1]}")
    
    # Get sentence embedding (mean of token embeddings)
    sentence_embedding = last_hidden_state.mean(dim=1)
    print(f"  Sentence embedding shape: {sentence_embedding.shape}")
    print(f"  First 5 values: {sentence_embedding[0][:5]}")
```

### Example 2: spaCy + Transformers for Document Embeddings
```python
import spacy
from transformers import AutoTokenizer, AutoModel
import torch

# Load spaCy for preprocessing
nlp = spacy.load('en_core_web_sm')

# Load BERT
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Documents
docs = [
    "Artificial intelligence is revolutionizing healthcare",
    "Machine learning models can predict diseases",
    "I love eating pizza and pasta"
]

# Preprocess with spaCy (optional cleaning)
def clean_text(text):
    doc = nlp(text.lower())
    # Remove unwanted tokens if needed
    return text.lower()  # Simple lowercasing for BERT

# Get BERT embeddings
def get_bert_embedding(text):
    # Tokenize
    inputs = tokenizer(text, return_tensors='pt', padding=True, 
                      truncation=True, max_length=512)
    
    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use [CLS] token embedding as sentence representation
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    
    # Or use mean pooling
    # mean_embedding = outputs.last_hidden_state.mean(dim=1)
    
    return cls_embedding

# Get embeddings for all documents
print("BERT Embeddings for Documents:")
embeddings = []
for doc in docs:
    cleaned = clean_text(doc)
    embedding = get_bert_embedding(cleaned)
    embeddings.append(embedding)
    
    print(f"\nDocument: {doc}")
    print(f"Embedding shape: {embedding.shape}")

# Calculate similarity between documents
from torch.nn.functional import cosine_similarity

print("\n\nDocument Similarities (cosine):")
for i in range(len(docs)):
    for j in range(i+1, len(docs)):
        sim = cosine_similarity(embeddings[i], embeddings[j])
        print(f"\nDoc{i+1} vs Doc{j+1}: {sim.item():.4f}")
        print(f"  Doc{i+1}: {docs[i]}")
        print(f"  Doc{j+1}: {docs[j]}")
```

### Example 3: Contextual Word Embeddings with BERT
```python
from transformers import AutoTokenizer, AutoModel
import torch

# Load BERT
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Two sentences with same word in different contexts
sentences = [
    "I went to the bank to deposit money",  # bank = financial institution
    "I sat by the river bank to relax"      # bank = river edge
]

print("Demonstrating Contextual Embeddings:")
print("Word 'bank' in different contexts\n")

# Get embeddings for word "bank" in each context
bank_embeddings = []

for sent in sentences:
    print(f"Sentence: {sent}")
    
    # Tokenize
    inputs = tokenizer(sent, return_tensors='pt')
    tokens = tokenizer.tokenize(sent)
    
    print(f"Tokens: {tokens}")
    
    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Find position of "bank" token
    try:
        bank_idx = tokens.index('bank')
        # +1 because of [CLS] token at position 0
        bank_embedding = outputs.last_hidden_state[0, bank_idx + 1, :]
        bank_embeddings.append(bank_embedding)
        
        print(f"'bank' token at position: {bank_idx}")
        print(f"Embedding shape: {bank_embedding.shape}")
        print(f"First 5 values: {bank_embedding[:5]}\n")
    except ValueError:
        print("'bank' not found in tokens\n")

# Compare embeddings of "bank" in different contexts
if len(bank_embeddings) == 2:
    from torch.nn.functional import cosine_similarity
    
    sim = cosine_similarity(
        bank_embeddings[0].unsqueeze(0),
        bank_embeddings[1].unsqueeze(0)
    )
    
    print(f"\nSimilarity between 'bank' in two contexts: {sim.item():.4f}")
    print("(Lower similarity shows contextual awareness)")
```

### Example 4: spaCy-Transformers Pipeline
```python
# Note: Requires spacy-transformers
# pip install spacy-transformers
# python -m spacy download en_core_web_trf

import spacy

# Load transformer model
nlp = spacy.load('en_core_web_trf')  # Transformer-based model

# Documents
texts = [
    "Apple Inc. is a technology company based in California",
    "I ate an apple for breakfast today",
    "The apple tree in my garden is blooming"
]

print("Processing with spaCy Transformers:\n")

# Process texts
docs = [nlp(text) for text in texts]

# Extract embeddings and analyze
for i, (text, doc) in enumerate(zip(texts, docs)):
    print(f"Text {i+1}: {text}")
    
    # Document vector (from transformer)
    print(f"  Document vector shape: {doc.vector.shape}")
    print(f"  First 5 dims: {doc.vector[:5]}")
    
    # Token-level embeddings
    for token in doc:
        if token.text.lower() == 'apple':
            print(f"\n  Token 'apple':")
            print(f"    Vector shape: {token.vector.shape}")
            print(f"    First 5 dims: {token.vector[:5]}")
    print()

# Compare "apple" embeddings across contexts
apple_vectors = []
for doc in docs:
    for token in doc:
        if token.text.lower() == 'apple':
            apple_vectors.append(token.vector)
            break

# Calculate similarities
print("\nContextual 'apple' similarities:")
import numpy as np

def cosine_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

for i in range(len(apple_vectors)):
    for j in range(i+1, len(apple_vectors)):
        sim = cosine_sim(apple_vectors[i], apple_vectors[j])
        print(f"  Context {i+1} vs Context {j+1}: {sim:.4f}")
```

### Example 5: Fine-tuned BERT for Specific Tasks
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import torch

# Load pre-trained sentiment model (fine-tuned BERT)
model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Create sentiment analysis pipeline
sentiment_pipeline = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

# Texts to analyze
texts = [
    "This product is absolutely amazing!",
    "I'm very disappointed with the quality",
    "It's okay, nothing special"
]

print("Sentiment Analysis using Fine-tuned BERT:\n")

# Analyze sentiments
for text in texts:
    # Get prediction
    result = sentiment_pipeline(text)[0]
    
    print(f"Text: {text}")
    print(f"  Sentiment: {result['label']}")
    print(f"  Confidence: {result['score']:.4f}")
    
    # Also get embeddings
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model.distilbert(**inputs)  # Access base model
    
    # Get [CLS] token embedding
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    print(f"  Embedding shape: {cls_embedding.shape}")
    print()

# Demonstrate that embeddings are task-specific after fine-tuning
print("\nNote: These embeddings are optimized for sentiment classification!")
```

---

## 10. ELMo (Embeddings from Language Models)

**Kya Hai:** Deep contextualized word representations using bidirectional LSTM. BERT se pehle ka state-of-the-art contextual embedding method.

**Kab Use Karein:** Jab BERT se lightweight solution chahiye ya legacy systems mein. Research purposes ya specific use cases.

**Scenario:** Resource-constrained environments jahan BERT heavy hai but contextual embeddings chahiye.

### Example 1: TensorFlow Hub ELMo (Basic Usage)
```python
# Note: Requires tensorflow and tensorflow-hub
# pip install tensorflow tensorflow-hub

import tensorflow_hub as hub
import tensorflow as tf
import numpy as np

# Load ELMo model from TensorFlow Hub
print("Loading ELMo model (this may take a moment)...")
elmo = hub.load("https://tfhub.dev/google/elmo/3")

# Sample sentences
sentences = [
    "The bank is on the river bank",
    "I need to bank some money",
    "Machine learning is fascinating"
]

print("\nGenerating ELMo embeddings:\n")

# Get embeddings
for sent in sentences:
    # ELMo expects list of strings
    embeddings = elmo.signatures['default']([sent])
    
    # ELMo returns multiple layers
    # We'll use the default output
    sentence_embedding = embeddings['default'][0]
    
    print(f"Sentence: {sent}")
    print(f"  Embedding shape: {sentence_embedding.shape}")
    print(f"  Mean value: {tf.reduce_mean(sentence_embedding).numpy():.4f}")
    print()

# Compare "bank" in different contexts
print("Demonstrating contextual awareness:")
sent1 = "I went to the bank"
sent2 = "The river bank was beautiful"

emb1 = elmo.signatures['default']([sent1])['default']
emb2 = elmo.signatures['default']([sent2])['default']

# Calculate similarity (simplified)
sim = tf.reduce_sum(emb1 * emb2) / (tf.norm(emb1) * tf.norm(emb2))
print(f"\nSimilarity between sentences: {sim.numpy():.4f}")
```

### Example 2: allennlp ELMo Integration
```python
# Note: Requires allennlp
# pip install allennlp

from allennlp.modules.elmo import Elmo, batch_to_ids
import torch

# ELMo configuration
options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

# Initialize ELMo
print("Loading ELMo model...")
elmo = Elmo(options_file, weight_file, num_output_representations=1, dropout=0)

# Sample sentences (tokenized)
sentences = [
    ['I', 'love', 'machine', 'learning'],
    ['Natural', 'language', 'processing', 'is', 'interesting'],
]

print("\nProcessing sentences with ELMo:")

# Convert to character ids
character_ids = batch_to_ids(sentences)

# Get embeddings
with torch.no_grad():
    embeddings = elmo(character_ids)

# Extract embeddings
elmo_embeddings = embeddings['elmo_representations'][0]

print(f"\nEmbedding shape: {elmo_embeddings.shape}")
print(f"  (batch_size, seq_length, embedding_dim)")

# Analyze each sentence
for i, sent in enumerate(sentences):
    print(f"\nSentence {i+1}: {' '.join(sent)}")
    print(f"  Embedding shape: {elmo_embeddings[i].shape}")
    print(f"  Mean embedding: {elmo_embeddings[i].mean().item():.4f}")
```

### Example 3: spaCy with ELMo (Third-party Extension)
```python
# Conceptual example - requires spacy-elmo extension
# pip install spacy-elmo

print("ELMo with spaCy (conceptual example)")
print("Install: pip install spacy-elmo\n")

# In practice:
"""
import spacy
from spacy_elmo import load_elmo

nlp = spacy.load('en_core_web_sm')
elmo = load_elmo('elmo_model')
nlp.add_pipe(elmo)

doc = nlp("Machine learning is powerful")

# Access ELMo embeddings
for token in doc:
    elmo_embedding = token._.elmo_embedding
    print(f"{token.text}: {elmo_embedding.shape}")
"""

# Alternative: Manual integration
import spacy
import numpy as np

nlp = spacy.load('en_core_web_sm')

# Simulate ELMo processing
texts = [
    "The bank manager approved the loan",
    "We sat by the river bank"
]

print("Processing with spaCy (simulated ELMo):\n")

for text in texts:
    doc = nlp(text)
    print(f"Text: {text}")
    print(f"Tokens: {[token.text for token in doc]}")
    
    # In real scenario, you'd get ELMo embeddings here
    # For now, showing structure
    print(f"Would generate {len(doc)} contextual embeddings")
    print()
```

### Example 4: Comparing ELMo vs Static Embeddings
```python
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np

# Load ELMo
print("Loading models...")
elmo = hub.load("https://tfhub.dev/google/elmo/3")

# Sentences with polysemous word "light"
sentences = [
    "Turn on the light please",       # light = illumination
    "This bag is very light",         # light = not heavy
    "The morning light was beautiful"  # light = daylight
]

print("\nAnalyzing word 'light' in different contexts:\n")

# Get ELMo embeddings
light_embeddings = []

for sent in sentences:
    print(f"Sentence: {sent}")
    
    # Get ELMo embedding
    emb = elmo.signatures['default']([sent])['default'][0]
    
    # For simplicity, using full sentence embedding
    # In practice, you'd extract specific word embedding
    light_embeddings.append(emb.numpy())
    
    print(f"  Embedding shape: {emb.shape}")

# Calculate similarities
print("\n\nContextual similarities:")
for i in range(len(sentences)):
    for j in range(i+1, len(sentences)):
        # Cosine similarity
        sim = np.dot(light_embeddings[i].flatten(), 
                    light_embeddings[j].flatten())
        sim /= (np.linalg.norm(light_embeddings[i]) * 
               np.linalg.norm(light_embeddings[j]))
        
        print(f"Sentence {i+1} vs {j+1}: {sim:.4f}")

print("\nNote: Different contexts produce different embeddings!")
print("This is ELMo's contextual awareness in action.")
```

### Example 5: Hugging Face Alternative (Using RoBERTa as Modern Replacement)
```python
from transformers import AutoTokenizer, AutoModel
import torch

# Modern replacement for ELMo: RoBERTa
# (BERT family, better than ELMo)
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
model = AutoModel.from_pretrained('roberta-base')

print("Using RoBERTa (modern ELMo alternative):\n")

# Same polysemy example
sentences = [
    "Turn on the light please",
    "This bag is very light",
    "The morning light was beautiful"
]

# Get contextual embeddings
def get_word_embedding(sentence, target_word):
    """Get embedding for specific word in context"""
    # Tokenize
    inputs = tokenizer(sentence, return_tensors='pt')
    tokens = tokenizer.tokenize(sentence)
    
    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Find target word position
    try:
        # Note: RoBERTa uses different tokenization
        word_idx = [i for i, t in enumerate(tokens) 
                   if target_word in t.lower()][0]
        
        # +1 for special token
        embedding = outputs.last_hidden_state[0, word_idx + 1, :]
        return embedding
    except:
        return None

# Extract "light" embeddings
print("Extracting contextual embeddings for 'light':\n")

light_embeddings = []
for sent in sentences:
    emb = get_word_embedding(sent, 'light')
    if emb is not None:
        light_embeddings.append(emb)
        print(f"Sentence: {sent}")
        print(f"  Embedding shape: {emb.shape}")
        print(f"  First 5 values: {emb[:5]}\n")

# Compare embeddings
if len(light_embeddings) >= 2:
    from torch.nn.functional import cosine_similarity
    
    print("\nContextual similarities:")
    for i in range(len(light_embeddings)):
        for j in range(i+1, len(light_embeddings)):
            sim = cosine_similarity(
                light_embeddings[i].unsqueeze(0),
                light_embeddings[j].unsqueeze(0)
            )
            print(f"  Context {i+1} vs {j+1}: {sim.item():.4f}")
```

---

## **SUMMARY: When to Use Which Technique**

### **Traditional Methods (Frequency-based):**
1. **Bag of Words / Count Vectorization** → Simple classification, small datasets
2. **TF-IDF** → Document similarity, search, IR systems
3. **N-grams** → Phrase detection, sentiment with negation

### **Dense Embeddings (Semantic):**
4. **Word2Vec (CBOW/Skip-gram)** → Word similarity, analogies, small to medium data
5. **GloVe** → Similar to Word2Vec, pre-trained vectors available
6. **FastText** → Handles OOV words, morphologically rich languages

### **Simple Encoding:**
7. **One-Hot Encoding** → Small vocabulary, categorical data
8. **Count Vectorization** → Production-ready BoW, sklearn integration

### **Modern Deep Learning:**
9. **BERT Embeddings** → Best accuracy, contextual understanding, modern NLP
10. **ELMo** → Legacy contextual embeddings (use BERT instead)

### **Industry Recommendation:**
- **Small Projects:** TF-IDF + sklearn
- **Medium Projects:** FastText or Word2Vec
- **Production/Modern:** BERT/RoBERTa embeddings
- **Resource-Constrained:** TF-IDF or GloVe

**Key Point:** Modern industry mostly uses **BERT/Transformer embeddings** for high accuracy. Traditional methods still useful for baseline models and resource-constrained scenarios! 🚀    
    