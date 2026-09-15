# Text Preprocessing - Complete Guide

## Text Preprocessing Kyu Jaruri Hai?

**3 Main Reasons:**
1. **Data Quality**: Raw text mein noise hota hai (special chars, URLs, etc) jo model ko confuse karta hai
2. **Consistency**: Same meaning wale words ko same form mein laana (running = run)
3. **Performance**: Clean data se model accuracy badhti hai aur training fast hoti hai

---

## 1. TOKENIZATION

**Kya Hai:** Text ko smaller units (words, sentences) mein todna. Ye sabse pehli step hai preprocessing mein.

**Kab Use Karein:** Har NLP task mein - sentiment analysis, translation, chatbots sab mein sabse pehle tokenization zaruri hai.

**Scenario:** Jab aapko text ko words ya sentences mein break karna ho taaki aage processing kar sakein.

### Example 1: NLTK - Word Tokenization
```python
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

# Raw text input
text = "Hello! How are you doing today?"

# Tokenize text into words
tokens = word_tokenize(text)
print(tokens)
# Output: ['Hello', '!', 'How', 'are', 'you', 'doing', 'today', '?']
```

### Example 2: NLTK - Sentence Tokenization
```python
from nltk.tokenize import sent_tokenize

# Multiple sentences wala text
text = "I love NLP. It is amazing! Let's learn more."

# Tokenize text into sentences
sentences = sent_tokenize(text)
print(sentences)
# Output: ['I love NLP.', 'It is amazing!', "Let's learn more."]
```

### Example 3: spaCy - Word Tokenization
```python
import spacy

# Load English language model
nlp = spacy.load('en_core_web_sm')

# Process the text
doc = nlp("Natural Language Processing is fun!")

# Extract tokens from doc
tokens = [token.text for token in doc]
print(tokens)
# Output: ['Natural', 'Language', 'Processing', 'is', 'fun', '!']
```

### Example 4: Python Split Method (Simple)
```python
# Simple text
text = "Machine learning is awesome"

# Split by spaces to get tokens
tokens = text.split()
print(tokens)
# Output: ['Machine', 'learning', 'is', 'awesome']
```

### Example 5: RegEx Tokenization
```python
import re

# Text with multiple spaces and punctuation
text = "Hey!!!  How's  it   going?"

# Use regex to tokenize (alphabetic characters only)
tokens = re.findall(r'\b\w+\b', text)
print(tokens)
# Output: ['Hey', 'How', 's', 'it', 'going']
```

---

## 2. LOWERCASING

**Kya Hai:** Saare characters ko lowercase mein convert karna. "Apple" aur "apple" ko same samajhne ke liye.

**Kab Use Karein:** Jab case sensitivity important na ho - search engines, sentiment analysis mein useful hai.

**Scenario:** Agar "Good" aur "good" ko alag words nahi samajhna chahte to lowercasing karo.

### Example 1: Python Lower Method
```python
# Text with mixed case
text = "Natural Language Processing Is AWESOME!"

# Convert entire text to lowercase
lowercase_text = text.lower()
print(lowercase_text)
# Output: natural language processing is awesome!
```

### Example 2: List of Tokens ko Lowercase
```python
# List of tokens with mixed case
tokens = ['Hello', 'WORLD', 'Python', 'NLP']

# Convert each token to lowercase
lowercase_tokens = [token.lower() for token in tokens]
print(lowercase_tokens)
# Output: ['hello', 'world', 'python', 'nlp']
```

### Example 3: NLTK Tokenize + Lowercase
```python
from nltk.tokenize import word_tokenize

# Raw text
text = "Machine Learning Is FUN!"

# Tokenize and convert to lowercase
tokens = word_tokenize(text.lower())
print(tokens)
# Output: ['machine', 'learning', 'is', 'fun', '!']
```

### Example 4: spaCy with Lowercase
```python
import spacy

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Process text
doc = nlp("Deep Learning Is POWERFUL")

# Get lowercase tokens
lowercase_tokens = [token.text.lower() for token in doc]
print(lowercase_tokens)
# Output: ['deep', 'learning', 'is', 'powerful']
```

### Example 5: Pandas DataFrame Column Lowercase
```python
import pandas as pd

# Create sample dataframe
df = pd.DataFrame({'text': ['Hello World', 'PYTHON NLP', 'Data Science']})

# Apply lowercase to entire column
df['lowercase_text'] = df['text'].str.lower()
print(df)
# Output: DataFrame with lowercase column
```

---

## 3. STOPWORDS REMOVAL

**Kya Hai:** Common words jo zyada meaning nahi add karte (is, am, are, the, a) unhe remove karna.

**Kab Use Karein:** Text classification, topic modeling mein. Lekin sentiment analysis mein careful rahe kyunki "not" important hai.

**Scenario:** Agar document similarity check kar rahe ho to stopwords hatane se useful words pe focus milta hai.

### Example 1: NLTK Stopwords
```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')

# Sample text
text = "This is a sample sentence showing off stopwords removal"

# Get English stopwords
stop_words = set(stopwords.words('english'))

# Tokenize text
tokens = word_tokenize(text.lower())

# Remove stopwords from tokens
filtered_tokens = [word for word in tokens if word not in stop_words]
print(filtered_tokens)
# Output: ['sample', 'sentence', 'showing', 'stopwords', 'removal']
```

### Example 2: spaCy Stopwords
```python
import spacy

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Process text
text = "I am learning natural language processing with Python"
doc = nlp(text)

# Filter out stopwords using spaCy's is_stop attribute
filtered_tokens = [token.text for token in doc if not token.is_stop]
print(filtered_tokens)
# Output: ['learning', 'natural', 'language', 'processing', 'Python']
```

### Example 3: Custom Stopwords List
```python
from nltk.tokenize import word_tokenize

# Your own custom stopwords
custom_stopwords = {'hello', 'hi', 'hey', 'thanks', 'ok'}

# Sample text
text = "Hi there, thanks for the help, ok bye"

# Tokenize
tokens = word_tokenize(text.lower())

# Remove custom stopwords
filtered = [word for word in tokens if word not in custom_stopwords]
print(filtered)
# Output: ['there', ',', 'for', 'help', ',', 'bye']
```

### Example 4: NLTK + Custom Stopwords Combined
```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Combine NLTK stopwords with your custom ones
stop_words = set(stopwords.words('english'))
stop_words.update(['also', 'however', 'therefore'])

# Text input
text = "However, machine learning is also very important therefore we study it"

# Tokenize and filter
tokens = word_tokenize(text.lower())
filtered = [w for w in tokens if w.isalpha() and w not in stop_words]
print(filtered)
# Output: ['machine', 'learning', 'important', 'study']
```

### Example 5: Multiple Languages Stopwords
```python
from nltk.corpus import stopwords

# Text in Hindi (example)
text_hindi = "यह एक उदाहरण है"

# Get Hindi stopwords
stop_words_hindi = set(stopwords.words('hindi'))

# Filter (for demonstration)
tokens = text_hindi.split()
filtered = [w for w in tokens if w not in stop_words_hindi]
print(f"Original: {tokens}")
print(f"Filtered: {filtered}")
```

---

## 4. PUNCTUATION REMOVAL

**Kya Hai:** Text se punctuation marks (!, ?, ., ,) remove karna kyunki ye generally meaning mein contribute nahi karte.

**Kab Use Karein:** Text classification, word frequency analysis mein. Lekin sentiment mein "!!!" intensity show karta hai to careful.

**Scenario:** Jab aap word count ya frequency analysis kar rahe ho tab punctuation unnecessary noise hai.

### Example 1: String Module
```python
import string

# Text with punctuation
text = "Hello! How are you? I'm doing great."

# Remove all punctuation using translate
text_no_punct = text.translate(str.maketrans('', '', string.punctuation))
print(text_no_punct)
# Output: Hello How are you Im doing great
```

### Example 2: RegEx Method
```python
import re

# Text with various punctuation marks
text = "NLP is amazing!!! Don't you think so???"

# Replace all punctuation with empty string using regex
clean_text = re.sub(r'[^\w\s]', '', text)
print(clean_text)
# Output: NLP is amazing Dont you think so
```

### Example 3: List Comprehension with isalpha()
```python
from nltk.tokenize import word_tokenize

# Text input
text = "Machine, learning! and AI? are #awesome."

# Tokenize first
tokens = word_tokenize(text)

# Keep only alphabetic tokens (removes punctuation)
alpha_tokens = [word for word in tokens if word.isalpha()]
print(alpha_tokens)
# Output: ['Machine', 'learning', 'and', 'AI', 'are', 'awesome']
```

### Example 4: spaCy with Punctuation Filter
```python
import spacy

# Load model
nlp = spacy.load('en_core_web_sm')

# Process text with punctuation
text = "Hello, world! This is NLP."
doc = nlp(text)

# Filter out punctuation using is_punct attribute
no_punct = [token.text for token in doc if not token.is_punct]
print(no_punct)
# Output: ['Hello', 'world', 'This', 'is', 'NLP']
```

### Example 5: Pandas Apply for DataFrame
```python
import pandas as pd
import string

# Sample dataframe
df = pd.DataFrame({'reviews': ['Great product!!!', 'Bad quality...', 'Nice, I liked it!']})

# Function to remove punctuation
def remove_punct(text):
    return text.translate(str.maketrans('', '', string.punctuation))

# Apply to column
df['clean_reviews'] = df['reviews'].apply(remove_punct)
print(df)
# Output: DataFrame with clean reviews
```

---

## 5. STEMMING

**Kya Hai:** Words ko unke root form mein convert karna by removing suffixes. Fast hai but sometimes incorrect results deta hai.

**Kab Use Karein:** Search engines, text mining jahan speed important ho aur accuracy thodi compromise kar sakte ho.

**Scenario:** "running", "runs", "ran" sab ko "run" mein convert karna ho (though ran -> ran rahega stemming mein).

### Example 1: Porter Stemmer (NLTK)
```python
from nltk.stem import PorterStemmer

# Initialize stemmer
ps = PorterStemmer()

# List of words to stem
words = ['running', 'runs', 'ran', 'runner', 'easily', 'fairly']

# Apply stemming to each word
stemmed = [ps.stem(word) for word in words]
print(stemmed)
# Output: ['run', 'run', 'ran', 'runner', 'easili', 'fairli']
```

### Example 2: Snowball Stemmer (NLTK)
```python
from nltk.stem import SnowballStemmer

# Initialize Snowball stemmer for English
snowball = SnowballStemmer('english')

# Words to stem
words = ['organization', 'organizational', 'organize', 'organizes']

# Stem each word
stemmed = [snowball.stem(word) for word in words]
print(stemmed)
# Output: ['organ', 'organ', 'organ', 'organ']
```

### Example 3: Lancaster Stemmer (Aggressive)
```python
from nltk.stem import LancasterStemmer

# Initialize Lancaster stemmer (most aggressive)
lancaster = LancasterStemmer()

# Sample words
words = ['maximum', 'multiply', 'provision', 'owed']

# Apply stemming
stemmed = [lancaster.stem(word) for word in words]
print(stemmed)
# Output: ['maxim', 'multiply', 'prov', 'ow']
```

### Example 4: Stemming Complete Sentence
```python
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Initialize
ps = PorterStemmer()

# Sentence
sentence = "The striped bats are hanging on their feet for best"

# Tokenize and stem
tokens = word_tokenize(sentence)
stemmed_sentence = [ps.stem(word) for word in tokens]
print(stemmed_sentence)
# Output: ['the', 'stripe', 'bat', 'are', 'hang', 'on', 'their', 'feet', 'for', 'best']
```

### Example 5: DataFrame Column Stemming
```python
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Initialize stemmer
ps = PorterStemmer()

# Sample dataframe
df = pd.DataFrame({'text': ['running quickly', 'organizations meeting', 'played games']})

# Function to stem text
def stem_text(text):
    tokens = word_tokenize(text)
    return ' '.join([ps.stem(token) for token in tokens])

# Apply stemming
df['stemmed'] = df['text'].apply(stem_text)
print(df)
# Output: DataFrame with stemmed text column
```

---

## 6. LEMMATIZATION

**Kya Hai:** Words ko unke base/dictionary form (lemma) mein convert karna. Stemming se accurate hai lekin slow.

**Kab Use Karein:** Jab accuracy important ho - chatbots, question answering, sentiment analysis.

**Scenario:** "better" ka lemma "good" hai, "ran" ka lemma "run" hai - context-aware processing.

### Example 1: NLTK WordNet Lemmatizer
```python
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Words to lemmatize
words = ['running', 'ran', 'runs', 'better', 'geese']

# Lemmatize each word (default POS is noun)
lemmatized = [lemmatizer.lemmatize(word, pos='v') for word in words]
print(lemmatized)
# Output: ['run', 'run', 'run', 'better', 'geese']
```

### Example 2: spaCy Lemmatization
```python
import spacy

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Process text
text = "The striped bats were hanging on their feet"
doc = nlp(text)

# Get lemma for each token
lemmas = [token.lemma_ for token in doc]
print(lemmas)
# Output: ['the', 'strip', 'bat', 'be', 'hang', 'on', 'their', 'foot']
```

### Example 3: With POS Tagging (NLTK)
```python
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize

# Initialize
lemmatizer = WordNetLemmatizer()

# Sentence
sentence = "The cats are running and playing"

# Tokenize
tokens = word_tokenize(sentence)

# Lemmatize with proper POS tags
lemmatized = [lemmatizer.lemmatize(token, pos='v') if pos_tag([token])[0][1].startswith('V') 
              else lemmatizer.lemmatize(token) for token in tokens]
print(lemmatized)
# Output: Properly lemmatized based on POS
```

### Example 4: Lemmatization vs Stemming Comparison
```python
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Initialize both
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Test word
word = "studies"

# Compare results
stemmed = ps.stem(word)
lemmatized = lemmatizer.lemmatize(word, pos='v')

print(f"Original: {word}")
print(f"Stemmed: {stemmed}")    # Output: studi
print(f"Lemmatized: {lemmatized}")  # Output: study
```

### Example 5: Batch Processing with spaCy
```python
import spacy

# Load model
nlp = spacy.load('en_core_web_sm')

# Multiple sentences
texts = ["I am running", "She was reading books", "They have eaten"]

# Process and lemmatize all
for text in texts:
    doc = nlp(text)
    lemmas = ' '.join([token.lemma_ for token in doc])
    print(f"Original: {text} -> Lemmatized: {lemmas}")
# Output: Shows original and lemmatized versions
```

---

## 7. REMOVING NUMBERS

**Kya Hai:** Text se sabhi numeric values (0-9) ko remove karna kyunki kaafi tasks mein numbers relevant nahi hote.

**Kab Use Karein:** Sentiment analysis, topic modeling mein. Lekin financial text ya scientific text mein numbers important ho sakte hain.

**Scenario:** Product reviews mein "Great product" useful hai but "Product123" mein 123 unnecessary hai.

### Example 1: RegEx to Remove Numbers
```python
import re

# Text with numbers
text = "I bought 5 apples and 10 oranges for $25"

# Remove all digits using regex
no_numbers = re.sub(r'\d+', '', text)
print(no_numbers)
# Output: I bought  apples and  oranges for $
```

### Example 2: isdigit() Method
```python
# Text with numbers
text = "Python3 is better than Python2"

# Remove characters that are digits
clean_text = ''.join([char for char in text if not char.isdigit()])
print(clean_text)
# Output: Python is better than Python
```

### Example 3: Remove Numbers but Keep Text
```python
from nltk.tokenize import word_tokenize

# Text
text = "AI 2024 and ML 2025 are trending"

# Tokenize
tokens = word_tokenize(text)

# Keep only non-numeric tokens
no_nums = [token for token in tokens if not token.isdigit()]
print(no_nums)
# Output: ['AI', 'and', 'ML', 'are', 'trending']
```

### Example 4: Remove Numbers with Letters
```python
import re

# Text with alphanumeric words
text = "Model123 performed better than Model456 in test789"

# Remove any word containing numbers
no_nums = re.sub(r'\w*\d\w*', '', text)
print(no_nums.strip())
# Output: performed better than  in
```

### Example 5: Pandas Column Processing
```python
import pandas as pd
import re

# Sample dataframe
df = pd.DataFrame({'text': ['Buy 2 get 1 free', 'Contact: 9876543210', 'Price: $99']})

# Function to remove numbers
def remove_numbers(text):
    return re.sub(r'\d+', '', text)

# Apply to column
df['clean_text'] = df['text'].apply(remove_numbers)
print(df)
# Output: DataFrame with numbers removed
```

---

## 8. REMOVING SPECIAL CHARACTERS

**Kya Hai:** Text se special symbols (@, #, $, %, &, *) remove karna jo normal words nahi hain.

**Kab Use Karein:** General text classification mein. Lekin social media analysis mein # (hashtags) important ho sakte hain.

**Scenario:** Email text ko clean karna where @, $ jaise symbols meaning nahi add karte (except email IDs).

### Example 1: RegEx Keep Only Alphanumeric
```python
import re

# Text with special characters
text = "Hello@World! #NLP is $awesome & *cool*"

# Keep only alphanumeric and spaces
clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
print(clean_text)
# Output: HelloWorld NLP is awesome  cool
```

### Example 2: Remove Specific Special Chars
```python
import string

# Text
text = "Email: test@example.com | Price: $100 #discount"

# Define special characters to remove
special_chars = string.punctuation

# Remove them
clean_text = ''.join([char for char in text if char not in special_chars])
print(clean_text)
# Output: Email testexamplecom  Price 100 discount
```

### Example 3: Keep Letters and Spaces Only
```python
# Text with various special characters
text = "Machine Learning & AI >> Future @2024"

# Keep only alphabets and spaces
clean_text = ''.join([char for char in text if char.isalpha() or char.isspace()])
print(clean_text)
# Output: Machine Learning  AI  Future
```

### Example 4: spaCy with Token Filter
```python
import spacy

# Load model
nlp = spacy.load('en_core_web_sm')

# Text with special characters
text = "Natural-Language-Processing is #awesome!!!"
doc = nlp(text)

# Keep only alphabetic tokens
clean_tokens = [token.text for token in doc if token.is_alpha]
print(clean_tokens)
# Output: ['Natural', 'Language', 'Processing', 'is', 'awesome']
```

### Example 5: Custom Character Removal
```python
# Text
text = "Python_3.9 >> Java_11 && C++_14"

# Define characters to remove
chars_to_remove = ['_', '>', '&', '+']

# Remove specified characters
clean_text = text
for char in chars_to_remove:
    clean_text = clean_text.replace(char, '')
    
print(clean_text)
# Output: Python3.9  Java11  C14
```

---

## 9. REMOVING EXTRA WHITESPACES

**Kya Hai:** Multiple spaces, tabs, newlines ko single space mein convert karna taaki text consistent rahe.

**Kab Use Karein:** Har preprocessing pipeline mein - kyunki extra spaces tokenization aur analysis mein problem create karte hain.

**Scenario:** Web scraping se mila data jisme irregular spacing hoti hai use clean karna.

### Example 1: Using split() and join()
```python
# Text with extra spaces
text = "This    is   a    text   with    irregular     spacing"

# Split and rejoin (removes extra spaces automatically)
clean_text = ' '.join(text.split())
print(clean_text)
# Output: This is a text with irregular spacing
```

### Example 2: RegEx Method
```python
import re

# Text with tabs, newlines, multiple spaces
text = "Hello\n\n\tWorld    This  is   NLP"

# Replace multiple whitespaces with single space
clean_text = re.sub(r'\s+', ' ', text)
print(clean_text)
# Output: Hello World This is NLP
```

### Example 3: Strip + Internal Spaces
```python
# Text with leading, trailing and internal spaces
text = "   Machine   Learning    is   awesome   "

# Remove leading/trailing and fix internal spaces
clean_text = ' '.join(text.split())
print(clean_text)
# Output: Machine Learning is awesome
```

### Example 4: Multiple Line Text
```python
# Multiline text with inconsistent spacing
text = """
    Natural    Language
    
    Processing     is
    
         amazing
"""

# Clean all extra whitespaces
clean_text = ' '.join(text.split())
print(clean_text)
# Output: Natural Language Processing is amazing
```

### Example 5: Pandas DataFrame Column
```python
import pandas as pd
import re

# Sample dataframe with messy spacing
df = pd.DataFrame({'text': ['Hello   World', 'Python    NLP', 'Data     Science   ']})

# Function to clean whitespace
def clean_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip()

# Apply to column
df['clean_text'] = df['text'].apply(clean_whitespace)
print(df)
# Output: DataFrame with cleaned spacing
```

---

## 10. REMOVING URLs

**Kya Hai:** Text se URLs (http://, https://, www.) remove karna kyunki ye typically analysis mein useful nahi hote.

**Kab Use Karein:** Social media text, web scraping, review analysis - jahan URLs noise hain aur content mein meaning nahi add karte.

**Scenario:** Twitter tweets ya Reddit posts se URLs hatana taaki actual text content pe focus kar sakein.

### Example 1: RegEx for HTTP/HTTPS URLs
```python
import re

# Text with URLs
text = "Check this out: https://example.com and http://test.com for more info"

# Remove URLs starting with http or https
clean_text = re.sub(r'http\S+|https\S+', '', text)
print(clean_text)
# Output: Check this out:  and  for more info
```

### Example 2: Remove All URLs (Including www)
```python
import re

# Text with various URL formats
text = "Visit www.example.com or https://test.com or http://demo.org"

# Remove all types of URLs
clean_text = re.sub(r'http\S+|www\.\S+', '', text)
print(clean_text)
# Output: Visit  or  or
```

### Example 3: Comprehensive URL Removal
```python
import re

# Text with different URL patterns
text = "Go to example.com or https://github.com/user/repo or www.test.co.in"

# More comprehensive pattern
clean_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
clean_text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', clean_text)
print(clean_text)
# Output: Go to example.com or  or
```

### Example 4: NLTK Tokenize and Filter
```python
from nltk.tokenize import word_tokenize
import re

# Text with URLs
text = "Read more at https://blog.example.com and share your thoughts"

# Remove URLs first
clean_text = re.sub(r'http\S+|www\.\S+', '', text)

# Then tokenize
tokens = word_tokenize(clean_text)
print(tokens)
# Output: ['Read', 'more', 'at', 'and', 'share', 'your', 'thoughts']
```

### Example 5: Pandas Column URL Removal
```python
import pandas as pd
import re

# Sample dataframe with social media posts
df = pd.DataFrame({
    'posts': [
        'Check https://example.com for details',
        'Visit www.test.com to learn',
        'No URL here just text'
    ]
})

# Function to remove URLs
def remove_urls(text):
    return re.sub(r'http\S+|www\.\S+', '', text).strip()

# Apply to column
df['clean_posts'] = df['posts'].apply(remove_urls)
print(df)
# Output: DataFrame without URLs
```

---

## 11. REMOVING HTML TAGS

**Kya Hai:** Web scraping se mila HTML/XML code se tags (<p>, <div>, <br>) remove karna taaki sirf text content rahe.

**Kab Use Karein:** Web scraping, email parsing, jahan HTML formatted text aata hai usse clean text extract karna.

**Scenario:** Website se reviews scrape kiye jisme HTML tags hain, unhe remove karke clean reviews nikalna.

### Example 1: RegEx HTML Tag Removal
```python
import re

# HTML text
html_text = "<p>This is a <b>bold</b> statement.</p>"

# Remove HTML tags using regex
clean_text = re.sub(r'<[^>]+>', '', html_text)
print(clean_text)
# Output: This is a bold statement.
```

### Example 2: BeautifulSoup Library
```python
from bs4 import BeautifulSoup

# HTML content
html = "<html><body><h1>Title</h1><p>This is a paragraph.</p></body></html>"

# Parse HTML and extract text
soup = BeautifulSoup(html, 'html.parser')
clean_text = soup.get_text()
print(clean_text)
# Output: Title This is a paragraph.
```

### Example 3: Complex HTML with Multiple Tags
```python
from bs4 import BeautifulSoup

# Complex HTML
html = """
<div class="content">
    <h2>Heading</h2>
    <p>First paragraph with <a href="#">link</a></p>
    <ul><li>Item 1</li><li>Item 2</li></ul>
</div>
"""

# Extract clean text
soup = BeautifulSoup(html, 'html.parser')
clean_text = soup.get_text(separator=' ', strip=True)
print(clean_text)
# Output: Heading First paragraph with link Item 1 Item 2
```

### Example 4: Remove HTML Entities
```python
import html

# Text with HTML entities
text = "5 &gt; 3 and 2 &lt; 4, &quot;quotes&quot; &amp; &apos;apostrophe&apos;"

# Unescape HTML entities
clean_text = html.unescape(text)
print(clean_text)
# Output: 5 > 3 and 2 < 4, "quotes" & 'apostrophe'
```

### Example 5: Pandas with HTML Removal
```python
import pandas as pd
from bs4 import BeautifulSoup

# Dataframe with HTML content
df = pd.DataFrame({
    'html_text': [
        '<p>Product is <strong>great</strong></p>',
        '<div>Very <em>good</em> quality</div>',
        'No HTML here'
    ]
})

# Function to remove HTML
def remove_html(text):
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text()

# Apply function
df['clean_text'] = df['html_text'].apply(remove_html)
print(df)
# Output: DataFrame with clean text
```

---

## 12. REMOVING EMOJIS

**Kya Hai:** Text se emojis (😊, 👍, ❤️) remove karna. Social media analysis mein kabhi zaruri hota hai kabhi nahi.

**Kab Use Karein:** Traditional NLP models ke liye jo emojis handle nahi kar sakte. Lekin sentiment analysis mein emojis useful ho sakte hain.

**Scenario:** Formal document analysis ya traditional ML models ke liye emoji-free text chahiye.

### Example 1: Using emoji Library
```python
import emoji

# Text with emojis
text = "I love Python 😍 and NLP is amazing 🚀👍"

# Remove all emojis
clean_text = emoji.replace_emoji(text, '')
print(clean_text)
# Output: I love Python  and NLP is amazing
```

### Example 2: RegEx Pattern for Emojis
```python
import re

# Text with emojis
text = "Great product! 😊👍 Highly recommended ❤️"

# Emoji regex pattern (covers wide range)
emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        "]+", flags=re.UNICODE)

# Remove emojis
clean_text = emoji_pattern.sub(r'', text)
print(clean_text)
# Output: Great product!  Highly recommended
```

### Example 3: demoji Library
```python
# pip install demoji first
import demoji

# Download emoji data (run once)
# demoji.download_codes()

# Text with emojis
text = "Learning NLP 📚 is fun 🎉🎊"

# Remove all emojis
clean_text = demoji.replace(text, '')
print(clean_text)
# Output: Learning NLP  is fun
```

### Example 4: Keep Text, Remove Emojis and Extra Spaces
```python
import re

# Text with emojis
text = "Python 🐍 programming 💻 is awesome 🌟"

# Remove emojis
emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        "]+", flags=re.UNICODE)

clean_text = emoji_pattern.sub(r'', text)

# Also clean extra spaces
clean_text = ' '.join(clean_text.split())
print(clean_text)
# Output: Python programming is awesome
```

### Example 5: Pandas DataFrame Emoji Removal
```python
import pandas as pd
import re

# Sample dataframe with social media posts
df = pd.DataFrame({
    'tweets': [
        'Good morning! ☀️',
        'I love coding 💻❤️',
        'Happy weekend 🎉🎊🥳'
    ]
})

# Emoji pattern
emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        "]+", flags=re.UNICODE)

# Remove emojis from column
df['clean_tweets'] = df['tweets'].apply(lambda x: emoji_pattern.sub(r'', x))
print(df)
# Output: DataFrame without emojis
```

---

## 13. SPELL CORRECTION

**Kya Hai:** Misspelled words ko sahi spelling mein convert karna. "helo" -> "hello", "progrmming" -> "programming".

**Kab Use Karein:** User-generated content (reviews, social media) jahan spelling mistakes common hain. Search queries ko correct karna.

**Scenario:** Customer feedback forms ya chat messages mein typos automatically correct karna.

### Example 1: TextBlob Library
```python
from textblob import TextBlob

# Text with spelling mistakes
text = "I havv goood speling"

# Create TextBlob object and correct
blob = TextBlob(text)
corrected = blob.correct()
print(corrected)
# Output: I have good spelling
```

### Example 2: pyspellchecker Library
```python
from spellchecker import SpellChecker

# Initialize spell checker
spell = SpellChecker()

# Text with mistakes
words = "machne lerning is amzing".split()

# Find misspelled words and correct them
corrected_words = []
for word in words:
    corrected_words.append(spell.correction(word))
    
corrected_text = ' '.join(corrected_words)
print(corrected_text)
# Output: machine learning is amazing
```

### Example 3: autocorrect Library
```python
from autocorrect import Speller

# Initialize speller
spell = Speller()

# Text with typos
text = "Natral languag processng"

# Correct text
corrected = spell(text)
print(corrected)
# Output: Natural language processing
```

### Example 4: Custom Dictionary with pyspellchecker
```python
from spellchecker import SpellChecker

# Initialize with custom words
spell = SpellChecker()

# Add custom technical words to dictionary
spell.word_frequency.load_words(['tensorflow', 'pytorch', 'sklearn'])

# Text with technical terms
text = "I use tensorflw and pytoch for deep lerning"

# Correct each word
words = text.split()
corrected = [spell.correction(word) for word in words]
print(' '.join(corrected))
# Output: I use tensorflow and pytorch for deep learning
```

### Example 5: Batch Correction with Pandas
```python
import pandas as pd
from textblob import TextBlob

# Sample dataframe with typos
df = pd.DataFrame({
    'reviews': [
        'Tis prodct is gud',
        'Vry bad qualiti',
        'Excelent servce'
    ]
})

# Function to correct spelling
def correct_spelling(text):
    return str(TextBlob(text).correct())

# Apply correction
df['corrected_reviews'] = df['reviews'].apply(correct_spelling)
print(df)
# Output: DataFrame with corrected spelling
```

---

## 14. EXPANDING CONTRACTIONS

**Kya Hai:** Shortened forms ko full form mein expand karna. "don't" -> "do not", "I'm" -> "I am".

**Kab Use Karein:** Formal analysis, sentiment analysis jahan full forms clarity dete hain. Search indexing mein bhi useful.

**Scenario:** Social media posts ya informal text ko formal text mein convert karna for better analysis.

### Example 1: contractions Library
```python
import contractions

# Text with contractions
text = "I'm learning NLP and it's awesome! I can't wait to learn more."

# Expand all contractions
expanded = contractions.fix(text)
print(expanded)
# Output: I am learning NLP and it is awesome! I cannot wait to learn more.
```

### Example 2: Custom Dictionary Method
```python
# Define contraction mapping
contraction_map = {
    "don't": "do not",
    "can't": "cannot",
    "won't": "will not",
    "I'm": "I am",
    "you're": "you are",
    "it's": "it is"
}

# Text with contractions
text = "I'm sure you're right. Don't worry, it's fine."

# Replace contractions
expanded_text = text
for contraction, expansion in contraction_map.items():
    expanded_text = expanded_text.replace(contraction, expansion)
    
print(expanded_text)
# Output: I am sure you are right. do not worry, it is fine.
```

### Example 3: RegEx with Replacement Dictionary
```python
import re

# Comprehensive contraction mapping
contractions_dict = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "i'd": "I would",
    "i'll": "I will",
    "i'm": "I am",
    "i've": "I have",
    "isn't": "is not",
    "it's": "it is",
    "let's": "let us",
    "shouldn't": "should not",
    "that's": "that is",
    "wasn't": "was not",
    "we're": "we are",
    "weren't": "were not",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are"
}

# Text
text = "I can't believe he's done that. We shouldn't worry."

# Expand contractions (case-insensitive)
def expand_contractions(text, contraction_dict):
    pattern = re.compile('({})'.format('|'.join(contraction_dict.keys())), 
                        flags=re.IGNORECASE|re.DOTALL)
    def replace(match):
        return contraction_dict[match.group(0).lower()]
    return pattern.sub(replace, text)

expanded = expand_contractions(text, contractions_dict)
print(expanded)
# Output: I cannot believe he is done that. We should not worry.
```

### Example 4: TextBlob with Contractions
```python
import contractions
from textblob import TextBlob

# Text with contractions
text = "It's a great day! I'll be there soon."

# First expand contractions
expanded = contractions.fix(text)

# Then can apply other processing
blob = TextBlob(expanded)
print(expanded)
# Output: It is a great day! I will be there soon.
```

### Example 5: Pandas Column Expansion
```python
import pandas as pd
import contractions

# Sample dataframe
df = pd.DataFrame({
    'comments': [
        "I'm loving this!",
        "Can't wait for more",
        "You're absolutely right"
    ]
})

# Function to expand contractions
def expand_text(text):
    return contractions.fix(text)

# Apply to column
df['expanded_comments'] = df['comments'].apply(expand_text)
print(df)
# Output: DataFrame with expanded contractions
```

---

## 15. REMOVING ACCENTS

**Kya Hai:** Accented characters ko normal characters mein convert karna. "café" -> "cafe", "naïve" -> "naive".

**Kab Use Karein:** English text processing jahan accents standardization chahiye. Search systems mein useful.

**Scenario:** International names/words ko English-compatible format mein convert karna.

### Example 1: unidecode Library
```python
from unidecode import unidecode

# Text with accents
text = "café résumé naïve Zürich"

# Remove accents
clean_text = unidecode(text)
print(clean_text)
# Output: cafe resume naive Zurich
```

### Example 2: unicodedata Library
```python
import unicodedata

# Text with accents
text = "José García in Montréal"

# Normalize and remove accents
nfd = unicodedata.normalize('NFD', text)
clean_text = ''.join(char for char in nfd if unicodedata.category(char) != 'Mn')
print(clean_text)
# Output: Jose Garcia in Montreal
```

### Example 3: Multiple Languages
```python
from unidecode import unidecode

# Text with various language accents
text = "Björk from Reykjavík, François in Zürich, José in São Paulo"

# Remove all accents
clean_text = unidecode(text)
print(clean_text)
# Output: Bjork from Reykjavik, Francois in Zurich, Jose in Sao Paulo
```

### Example 4: Preserve Some, Remove Others
```python
import unicodedata

# Function to remove specific accent types
def remove_accents(text, preserve_chars=''):
    nfd = unicodedata.normalize('NFD', text)
    output = []
    for char in nfd:
        if unicodedata.category(char) != 'Mn' or char in preserve_chars:
            output.append(char)
    return ''.join(output)

# Text
text = "résumé Zürich naïve"

# Remove accents
clean = remove_accents(text)
print(clean)
# Output: resume Zurich naive
```

### Example 5: Pandas DataFrame Processing
```python
import pandas as pd
from unidecode import unidecode

# Sample dataframe with international names
df = pd.DataFrame({
    'names': ['José', 'François', 'Müller', 'Søren', 'Chloé']
})

# Remove accents from column
df['clean_names'] = df['names'].apply(unidecode)
print(df)
# Output: DataFrame with accent-free names
```

---

## 16. PART-OF-SPEECH TAGGING

**Kya Hai:** Har word ko uska grammatical category assign karna (noun, verb, adjective etc). Analysis ke liye useful.

**Kab Use Karein:** Advanced NLP tasks - information extraction, dependency parsing, feature engineering.

**Scenario:** Sirf nouns extract karne hain document se, ya verbs identify karni hain action detection ke liye.

### Example 1: NLTK POS Tagging
```python
import nltk
from nltk import word_tokenize, pos_tag
nltk.download('averaged_perceptron_tagger')

# Sentence
sentence = "The quick brown fox jumps over the lazy dog"

# Tokenize and tag
tokens = word_tokenize(sentence)
pos_tags = pos_tag(tokens)
print(pos_tags)
# Output: [('The', 'DT'), ('quick', 'JJ'), ('brown', 'JJ'), ...]
```

### Example 2: spaCy POS Tagging
```python
import spacy

# Load model
nlp = spacy.load('en_core_web_sm')

# Process text
text = "Natural Language Processing is fascinating"
doc = nlp(text)

# Extract POS tags
for token in doc:
    print(f"{token.text}: {token.pos_} ({token.tag_})")
# Output: Natural: ADJ (JJ), Language: PROPN (NNP), ...
```

### Example 3: Extract Only Nouns
```python
import nltk
from nltk import word_tokenize, pos_tag

# Text
text = "Machine learning algorithms process large datasets efficiently"

# Tokenize and tag
tokens = word_tokenize(text)
pos_tags = pos_tag(tokens)

# Extract only nouns (NN, NNS, NNP, NNPS)
nouns = [word for word, pos in pos_tags if pos.startswith('NN')]
print(nouns)
# Output: ['Machine', 'learning', 'algorithms', 'datasets']
```

### Example 4: Extract Verbs Using spaCy
```python
import spacy

# Load model
nlp = spacy.load('en_core_web_sm')

# Text
text = "The student reads books and writes essays regularly"
doc = nlp(text)

# Extract only verbs
verbs = [token.text for token in doc if token.pos_ == 'VERB']
print(verbs)
# Output: ['reads', 'writes']
```

### Example 5: POS Tag Distribution Analysis
```python
import nltk
from nltk import word_tokenize, pos_tag
from collections import Counter

# Long text
text = "Python is a popular programming language. It is used for web development, data science, and machine learning."

# Tokenize and tag
tokens = word_tokenize(text)
pos_tags = pos_tag(tokens)

# Count POS tag frequencies
pos_counts = Counter([pos for word, pos in pos_tags])
print(pos_counts)
# Output: Counter showing frequency of each POS tag
```

---

## 17. NAMED ENTITY RECOGNITION (NER)

**Kya Hai:** Text mein important entities identify karna - person names, organizations, locations, dates etc.

**Kab Use Karein:** Information extraction, document summarization, question answering systems.

**Scenario:** News articles se automatically person names aur locations extract karna, ya resume se company names nikalna.

### Example 1: spaCy NER
```python
import spacy

# Load model
nlp = spacy.load('en_core_web_sm')

# Text with entities
text = "Apple Inc. was founded by Steve Jobs in California in 1976"
doc = nlp(text)

# Extract named entities
for ent in doc.ents:
    print(f"{ent.text}: {ent.label_}")
# Output: Apple Inc.: ORG, Steve Jobs: PERSON, California: GPE, 1976: DATE
```

### Example 2: NLTK NER
```python
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Text
text = "Barack Obama was born in Hawaii"

# Tokenize, POS tag, and NER
tokens = word_tokenize(text)
pos_tags = pos_tag(tokens)
named_entities = ne_chunk(pos_tags)

# Print tree structure
print(named_entities)
# Output: Tree structure with entities marked
```

### Example 3: Extract Specific Entity Types
```python
import spacy

# Load model
nlp = spacy.load('en_core_web_sm')

# Text
text = "Microsoft and Google are competing in AI. Bill Gates and Sundar Pichai are key figures."
doc = nlp(text)

# Extract only PERSON entities
persons = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
print(f"Persons: {persons}")

# Extract only ORG entities
orgs = [ent.text for ent in doc.ents if ent.label_ == 'ORG']
print(f"Organizations: {orgs}")
# Output: Persons: ['Bill Gates', 'Sundar Pichai']
#         Organizations: ['Microsoft', 'Google']
```

### Example 4: Visualize Entities with displaCy
```python
import spacy
from spacy import displacy

# Load model
nlp = spacy.load('en_core_web_sm')

# Text with multiple entities
text = "Tesla, led by Elon Musk, opened a factory in Berlin, Germany in 2021"
doc = nlp(text)

# Display entities (in Jupyter notebook or save to HTML)
displacy.render(doc, style='ent', jupyter=False)

# Or print entity information
for ent in doc.ents:
    print(f"{ent.text} ({ent.label_})")
# Output: Entity visualization or printed list
```

### Example 5: Custom Entity Recognition
```python
import spacy
from spacy.tokens import Span

# Load model
nlp = spacy.load('en_core_web_sm')

# Text
text = "I love using PyTorch and TensorFlow for deep learning"
doc = nlp(text)

# Add custom entities (tech tools)
# Get character positions for "PyTorch"
pytorch_span = Span(doc, 3, 4, label="TECH")
tensorflow_span = Span(doc, 5, 6, label="TECH")

# Add to doc.ents
doc.ents = list(doc.ents) + [pytorch_span, tensorflow_span]

# Print all entities
for ent in doc.ents:
    print(f"{ent.text}: {ent.label_}")
# Output: Shows both standard and custom entities
```

---

## 18. TEXT NORMALIZATION

**Kya Hai:** Text ko consistent format mein laana - multiple preprocessing steps ka combination (lowercasing, removing extras etc).

**Kab Use Karein:** Complete preprocessing pipeline mein - sabhi cleaning steps ko ek saath apply karna.

**Scenario:** Raw data ko model-ready format mein convert karne ke liye comprehensive cleaning.

### Example 1: Basic Normalization Pipeline
```python
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Initialize
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def normalize_text(text):
    # Lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespaces
    text = ' '.join(text.split())
    # Tokenize
    tokens = text.split()
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Test
text = "This is an EXAMPLE!!! It has 123 numbers and symbols @#$"
normalized = normalize_text(text)
print(normalized)
# Output: example number symbol
```

### Example 2: Complete Normalization with All Steps
```python
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def complete_normalization(text):
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # 3. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # 4. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 5. Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # 6. Remove extra whitespace
    text = ' '.join(text.split())
    
    # 7. Tokenize
    tokens = word_tokenize(text)
    
    # 8. Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # 9. Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

# Test
text = "<p>Check https://example.com for 100 deals!!! It's AMAZING.</p>"
normalized = complete_normalization(text)
print(normalized)
# Output: check deal amazing
```

### Example 3: spaCy-based Normalization
```python
import spacy
import re

# Load model
nlp = spacy.load('en_core_web_sm')

def spacy_normalize(text):
    # Remove URLs and emails
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    
    # Process with spaCy
    doc = nlp(text.lower())
    
    # Lemmatize, remove stopwords and punctuation
    tokens = [token.lemma_ for token in doc 
             if not token.is_stop 
             and not token.is_punct 
             and token.is_alpha]
    
    return ' '.join(tokens)

# Test
text = "I'm running to the store!!! Email me at test@example.com"
normalized = spacy_normalize(text)
print(normalized)
# Output: run store email
```

### Example 4: Normalization for Social Media Text
```python
import re
from nltk.tokenize import word_tokenize

def social_media_normalize(text):
    # Lowercase
    text = text.lower()
    
    # Remove mentions (@username)
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags (keep the word)
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove emojis (basic pattern)
    emoji_pattern = re.compile("["
                              u"\U0001F600-\U0001F64F"
                              u"\U0001F300-\U0001F5FF"
                              "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    # Remove extra spaces
    text = ' '.join(text.split())
    
    return text

# Test tweet
tweet = "@user Check out this #amazing product 😊 https://example.com"
normalized = social_media_normalize(tweet)
print(normalized)
# Output: check out this amazing product
```

### Example 5: Pandas DataFrame Normalization Pipeline
```python
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Sample dataframe
df = pd.DataFrame({
    'reviews': [
        'This product is AMAZING!!! 5 stars ⭐',
        'Bad quality.. not worth $50',
        'Great service at www.example.com'
    ]
})

# Normalization function
def normalize_pipeline(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # All cleaning steps
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    
    return ' '.join(tokens)

# Apply normalization
df['normalized'] = df['reviews'].apply(normalize_pipeline)
print(df)
# Output: DataFrame with normalized reviews
```

---

## 19. HANDLING MISSING VALUES

**Kya Hai:** Dataset mein jo text fields empty/null hain unhe handle karna - remove ya fill karna.

**Kab Use Karein:** Data cleaning ka initial step - before any preprocessing, missing data handle karna zaruri hai.

**Scenario:** CSV file load ki jisme kuch rows mein text missing hai, unhe handle karna before analysis.

### Example 1: Pandas Drop Missing Values
```python
import pandas as pd
import numpy as np

# Sample dataframe with missing values
df = pd.DataFrame({
    'text': ['Good product', None, 'Bad quality', np.nan, 'Excellent'],
    'rating': [5, 4, 2, 3, 5]
})

print("Before:")
print(df)

# Drop rows with missing text
df_clean = df.dropna(subset=['text'])

print("\nAfter:")
print(df_clean)
# Output: DataFrame without missing values
```

### Example 2: Fill Missing Values with Default Text
```python
import pandas as pd
import numpy as np

# Dataframe with missing values
df = pd.DataFrame({
    'comments': ['Great', None, 'Poor', np.nan, 'Average'],
    'id': [1, 2, 3, 4, 5]
})

# Fill missing values with placeholder
df['comments'] = df['comments'].fillna('No comment provided')

print(df)
# Output: Missing values replaced with placeholder
```

### Example 3: Check and Report Missing Values
```python
import pandas as pd
import numpy as np

# Sample data
df = pd.DataFrame({
    'review': ['Good', None, 'Bad', 'OK', np.nan],
    'title': ['Title1', 'Title2', None, 'Title4', 'Title5']
})

# Check missing values
print("Missing values count:")
print(df.isnull().sum())

# Percentage of missing values
print("\nMissing percentage:")
print((df.isnull().sum() / len(df)) * 100)
# Output: Count and percentage of missing values
```

### Example 4: Forward/Backward Fill
```python
import pandas as pd
import numpy as np

# Sequential data with missing values
df = pd.DataFrame({
    'description': ['First', None, None, 'Fourth', None, 'Sixth']
})

print("Original:")
print(df)

# Forward fill (use previous value)
df['ffill'] = df['description'].fillna(method='ffill')

# Backward fill (use next value)
df['bfill'] = df['description'].fillna(method='bfill')

print("\nAfter fill:")
print(df)
# Output: Shows forward and backward fill results
```

### Example 5: Complex Missing Value Strategy
```python
import pandas as pd
import numpy as np

# Dataframe
df = pd.DataFrame({
    'text': ['Good', None, '', 'Bad', np.nan, 'Average', '   '],
    'category': ['A', 'B', 'C', 'A', 'B', 'C', 'A']
})

# Function to handle various missing cases
def handle_missing(df, column):
    # Replace empty strings and whitespace with NaN
    df[column] = df[column].replace(r'^\s*$', np.nan, regex=True)
    
    # Count missing
    missing_count = df[column].isnull().sum()
    print(f"Missing values: {missing_count}")
    
    # Strategy 1: Drop if too many missing (>50%)
    if missing_count > len(df) * 0.5:
        df = df.dropna(subset=[column])
        print("Dropped rows due to high missing percentage")
    else:
        # Strategy 2: Fill with placeholder
        df[column] = df[column].fillna('Unknown')
        print("Filled missing with 'Unknown'")
    
    return df

# Apply handling
df_clean = handle_missing(df, 'text')
print("\nCleaned data:")
print(df_clean)
# Output: Cleaned dataframe with missing values handled
```

---

## 20. REMOVING DUPLICATES

**Kya Hai:** Dataset mein jo same/similar text entries multiple times hain unhe remove karna - sirf unique records rakhna.

**Kab Use Karein:** Data cleaning ka important step - duplicates model ko confuse kar sakte hain aur bias create karte hain.

**Scenario:** Customer reviews scrape kiye jisme same review multiple times hai, ya survey data mein duplicate responses.

### Example 1: Pandas Drop Exact Duplicates
```python
import pandas as pd

# Sample dataframe with duplicates
df = pd.DataFrame({
    'text': ['Good product', 'Bad quality', 'Good product', 'Average', 'Bad quality'],
    'rating': [5, 2, 5, 3, 2]
})

print("Before removing duplicates:")
print(df)

# Remove exact duplicates
df_unique = df.drop_duplicates(subset=['text'])

print("\nAfter removing duplicates:")
print(df_unique)
# Output: DataFrame with only unique texts
```

### Example 2: Remove Duplicates Keep First/Last
```python
import pandas as pd

# Data with duplicates
df = pd.DataFrame({
    'comment': ['Great', 'Good', 'Great', 'Poor', 'Good', 'Excellent'],
    'timestamp': pd.date_range('2024-01-01', periods=6)
})

print("Original:")
print(df)

# Keep first occurrence
df_first = df.drop_duplicates(subset=['comment'], keep='first')
print("\nKeep first:")
print(df_first)

# Keep last occurrence
df_last = df.drop_duplicates(subset=['comment'], keep='last')
print("\nKeep last:")
print(df_last)
# Output: Shows different strategies
```

### Example 3: Case-Insensitive Duplicate Removal
```python
import pandas as pd

# Data with case variations
df = pd.DataFrame({
    'reviews': ['Good Product', 'bad quality', 'Good product', 'BAD QUALITY', 'Average']
})

print("Original:")
print(df)

# Create lowercase version for comparison
df['reviews_lower'] = df['reviews'].str.lower()

# Remove duplicates based on lowercase version
df_unique = df.drop_duplicates(subset=['reviews_lower'])

# Drop the helper column
df_unique = df_unique.drop('reviews_lower', axis=1)

print("\nAfter case-insensitive deduplication:")
print(df_unique)
# Output: Removes case-insensitive duplicates
```

### Example 4: Find and Display Duplicates
```python
import pandas as pd

# Sample data
df = pd.DataFrame({
    'text': ['First', 'Second', 'First', 'Third', 'Second', 'Fourth'],
    'id': [1, 2, 3, 4, 5, 6]
})

# Find duplicates
duplicates = df[df.duplicated(subset=['text'], keep=False)]

print("Duplicate entries:")
print(duplicates)

# Count of duplicates
print(f"\nTotal duplicate rows: {duplicates.shape[0]}")

# Remove duplicates
df_clean = df.drop_duplicates(subset=['text'])
print(f"\nOriginal rows: {df.shape[0]}, After dedup: {df_clean.shape[0]}")
# Output: Shows duplicate entries and statistics
```

### Example 5: Fuzzy/Similar Text Deduplication
```python
import pandas as pd
from difflib import SequenceMatcher

# Function to check similarity
def similar(a, b, threshold=0.9):
    return SequenceMatcher(None, a, b).ratio() > threshold

# Sample data with similar texts
df = pd.DataFrame({
    'text': [
        'Machine learning is great',
        'Machine learning is great!',
        'Machine learning is awesome',
        'Deep learning is powerful',
        'Machine learning is great.'
    ]
})

print("Original:")
print(df)

# Remove similar duplicates
unique_texts = []
for text in df['text']:
    # Check if similar text already exists
    is_duplicate = any(similar(text.lower(), existing.lower()) 
                      for existing in unique_texts)
    if not is_duplicate:
        unique_texts.append(text)

df_unique = pd.DataFrame({'text': unique_texts})

print("\nAfter fuzzy deduplication:")
print(df_unique)
# Output: Removes similar texts based on similarity threshold
```

---

## SUMMARY: Kab Kaunsi Technique Use Karein

**Basic Cleaning (Har task mein):**
- Tokenization (sabse pehle)
- Lowercasing
- Removing extra whitespaces
- Handling missing values
- Removing duplicates

**Text ka Source dekho:**
- Web scraping → Remove HTML tags, URLs
- Social media → Remove emojis, hashtags, mentions  
- User input → Spell correction, expand contractions
- Formal documents → Remove accents, normalize text

**Task dekho:**
- **Sentiment Analysis** → Stopwords carefully remove (not, never important hain)
- **Topic Modeling** → Aggressive cleaning - remove stopwords, lemmatize
- **Search/IR** → Stemming fast hai, use karo
- **Chatbot/QA** → Lemmatization better, NER useful
- **Classification** → Complete normalization pipeline

**Model type dekho:**
- Traditional ML (Naive Bayes, SVM) → Heavy preprocessing zaruri
- Deep Learning (BERT, GPT) → Minimal preprocessing, model khud handle karta hai
- Rule-based systems → POS tagging, NER important

Ye sab techniques aapko flexible hona sikhati hain - har situation mein sahi combination choose karo! 🚀

# Short Answer:

**Haan, mostly kar sakte ho, lekin 100% nahi.**

---

## NLTK Se Kar Sakte Ho (10/20):
1. ✅ Tokenization
2. ✅ Lowercasing (Python built-in)
3. ✅ Stopwords Removal
4. ✅ Punctuation Removal (Python string module)
5. ✅ Stemming
6. ✅ Lemmatization
7. ❌ Removing Numbers (RegEx chahiye)
8. ❌ Removing Special Characters (RegEx chahiye)
9. ❌ Removing Extra Whitespaces (RegEx chahiye)
10. ❌ Removing URLs (RegEx chahiye)
11. ❌ Removing HTML Tags (BeautifulSoup chahiye)
12. ❌ Removing Emojis (emoji/RegEx library chahiye)
13. ❌ Spell Correction (TextBlob/pyspellchecker chahiye)
14. ❌ Expanding Contractions (contractions library chahiye)
15. ❌ Removing Accents (unidecode chahiye)
16. ✅ POS Tagging
17. ✅ NER
18. ⚠️ Text Normalization (partial - combination hai)
19. ✅ Handling Missing Values (Pandas chahiye)
20. ✅ Removing Duplicates (Pandas chahiye)

---

## spaCy Se Kar Sakte Ho (12/20):
1. ✅ Tokenization
2. ✅ Lowercasing (Python built-in)
3. ✅ Stopwords Removal
4. ✅ Punctuation Removal
5. ❌ Stemming (spaCy mein nahi hai)
6. ✅ Lemmatization
7. ⚠️ Removing Numbers (token filtering se)
8. ⚠️ Removing Special Characters (token filtering se)
9. ❌ Removing Extra Whitespaces (RegEx chahiye)
10. ❌ Removing URLs (RegEx chahiye)
11. ❌ Removing HTML Tags (BeautifulSoup chahiye)
12. ❌ Removing Emojis (emoji library chahiye)
13. ❌ Spell Correction (external library chahiye)
14. ❌ Expanding Contractions (contractions library chahiye)
15. ❌ Removing Accents (unidecode chahiye)
16. ✅ POS Tagging
17. ✅ NER
18. ⚠️ Text Normalization (partial)
19. ✅ Handling Missing Values (Pandas chahiye)
20. ✅ Removing Duplicates (Pandas chahiye)

---

## Final Verdict:

**NLTK + Python built-ins (string, re)** = 15-16/20 techniques ✅

**spaCy + Python built-ins (string, re)** = 16-17/20 techniques ✅

**Zaruri Extra Libraries:**
- `pandas` - Missing values, duplicates
- `re` (RegEx) - URLs, numbers, special chars, whitespace
- `BeautifulSoup` - HTML tags
- `emoji` - Emoji removal
- `contractions` - Expand contractions
- `unidecode` - Accent removal
- `TextBlob/pyspellchecker` - Spell correction

**Conclusion:** Sirf NLTK ya spaCy se **80-85%** preprocessing ho sakti hai. Baaki 15-20% ke liye basic Python libraries (re, string) aur kuch specific libraries chahiye hongi. 🎯


# Industry Mein Kya Use Hota Hai:

## **Most Common (90% Companies):**

### 1. **spaCy** 🏆
- Production-ready, fast
- Google, Microsoft, Facebook use karte hain
- Industrial standard for NLP pipelines

### 2. **NLTK**
- Research aur prototyping ke liye
- Educational purposes
- Production mein kam

### 3. **Hugging Face Transformers**
- Modern NLP (BERT, GPT models)
- Minimal preprocessing (models khud handle karte hain)
- Trending heavily

---

## **Big Tech Stack:**

**Google/Facebook/Amazon:**
- spaCy (preprocessing)
- Custom in-house tools
- Transformer models (minimal preprocessing)

**Startups:**
- spaCy + Python basics (re, pandas)
- Sometimes NLTK for specific tasks

---

## **Modern Trend (2024-25):**

### **Minimal Preprocessing! ⚡**
- BERT/GPT jaise models use karo
- Sirf basic cleaning:
  - Lowercasing (optional)
  - Remove URLs/HTML
  - Tokenization (model ka built-in)
- **90% preprocessing techniques obsolete ho rahe hain modern models ke saath**

---

## **Industry Pipeline:**

```
Raw Text 
→ Basic Cleaning (URLs, HTML, special chars) 
→ spaCy/Hugging Face Tokenizer
→ Direct to Model
```

**Old way (ab kam use hoti):**
```
Raw Text 
→ 15-20 preprocessing steps 
→ TF-IDF/BoW 
→ Traditional ML Model
```

---

## **Summary:**

**Production mein:**
- **spaCy** (60-70% companies)
- **Hugging Face** (30-40% modern startups)
- **Custom pipelines** (big tech)

**Key Point:** Industry **minimal preprocessing** prefer karti hai modern transformers ke saath! 🚀


# Hugging Face Transformers - Text Preprocessing Guide

## Important Note Pehle:

**Hugging Face Transformers mein preprocessing bahut minimal hai!** Kyunki BERT, GPT jaise models khud bahut kuch handle karte hain.

**Hugging Face specifically karta hai:**
- Tokenization (main feature)
- Encoding
- Attention masks
- Special tokens handling

**Baaki preprocessing (URLs remove, HTML clean, etc.) aapko manually karni padegi using Python/regex.**

---

## **Real Industry Approach with Hugging Face:**

```python
# Traditional approach (20 steps preprocessing)
Raw Text → Clean → Tokenize → Stopwords → Stem → TF-IDF → Model

# Hugging Face approach (minimal preprocessing)
Raw Text → Basic clean (URLs/HTML) → Tokenizer → Model
```

---

Let me show you **practical examples** of what Hugging Face actually does + minimal preprocessing:

---

## 1. TOKENIZATION (Hugging Face Ka Main Feature)

### Example 1: Basic Tokenization with BERT
```python
from transformers import BertTokenizer

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Raw text
text = "Natural Language Processing is amazing!"

# Tokenize using Hugging Face
tokens = tokenizer.tokenize(text)
print("Tokens:", tokens)
# Output: ['natural', 'language', 'processing', 'is', 'amazing', '!']

# Get token IDs (numbers jo model samajhta hai)
token_ids = tokenizer.encode(text)
print("Token IDs:", token_ids)
# Output: [101, 3019, 2653, 6364, 2003, 6429, 999, 102]
```

### Example 2: Tokenization with Special Tokens
```python
from transformers import AutoTokenizer

# Load any model's tokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# Text
text = "I love machine learning"

# Tokenize with all features
encoding = tokenizer(
    text,
    add_special_tokens=True,  # [CLS] and [SEP] tokens add hote hain
    max_length=20,            # Maximum length
    padding='max_length',     # Pad to max_length
    truncation=True,          # Truncate if longer
    return_tensors='pt'       # PyTorch tensors return karo
)

print("Input IDs:", encoding['input_ids'])
print("Attention Mask:", encoding['attention_mask'])
# Output: Encoded tensors with padding and special tokens
```

---

## 2. LOWERCASING (Automatic with Most Tokenizers)

### Example 1: BERT Uncased (Automatic Lowercase)
```python
from transformers import BertTokenizer

# BERT uncased automatically lowercase karta hai
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Mixed case text
text = "Machine LEARNING is AWESOME"

# Tokenize (automatically lowercase ho jayega)
tokens = tokenizer.tokenize(text)
print(tokens)
# Output: ['machine', 'learning', 'is', 'awesome']
```

### Example 2: Manual Control - Cased vs Uncased
```python
from transformers import AutoTokenizer

# Uncased model (lowercase karta hai)
tokenizer_uncased = AutoTokenizer.from_pretrained('bert-base-uncased')

# Cased model (case preserve karta hai)
tokenizer_cased = AutoTokenizer.from_pretrained('bert-base-cased')

text = "Python and PYTHON are same"

# Uncased
print("Uncased:", tokenizer_uncased.tokenize(text))
# Output: ['python', 'and', 'python', 'are', 'same']

# Cased
print("Cased:", tokenizer_cased.tokenize(text))
# Output: ['Python', 'and', 'P', '##Y', '##TH', '##ON', 'are', 'same']
```

---

## 3. STOPWORDS REMOVAL (Not Recommended with Transformers!)

**Note:** Transformers ko stopwords chahiye! "not", "is" jaise words context ke liye important hain.

### Example 1: Why NOT to Remove (Sentiment Changes)
```python
from transformers import pipeline

# Sentiment analyzer
sentiment = pipeline('sentiment-analysis')

# Original text
text1 = "This is not good"
result1 = sentiment(text1)
print("Original:", result1)
# Output: NEGATIVE

# Stopwords removed (wrong!)
text2 = "good"  # "not" remove kar diya
result2 = sentiment(text2)
print("After removing stopwords:", result2)
# Output: POSITIVE (WRONG! Meaning change ho gaya!)
```

### Example 2: Manual Stopwords (Only for Special Cases)
```python
from transformers import AutoTokenizer
from nltk.corpus import stopwords

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Stopwords (usually NOT recommended)
stop_words = set(stopwords.words('english'))

# Text
text = "This is a sample text for demonstration"

# Tokenize first
tokens = tokenizer.tokenize(text)

# Remove stopwords (NOT recommended for transformers!)
filtered_tokens = [t for t in tokens if t not in stop_words]

print("Original tokens:", tokens)
print("After stopword removal:", filtered_tokens)
# Note: Ye approach transformers ke saath use NAHI karna chahiye!
```

---

## 4. PUNCTUATION REMOVAL (Already Handled by Tokenizers!)

### Example 1: Tokenizer Handles Punctuation
```python
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Text with punctuation
text = "Hello!!! How are you? I'm great."

# Tokenizer punctuation ko handle karta hai
tokens = tokenizer.tokenize(text)
print("Tokens:", tokens)
# Output: ['hello', '!', '!', '!', 'how', 'are', 'you', '?', 'i', "'", 'm', 'great', '.']

# Punctuation separate tokens ban jate hain, model handle kar leta hai
```

### Example 2: Manual Pre-cleaning (Optional)
```python
from transformers import AutoTokenizer
import re

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# Text with excessive punctuation
text = "Amazing product!!!??? Really????"

# Option 1: Direct tokenization (tokenizer handles it)
tokens1 = tokenizer.tokenize(text)
print("With punctuation:", tokens1)

# Option 2: Manual cleaning (if needed)
text_clean = re.sub(r'[^\w\s]', '', text)
tokens2 = tokenizer.tokenize(text_clean)
print("Punctuation removed:", tokens2)
# Note: Usually manual removal NOT needed
```

---

## 5. STEMMING (NOT Used with Transformers!)

**Important:** Transformers ko stemming ki zarurat NAHI! Models context se samajh lete hain.

### Example 1: Why Stemming is NOT Needed
```python
from transformers import pipeline

# Load model
classifier = pipeline('text-classification', model='distilbert-base-uncased-finetuned-sst-2-english')

# Different forms of same word
text1 = "I am running to the store"
text2 = "I run to the store"
text3 = "I ran to the store"

# Model samajh leta hai without stemming
print(classifier(text1))
print(classifier(text2))
print(classifier(text3))
# All handle ho jayenge properly without stemming
```

### Example 2: Comparison (Old vs New Approach)
```python
from transformers import AutoTokenizer
from nltk.stem import PorterStemmer

# Old approach (NOT for transformers)
stemmer = PorterStemmer()
text = "running runs runner"
old_approach = [stemmer.stem(word) for word in text.split()]
print("Old (stemming):", old_approach)
# Output: ['run', 'run', 'runner']

# New approach (Hugging Face)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
new_approach = tokenizer.tokenize(text)
print("New (tokenizer):", new_approach)
# Output: ['running', 'runs', 'runner']
# Model embeddings mein similar representation honge automatically
```

---

## 6. LEMMATIZATION (NOT Needed with Transformers!)

### Example 1: Transformers Handle Without Lemmatization
```python
from transformers import pipeline

# Fill mask pipeline
fill_mask = pipeline('fill-mask', model='bert-base-uncased')

# Different forms
sentence1 = "The cats [MASK] running"
sentence2 = "The cat [MASK] running"

# Model context se samajh leta hai
result1 = fill_mask(sentence1)
result2 = fill_mask(sentence2)

print("Plural:", result1[0]['token_str'])
print("Singular:", result2[0]['token_str'])
# Both cases properly handle hote hain
```

### Example 2: Pre-processing Comparison
```python
from transformers import AutoTokenizer

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Text with different word forms
text = "The children were running and the child was walking"

# Direct tokenization (NO lemmatization needed)
tokens = tokenizer.tokenize(text)
print("Tokens:", tokens)
# Output: ['the', 'children', 'were', 'running', 'and', 'the', 'child', 'was', 'walking']

# Model's embeddings automatically similar forms ko relate karta hai
encoding = tokenizer(text, return_tensors='pt')
print("Encoded successfully without lemmatization")
```

---

## 7. REMOVING NUMBERS (Manual Pre-processing)

### Example 1: Basic Number Removal Before Tokenization
```python
from transformers import AutoTokenizer
import re

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Text with numbers
text = "I bought 5 apples for $10 in 2024"

# Remove numbers (if needed for your task)
text_no_numbers = re.sub(r'\d+', '', text)

# Tokenize
tokens_with_nums = tokenizer.tokenize(text)
tokens_no_nums = tokenizer.tokenize(text_no_numbers)

print("With numbers:", tokens_with_nums)
print("Without numbers:", tokens_no_nums)
# Output shows difference
```

### Example 2: Conditional Number Handling
```python
from transformers import AutoTokenizer
import re

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# Function for flexible number handling
def preprocess_text(text, remove_numbers=False):
    if remove_numbers:
        # Remove standalone numbers
        text = re.sub(r'\b\d+\b', '', text)
    return text

# Example texts
text1 = "Product123 costs $50 and has 5 star rating"
text2 = "The year 2024 was great"

# With number removal
clean1 = preprocess_text(text1, remove_numbers=True)
clean2 = preprocess_text(text2, remove_numbers=True)

print("Original:", text1)
print("Cleaned:", clean1)
print("Tokens:", tokenizer.tokenize(clean1))
```

---

## 8. REMOVING SPECIAL CHARACTERS (Pre-processing)

### Example 1: Clean Before Tokenization
```python
from transformers import AutoTokenizer
import re

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Text with special characters
text = "Email: test@example.com | Price: $100 #discount @user"

# Remove special patterns
def clean_special_chars(text):
    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)
    # Remove @mentions
    text = re.sub(r'@\w+', '', text)
    # Remove # but keep text
    text = re.sub(r'#(\w+)', r'\1', text)
    # Clean extra spaces
    text = ' '.join(text.split())
    return text

# Clean and tokenize
clean_text = clean_special_chars(text)
tokens = tokenizer.tokenize(clean_text)

print("Original:", text)
print("Cleaned:", clean_text)
print("Tokens:", tokens)
```

### Example 2: Keep Only Alphanumeric
```python
from transformers import AutoTokenizer
import re

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Messy text
text = "Python@3.9 >> Java & C++ are #trending!!!"

# Method 1: Keep letters and spaces only
clean1 = re.sub(r'[^a-zA-Z\s]', ' ', text)
clean1 = ' '.join(clean1.split())

# Method 2: Keep alphanumeric
clean2 = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
clean2 = ' '.join(clean2.split())

print("Original:", text)
print("Letters only:", tokenizer.tokenize(clean1))
print("Alphanumeric:", tokenizer.tokenize(clean2))
```

---

## 9. REMOVING EXTRA WHITESPACES (Pre-processing)

### Example 1: Clean Whitespace Before Tokenization
```python
from transformers import AutoTokenizer
import re

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Text with irregular spacing
text = "This    has   multiple     spaces   and\n\nnewlines\t\ttabs"

# Clean whitespace
clean_text = re.sub(r'\s+', ' ', text).strip()

# Tokenize
tokens_messy = tokenizer.tokenize(text)
tokens_clean = tokenizer.tokenize(clean_text)

print("Original text:", repr(text))
print("Cleaned text:", clean_text)
print("Tokens (clean):", tokens_clean)
```

### Example 2: Batch Cleaning
```python
from transformers import AutoTokenizer
import re

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# Multiple texts with spacing issues
texts = [
    "Text   with    spaces",
    "Another\n\ntext\twith\nissues",
    "   Leading and trailing   "
]

# Function to clean
def clean_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip()

# Clean and tokenize all
clean_texts = [clean_whitespace(t) for t in texts]
encodings = tokenizer(clean_texts, padding=True, truncation=True, return_tensors='pt')

print("Cleaned texts:", clean_texts)
print("Encoded shape:", encodings['input_ids'].shape)
```

---

## 10. REMOVING URLs (Pre-processing)

### Example 1: Remove URLs Before Tokenization
```python
from transformers import AutoTokenizer
import re

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Text with URLs
text = "Check out https://example.com and www.test.com for more info"

# Remove URLs
def remove_urls(text):
    # Remove http/https URLs
    text = re.sub(r'http\S+|https\S+', '', text)
    # Remove www URLs
    text = re.sub(r'www\.\S+', '', text)
    # Clean extra spaces
    text = ' '.join(text.split())
    return text

# Clean and tokenize
clean_text = remove_urls(text)
tokens = tokenizer.tokenize(clean_text)

print("Original:", text)
print("Cleaned:", clean_text)
print("Tokens:", tokens)
```

### Example 2: Social Media Text Cleaning
```python
from transformers import AutoTokenizer
import re

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Social media post with URLs
posts = [
    "Great article! https://blog.example.com/post123",
    "Visit www.mysite.com for deals",
    "No URLs in this post"
]

# Comprehensive URL removal
def clean_social_text(text):
    # Remove all URL patterns
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(url_pattern, '', text)
    text = re.sub(r'www\.\S+', '', text)
    return text.strip()

# Process all posts
clean_posts = [clean_social_text(post) for post in posts]

# Tokenize batch
encodings = tokenizer(clean_posts, padding=True, truncation=True)

for i, (orig, clean) in enumerate(zip(posts, clean_posts)):
    print(f"\nPost {i+1}:")
    print(f"Original: {orig}")
    print(f"Cleaned: {clean}")
```

---

## 11. REMOVING HTML TAGS (Pre-processing)

### Example 1: BeautifulSoup + Hugging Face
```python
from transformers import AutoTokenizer
from bs4 import BeautifulSoup

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# HTML content
html_text = "<p>This is a <b>great</b> product!</p><br/><a href='#'>Click here</a>"

# Remove HTML tags
soup = BeautifulSoup(html_text, 'html.parser')
clean_text = soup.get_text(separator=' ')

# Tokenize clean text
tokens = tokenizer.tokenize(clean_text)

print("Original HTML:", html_text)
print("Cleaned text:", clean_text)
print("Tokens:", tokens)
```

### Example 2: RegEx Method (Faster)
```python
from transformers import AutoTokenizer
import re

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# HTML content
html_texts = [
    "<div>Review: <strong>Excellent</strong> quality</div>",
    "<p>Price is <span class='discount'>$50</span></p>",
    "No HTML here"
]

# Function to remove HTML
def remove_html(text):
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Unescape HTML entities
    import html
    text = html.unescape(text)
    return text.strip()

# Clean all texts
clean_texts = [remove_html(t) for t in html_texts]

# Batch tokenization
encodings = tokenizer(clean_texts, padding=True, return_tensors='pt')

for orig, clean in zip(html_texts, clean_texts):
    print(f"Original: {orig}")
    print(f"Cleaned: {clean}\n")
```

---

## 12. REMOVING EMOJIS (Pre-processing)

### Example 1: Remove Emojis with Regex
```python
from transformers import AutoTokenizer
import re

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Text with emojis
text = "I love Python 😍 and AI is amazing 🚀👍"

# Emoji pattern
emoji_pattern = re.compile("["
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map
    u"\U0001F1E0-\U0001F1FF"  # flags
    "]+", flags=re.UNICODE)

# Remove emojis
clean_text = emoji_pattern.sub(r'', text)
clean_text = ' '.join(clean_text.split())  # Clean extra spaces

# Tokenize
tokens = tokenizer.tokenize(clean_text)

print("Original:", text)
print("Cleaned:", clean_text)
print("Tokens:", tokens)
```

### Example 2: Using emoji Library
```python
from transformers import AutoTokenizer
import emoji

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Social media posts with emojis
posts = [
    "Great product! 😊👍",
    "Not satisfied 😠😤",
    "Amazing experience ❤️🎉🎊"
]

# Remove emojis
clean_posts = [emoji.replace_emoji(post, '') for post in posts]
clean_posts = [' '.join(p.split()) for p in clean_posts]

# Batch encode
encodings = tokenizer(clean_posts, padding=True, truncation=True, return_tensors='pt')

for orig, clean in zip(posts, clean_posts):
    print(f"Original: {orig}")
    print(f"Cleaned: {clean}\n")
```

---

## 13. SPELL CORRECTION (Pre-processing - Optional)

### Example 1: TextBlob + Hugging Face
```python
from transformers import AutoTokenizer
from textblob import TextBlob

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Text with typos
text = "I havv goood experienc with machne lerning"

# Correct spelling
blob = TextBlob(text)
corrected_text = str(blob.correct())

# Tokenize corrected text
tokens_original = tokenizer.tokenize(text)
tokens_corrected = tokenizer.tokenize(corrected_text)

print("Original:", text)
print("Tokens (original):", tokens_original)
print("\nCorrected:", corrected_text)
print("Tokens (corrected):", tokens_corrected)
```

### Example 2: Batch Spell Correction
```python
from transformers import AutoTokenizer
from textblob import TextBlob

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# User reviews with typos
reviews = [
    "Tis prodct is gud",
    "Vry bad qualiti",
    "Excelent servce"
]

# Correct all
def correct_spelling(text):
    return str(TextBlob(text).correct())

corrected_reviews = [correct_spelling(r) for r in reviews]

# Encode both versions
original_enc = tokenizer(reviews, padding=True, return_tensors='pt')
corrected_enc = tokenizer(corrected_reviews, padding=True, return_tensors='pt')

for orig, corr in zip(reviews, corrected_reviews):
    print(f"Original: {orig}")
    print(f"Corrected: {corr}\n")
```

---

## 14. EXPANDING CONTRACTIONS (Pre-processing)

### Example 1: Contractions Library + Tokenizer
```python
from transformers import AutoTokenizer
import contractions

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Text with contractions
text = "I'm learning NLP and it's awesome! Can't wait to learn more."

# Expand contractions
expanded_text = contractions.fix(text)

# Tokenize both
tokens_original = tokenizer.tokenize(text)
tokens_expanded = tokenizer.tokenize(expanded_text)

print("Original:", text)
print("Tokens:", tokens_original)
print("\nExpanded:", expanded_text)
print("Tokens:", tokens_expanded)
```

### Example 2: Batch Processing with Expansion
```python
from transformers import AutoTokenizer
import contractions

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Multiple texts with contractions
texts = [
    "I'm sure you're right",
    "Don't worry, it's fine",
    "We shouldn't have done that"
]

# Expand all contractions
expanded_texts = [contractions.fix(t) for t in texts]

# Batch tokenization
original_enc = tokenizer(texts, padding=True, return_tensors='pt')
expanded_enc = tokenizer(expanded_texts, padding=True, return_tensors='pt')

print("Comparison:")
for orig, exp in zip(texts, expanded_texts):
    print(f"Original: {orig}")
    print(f"Expanded: {exp}\n")
```

---

## 15. REMOVING ACCENTS (Pre-processing)

### Example 1: Unidecode + Tokenizer
```python
from transformers import AutoTokenizer
from unidecode import unidecode

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Text with accents
text = "café résumé naïve Zürich François"

# Remove accents
clean_text = unidecode(text)

# Tokenize both
tokens_original = tokenizer.tokenize(text)
tokens_clean = tokenizer.tokenize(clean_text)

print("Original:", text)
print("Tokens:", tokens_original)
print("\nWithout accents:", clean_text)
print("Tokens:", tokens_clean)
```

### Example 2: International Names Processing
```python
from transformers import AutoTokenizer
from unidecode import unidecode

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# International names
names = [
    "José García",
    "François Müller",
    "Søren Bjørk"
]

# Remove accents
clean_names = [unidecode(name) for name in names]

# Tokenize
original_enc = tokenizer(names, padding=True)
clean_enc = tokenizer(clean_names, padding=True)

print("Name Comparison:")
for orig, clean in zip(names, clean_names):
    print(f"Original: {orig} → Clean: {clean}")
```

---

## 16. PART-OF-SPEECH TAGGING (Use spaCy, Not Hugging Face)

**Note:** Hugging Face transformers directly POS tagging nahi karte. Use spaCy or NLTK.

### Example 1: spaCy POS + Hugging Face Pipeline
```python
import spacy
from transformers import AutoTokenizer

# Load spaCy for POS
nlp = spacy.load('en_core_web_sm')

# Load Hugging Face tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Text
text = "The quick brown fox jumps over the lazy dog"

# Get POS tags with spaCy
doc = nlp(text)
pos_tags = [(token.text, token.pos_) for token in doc]

print("POS Tags:", pos_tags)

# Then tokenize for model
model_tokens = tokenizer.tokenize(text)
print("Model Tokens:", model_tokens)
```

### Example 2: Extract Nouns Before Model Processing
```python
import spacy
from transformers import pipeline

# Load spaCy
nlp = spacy.load('en_core_web_sm')

# Load sentiment analyzer
sentiment = pipeline('sentiment-analysis')

# Text
text = "The machine learning algorithm processes large datasets efficiently"

# Extract only nouns using POS tagging
doc = nlp(text)
nouns = ' '.join([token.text for token in doc if token.pos_ == 'NOUN'])

print("Original text:", text)
print("Only nouns:", nouns)

# Analyze sentiment of both
print("\nSentiment (full):", sentiment(text))
print("Sentiment (nouns):", sentiment(nouns))
```

---

## 17. NAMED ENTITY RECOGNITION (Use Pipeline)

### Example 1: Hugging Face NER Pipeline
```python
from transformers import pipeline

# Load NER pipeline
ner = pipeline('ner', model='dbmdz/bert-large-cased-finetuned-conll03-english')

# Text with entities
text = "Apple Inc. was founded by Steve Jobs in California in 1976"

# Extract entities
entities = ner(text)

print("Named Entities:")
for entity in entities:
    print(f"{entity['word']}: {entity['entity']} (score: {entity['score']:.2f})")
```

### Example 2: Grouped NER Results
```python
from transformers import pipeline

# Load NER with grouped entities
ner = pipeline('ner', grouped_entities=True)

# Multiple texts
texts = [
    "Microsoft and Google are competing in AI",
    "Elon Musk leads Tesla and SpaceX",
    "Amazon's Jeff Bezos founded the company in Seattle"
]

# Extract entities from all texts
for text in texts:
    print(f"\nText: {text}")
    entities = ner(text)
    for entity in entities:
        print(f"  {entity['word']}: {entity['entity_group']}")
```

---

## 18. TEXT NORMALIZATION (Complete Pipeline)

### Example 1: Full Preprocessing Pipeline
```python
from transformers import AutoTokenizer
import re
import contractions
from bs4 import BeautifulSoup

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def normalize_text(text):
    """Complete normalization pipeline"""
    # 1. Remove HTML
    text = BeautifulSoup(text, 'html.parser').get_text()
    
    # 2. Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # 3. Expand contractions
    text = contractions.fix(text)
    
    # 4. Remove emails
    text = re.sub(r'\S+@\S+', '', text)
    
    # 5. Remove special characters (keep alphanumeric and spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # 6. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 7. Lowercase (optional, tokenizer might do it)
    text = text.lower()
    
    return text

# Test text
raw_text = "<p>Check https://example.com! I'm excited 😊 Email: test@test.com</p>"

# Normalize
clean_text = normalize_text(raw_text)

# Tokenize
tokens = tokenizer.tokenize(clean_text)

print("Raw:", raw_text)
print("Normalized:", clean_text)
print("Tokens:", tokens)
```

### Example 2: Batch Normalization with Encoding
```python
from transformers import AutoTokenizer
import re

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

def quick_normalize(text):
    """Quick normalization for production"""
    # Remove URLs, emails, mentions
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    
    # Clean extra whitespace
    text = ' '.join(text.split())
    
    return text

# Batch of raw texts
raw_texts = [
    "Check this out: https://example.com @user",
    "Email me at test@email.com for details",
    "Visit www.site.com and follow @company"
]

# Normalize all
clean_texts = [quick_normalize(t) for t in raw_texts]

# Batch encode for model
encodings = tokenizer(
    clean_texts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors='pt'
)

print("Batch processing complete!")
print("Input shape:", encodings['input_ids'].shape)
for orig, clean in zip(raw_texts, clean_texts):
    print(f"\nOriginal: {orig}")
    print(f"Clean: {clean}")
```

---

## 19. HANDLING MISSING VALUES (Pandas + Tokenizer)

### Example 1: Clean Missing Before Tokenization
```python
import pandas as pd
from transformers import AutoTokenizer
import numpy as np

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# DataFrame with missing values
df = pd.DataFrame({
    'text': ['Good product', None, 'Bad quality', np.nan, 'Excellent'],
    'label': [1, 0, 0, 1, 1]
})

print("Before cleaning:")
print(df)

# Handle missing values
df['text'] = df['text'].fillna('')  # Fill with empty string

# Filter out empty texts
df = df[df['text'].str.strip() != '']

print("\nAfter cleaning:")
print(df)

# Now tokenize
encodings = tokenizer(df['text'].tolist(), padding=True, truncation=True)
print("\nTokenized successfully!")
```

### Example 2: Advanced Missing Value Handling
```python
import pandas as pd
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# Data with various missing patterns
df = pd.DataFrame({
    'review': ['Great', None, '', '   ', 'Poor', np.nan],
    'rating': [5, 4, 3, 2, 1, 3]
})

print("Original:")
print(df)

# Comprehensive cleaning function
def clean_missing(df, text_col):
    # Replace None with empty string
    df[text_col] = df[text_col].fillna('')
    
    # Replace whitespace-only with empty
    df[text_col] = df[text_col].str.strip()
    
    # Option 1: Remove empty rows
    df_clean = df[df[text_col] != ''].copy()
    
    # Option 2: Or fill with placeholder
    # df[text_col] = df[text_col].replace('', 'No review provided')
    
    return df_clean

# Clean
df_clean = clean_missing(df, 'review')

print("\nCleaned:")
print(df_clean)

# Batch tokenize
texts = df_clean['review'].tolist()
encodings = tokenizer(texts, padding=True, return_tensors='pt')
print(f"\nEncoded {len(texts)} reviews successfully!")
```

---

## 20. REMOVING DUPLICATES (Pandas + Tokenizer)

### Example 1: Remove Exact Duplicates
```python
import pandas as pd
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Data with duplicates
df = pd.DataFrame({
    'text': ['Good product', 'Bad quality', 'Good product', 'Average', 'Bad quality'],
    'user_id': [1, 2, 3, 4, 5]
})

print("Before removing duplicates:")
print(df)
print(f"Total rows: {len(df)}")

# Remove duplicates
df_unique = df.drop_duplicates(subset=['text'], keep='first')

print("\nAfter removing duplicates:")
print(df_unique)
print(f"Total rows: {len(df_unique)}")

# Tokenize unique texts
unique_texts = df_unique['text'].tolist()
encodings = tokenizer(unique_texts, padding=True, return_tensors='pt')
print(f"\nTokenized {len(unique_texts)} unique texts")
```

### Example 2: Case-Insensitive Deduplication
```python
import pandas as pd
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Data with case variations
df = pd.DataFrame({
    'reviews': ['Good Product', 'bad quality', 'Good product', 'BAD QUALITY', 'Average'],
    'date': pd.date_range('2024-01-01', periods=5)
})

print("Original:")
print(df)

# Create lowercase column for comparison
df['reviews_lower'] = df['reviews'].str.lower()

# Remove duplicates based on lowercase
df_unique = df.drop_duplicates(subset=['reviews_lower'], keep='first')

# Drop helper column
df_unique = df_unique.drop('reviews_lower', axis=1)

print("\nAfter deduplication:")
print(df_unique)

# Batch encode
encodings = tokenizer(
    df_unique['reviews'].tolist(),
    padding=True,
    truncation=True,
    return_tensors='pt'
)

print(f"\nProcessed {len(df_unique)} unique reviews")
```

---

## **COMPLETE PRODUCTION PIPELINE EXAMPLE:**

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import re
import contractions
from bs4 import BeautifulSoup
import torch

# Load model and tokenizer
model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Preprocessing function
def preprocess_pipeline(text):
    """Minimal but effective preprocessing"""
    # Remove HTML
    text = BeautifulSoup(text, 'html.parser').get_text()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove emails and mentions
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    
    # Expand contractions (optional)
    text = contractions.fix(text)
    
    # Clean whitespace
    text = ' '.join(text.split())
    
    return text

# Sample data
raw_data = {
    'reviews': [
        "<p>Great product! https://example.com</p>",
        "Bad quality... Contact: test@email.com",
        None,
        "I'm loving it! 😊",
        "Average, nothing special"
    ],
    'product_id': [1, 2, 3, 4, 5]
}

# Create DataFrame
df = pd.DataFrame(raw_data)

# Handle missing values
df = df.dropna(subset=['reviews'])

# Remove duplicates
df = df.drop_duplicates(subset=['reviews'])

# Preprocess all texts
df['clean_text'] = df['reviews'].apply(preprocess_pipeline)

# Filter empty texts
df = df[df['clean_text'].str.strip() != '']

print("Preprocessed Data:")
print(df[['reviews', 'clean_text']])

# Tokenize for model
texts = df['clean_text'].tolist()
encodings = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors='pt'
)

# Get predictions
with torch.no_grad():
    outputs = model(**encodings)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    labels = torch.argmax(predictions, dim=-1)

# Add predictions to dataframe
df['sentiment'] = labels.numpy()
df['confidence'] = predictions.max(dim=-1).values.numpy()

print("\nFinal Results:")
print(df[['clean_text', 'sentiment', 'confidence']])
```

---

## **KEY TAKEAWAYS:**

### ✅ **Hugging Face Strengths:**
1. Tokenization (main feature)
2. Encoding & padding
3. Batch processing
4. NER pipeline
5. Pre-trained models

### ❌ **What Hugging Face DOESN'T Do:**
1. HTML cleaning → Use BeautifulSoup
2. URL removal → Use regex
3. Emoji removal → Use emoji library
4. Spell correction → Use TextBlob
5. Contractions → Use contractions library
6. Accent removal → Use unidecode

### 🎯 **Industry Standard Pipeline:**
```
Raw Text
↓
Manual Cleaning (URLs, HTML, special chars) - Python/regex
↓
Hugging Face Tokenizer
↓
Model
```

### 💡 **Modern Approach = Minimal Preprocessing!**
- Transformers powerful hain
- Sirf necessary cleaning karo
- Model ko khud samajhne do
- Old techniques (stemming, stopwords) mostly skip karo

**Focus:** Clean karna > Transform karna! 🚀