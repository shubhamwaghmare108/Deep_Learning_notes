import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
x = word_tokenize("Hello, my dog is cute")
print(x)
from nltk.stem import PorterStemmer
ps = PorterStemmer()
y = ps.stem("running")
print(y)
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
print(stop_words)
from nltk.tokenize import word_tokenize
sentence = "This is a sample sentence, showing off the stop words filtration."
word_tokens = word_tokenize(sentence)
filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
print(filtered_sentence)