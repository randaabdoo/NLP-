import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import PorterStemmer
import pandas as pd


# Sample emotion analysis dataset 
data = pd.read_csv('emotion_sentimen_dataset.csv')

# Tokenization and stemming

stemmer = PorterStemmer()

def tokenize_and_stem(text):
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return tokens, stemmed_tokens

data['tokenized_text'], data['stemmed_text'] = zip(*data['text'].apply(tokenize_and_stem))

# Display tokenized and stemmed words
for index, row in data.iterrows():
    print("Original Text:", row['text'])
    print("Tokenized Words:", row['tokenized_text'])
    print("Stemmed Words:", row['stemmed_text'])
    print()

