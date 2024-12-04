# run "python3 -m spacy download en_core_web_sm" first

import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy

# Download NLTK resources
nltk.download('punkt_tab')
nltk.download('vader_lexicon')

# Load the English NLP model from spacy
nlp = spacy.load('en_core_web_sm')

def analyze_sentence(sentence):
    # Tokenization
    tokens = word_tokenize(sentence)

    # Sentiment Analysis
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(sentence)
    if sentiment_score['compound'] >= 0.05:
        sentiment = 'Positive'
    elif sentiment_score['compound'] <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'

    # Named Entity Recognition
    doc = nlp(sentence)
    named_entities = [(ent.text, ent.label_) for ent in doc.ents]

    # Results
    return {
        'tokens': tokens,
        'sentiment': sentiment,
        'named_entities': named_entities
    }

# Input sentence
input_sentence = input("Enter a sentence: ")
result = analyze_sentence(input_sentence)

# Display results
print("\nResults:")
print("Tokens:", result['tokens'])
print("Sentiment:", result['sentiment'])
print("Named Entities:", result['named_entities'])