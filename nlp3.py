from transformers import pipeline

#model_id = "sshleifer/distilbart-cnn-12-6"
#model_id = "dslim/bert-base-NER"

# Load sentiment analysis pipeline
classifier = pipeline("sentiment-analysis")#, model=model_id)

print("\n\n\n")

# Example text
texts = [
    "The new iPhone is just amazing. I'm so happy with it!",
    "I feel so sad about the news",
    "I'm not sure how I feel about this product.",
    #"Great, another meeting!"
]

# Classify sentiments
for text in texts:
    label = classifier(text)[0]['label']
    print(f"Text: {text}\nSentiment: {label}\n")
    #print(classifier(text))
    
# Load summarization pipeline
summarizer = pipeline("summarization")#, model=model_id)

#Exmaple long text
text = """ 
The Apollo Program, also known as project apollo, was the third united states human spaceflight.
"""

# Generate summary
summary = summarizer(text, max_length=50, min_length=10, do_sample=False)   

print("\n\n\n")

print("Summary:", summary[0]['summary_text'])