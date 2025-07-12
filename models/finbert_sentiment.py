from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

def load_finbert():
    model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return nlp

def predict_sentiment(texts):
    if isinstance(texts, str):
        texts = [texts]
    classifier = load_finbert()
    return classifier(texts)
