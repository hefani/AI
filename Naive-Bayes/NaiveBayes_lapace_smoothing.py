import math
from collections import defaultdict, Counter
import re
import nltk
import pandas as pd



class NaiveBayes:
    def __init__(self, vocabulary):
        self.word_counter = defaultdict(Counter)
        self.cnt = Counter()
        self.vocab = set(vocabulary)
        self.total_doc = 0

    def tokenize(self, text):
        return re.findall(r'\b\w+\b', text.lower())
    
    def fit(self, docs):
        self.total_doc = len(docs)
        for label, text in docs:
            words = [w for w in self.tokenize(text) if w in self.vocab]
            self.word_counter[label].update(words)
            self.cnt[label] += 1

    def predict(self, text) :
        words = [w for w in self.tokenize(text) if w in self.vocab]
        scores = {}
        for label in self.cnt:
            log_prob = math.log(self.cnt[label] / self.total_doc)
            total = sum(self.word_counter[label].values())
            for word in words :
                word_count = self.word_counter[label][word]
                prob = (word_count + 1) / (total + len(self.vocab))
                log_prob += math.log(prob)

            scores[label] = log_prob

        return max(scores, key=scores.get)
    


def build_vocab_from_docs(docs):
    vocab = set()
    for _, text in docs:
        words = re.findall(r'\b\w+\b', text.lower())
        vocab.update(words)
    return vocab



df = pd.read_csv("spam_mail_classifier.csv")
label = df["label"].values
email = df["email_text"].values
docs = list(zip(label, email))

nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import wordnet as wn
fixed_vocab = set(lemma.name().lower() for synset in wn.all_synsets() for lemma in synset.lemmas())
vocab_from_data = build_vocab_from_docs(docs)
fixed_vocab = {word for word in fixed_vocab if word.isalpha()}
fixed_vocab = fixed_vocab.union(vocab_from_data)


nb = NaiveBayes(fixed_vocab)
nb.fit(docs)

print(nb.predict("let's meet for lunch"))
