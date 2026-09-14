import re
import math
from collections import Counter, defaultdict 
import numpy as np
#from part1_models.tokenizer import tokenize


#step 1: Tokenization function defined
TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|[.,!?;:]")

def tokenize(text):
    '''Convert a string to a lowercase sequence of word/punctuation tokens.'''
    normalized = text.lower().replace("’", "'")
    return TOKEN_PATTERN.findall(normalized)
#step 2: Add start/end symbols
START = "<s>"
END = "</s>"
def with_boundaries(sentence, n=2):
    '''Tokenize a sentence and add start/end symbols for an order-n model.'''
    tokens = tokenize(sentence)
    return [START] * (n - 1) + tokens + [END]

#step 3: Build n-gram counts
def build_ngram_counts(sentences, n):
    next_counts = defaultdict(Counter)
    history_counts = Counter()
    vocab = set()
    
    for sentence in sentences:
        tokens = with_boundaries(sentence, n=n)
        vocab.update(tokens)  # Collect all tokens in the sentence
        
        for position in range(n - 1, len(tokens)):
            history = tuple(tokens[position - n + 1 : position])
            word = tokens[position]
            next_counts[history][word] += 1
            history_counts[history] += 1
            
    return next_counts, history_counts, vocab


#step 4: Define the CountLanguageModel class
class CountLanguageModel:
    def __init__(self, max_order):
        self.max_order = max_order
        self.next_counts = {}
        self.history_counts = {}
        self.vocabulary = set()

    def fit(self, sentences):
        for order in range(1, self.max_order + 1):
            next_counts, history_counts, vocab = build_ngram_counts(sentences, order)
            self.next_counts[order] = next_counts
            self.history_counts[order] = history_counts
            self.vocabulary.update(vocab)
        return self

    def history_for(self, history_tokens, order):
        if order == 1:
            return ()
        padded = [START] * max(0, order - 1 - len(history_tokens)) + list(history_tokens)
        return tuple(padded[-(order - 1):])

    def probability(self, history_tokens, word, order, add_k=0.0):
      history = self.history_for(history_tokens, order)
    
      # Unsmoothed N-gram logic (add_k == 0)
      if add_k == 0.0:
        history_count = self.history_counts[order][history]
        if history_count == 0:
            return 0.0
        return self.next_counts[order][history][word] / history_count

    # Add-k Smoothed N-gram logic (add_k > 0)
      numerator = self.next_counts[order][history][word] + add_k
      denominator = self.history_counts[order][history] + (add_k * len(self.vocabulary))
      return numerator / denominator if denominator > 0 else 0.0

    def interpolated_probability(self, history_tokens, word, weights, add_k=0.0):
    prob = 0.0
    for order, w in weights.items():
        prob += w * self.probability(history_tokens, word, order, add_k=add_k)
    return prob

#step 5: function for evaluating perplexity and cross-entropy on a given dataset  
def evaluate_model(model, sentences, weights, add_k):
    """Calculates cross-entropy loss and perplexity on a given dataset."""
    log_prob_sum = 0.0
    token_count = 0
    
    for sentence in sentences:
        tokens = with_boundaries(sentence, n=model.max_order)
        for position in range(model.max_order - 1, len(tokens)):
            history_tokens = tokens[position - model.max_order + 1 : position]
            word = tokens[position]
            
            prob = model.interpolated_probability(
                history_tokens, word, weights=weights, add_k=add_k
            )
            
            # Guard against zero probability using a small epsilon fallback
            prob = max(prob, 1e-12)
            log_prob_sum += math.log2(prob)
            token_count += 1

    if token_count == 0:
        return float('inf'), float('inf')

    avg_cross_entropy = -log_prob_sum / token_count
    perplexity = 2 ** avg_cross_entropy
    return avg_cross_entropy, perplexity


# step 7: Load Datasets
with open("data/processed/train.txt", "r", encoding="utf-8") as f:
    train_sentences = f.readlines()
with open("data/processed/dev.txt", "r", encoding="utf-8") as f:
    dev_sentences = f.readlines()
with open("data/processed/test.txt", "r", encoding="utf-8") as f:
    test_sentences = f.readlines()

# step 8: Fit Model Parameters ONLY on Training Data
max_order = 3
model = CountLanguageModel(max_order=max_order)
model.fit(train_sentences)

# Equal weights for interpolation across orders 1..N
weights = {order: 1.0 / max_order for order in range(1, max_order + 1)}

# step 9: Hyperparameter Tuning on Validation (Dev) Set
candidate_k_values = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
best_k = None
best_dev_pp = float('inf')

print("--- Tuning Hyperparameters on Dev Set ---")
for k in candidate_k_values:
    _, dev_pp = evaluate_model(model, dev_sentences, weights=weights, add_k=k)
    print(f"add_k = {k:<5} | Dev Perplexity: {dev_pp:.4f}")
    
    if dev_pp < best_dev_pp:
        best_dev_pp = dev_pp
        best_k = k

print(f"\nBest add_k found: {best_k} (Dev Perplexity: {best_dev_pp:.4f})")

# step 10: Final Evaluation on Test Set using Best Hyperparameters
test_ce, test_pp = evaluate_model(model, test_sentences, weights=weights, add_k=best_k)

print("\n--- Final Test Set Results ---")
print(f"Test Cross-Entropy: {test_ce:.4f} bits/token")
print(f"Test Perplexity:    {test_pp:.4f}")