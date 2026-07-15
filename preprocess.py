"""
Preprocessing for n-gram smoothing analysis.
- Tokenizes Tiny Shakespeare into sentences
- Handles OOV via <UNK> (words seen once in train are folded into <UNK>)
- Splits train/test
- Builds unigram/bigram/trigram counts
"""
import re, random, pickle, os, urllib.request
from collections import Counter, defaultdict

random.seed(42)

# Put your downloaded corpus here
RAW_PATH = "shakespeare.txt"
CORPUS_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

def ensure_corpus(path):
    if not os.path.exists(path):
        print(f"Downloading corpus to {path} ...")
        urllib.request.urlretrieve(CORPUS_URL, path)

def load_sentences(path):
    text = open(path, encoding="utf-8").read().lower()
    # Split on sentence-ending punctuation OR blank lines (scene/speaker breaks)
    chunks = re.split(r'[.!?\n]+', text)
    sentences = []
    for c in chunks:
        toks = re.findall(r"[a-z']+", c)
        if len(toks) >= 2:  # skip trivial/empty
            sentences.append(toks)
    return sentences

def apply_unk(sentences, min_count=2):
    """Words occurring < min_count times in TRAIN are replaced with <UNK>."""
    freq = Counter(w for s in sentences for w in s)
    vocab = {w for w, c in freq.items() if c >= min_count}
    def replace(s):
        return [w if w in vocab else "<UNK>" for w in s]
    return [replace(s) for s in sentences], vocab

def add_boundaries(sentences, n=3):
    """Add <s> start pads (n-1 of them) and one </s> end pad per sentence."""
    out = []
    pad = ["<s>"] * (n - 1)
    for s in sentences:
        out.append(pad + s + ["</s>"])
    return out

def build_counts(sentences):
    unigrams, bigrams, trigrams = Counter(), Counter(), Counter()
    for s in sentences:
        for w in s:
            unigrams[w] += 1
        for i in range(len(s) - 1):
            bigrams[(s[i], s[i+1])] += 1
        for i in range(len(s) - 2):
            trigrams[(s[i], s[i+1], s[i+2])] += 1
    return unigrams, bigrams, trigrams

if __name__ == "__main__":
    ensure_corpus(RAW_PATH)
    sentences = load_sentences(RAW_PATH)
    print(f"Total sentences (raw): {len(sentences)}")

    random.shuffle(sentences)
    n_test = int(0.1 * len(sentences))
    n_dev = int(0.1 * len(sentences))
    test_raw = sentences[:n_test]
    dev_raw = sentences[n_test:n_test + n_dev]
    train_raw = sentences[n_test + n_dev:]
    print(f"Train sentences: {len(train_raw)}, Dev sentences: {len(dev_raw)}, Test sentences: {len(test_raw)}")

    train_unk, vocab = apply_unk(train_raw, min_count=2)
    # </s> is a predictable token (ends every sentence); <s> is context-only, never predicted
    vocab = vocab | {"<UNK>", "</s>"}
    print(f"Vocabulary size (train, min_count=2, incl. </s>,<UNK>): {len(vocab)}")

    # apply same vocab to dev/test (OOV -> <UNK>)
    dev_unk = [[w if w in vocab else "<UNK>" for w in s] for s in dev_raw]
    test_unk = [[w if w in vocab else "<UNK>" for w in s] for s in test_raw]

    train_padded = add_boundaries(train_unk, n=3)
    dev_padded = add_boundaries(dev_unk, n=3)
    test_padded = add_boundaries(test_unk, n=3)

    uni, bi, tri = build_counts(train_padded)
    print(f"Unique unigrams: {len(uni)}  Unique bigrams: {len(bi)}  Unique trigrams: {len(tri)}")
    print(f"Total train tokens (incl. boundaries): {sum(uni.values())}")

    with open("data.pkl", "wb") as f:
        pickle.dump({
            "train_padded": train_padded,
            "dev_padded": dev_padded,
            "test_padded": test_padded,
            "vocab": vocab,
            "unigrams": uni,
            "bigrams": bi,
            "trigrams": tri,
        }, f)
    print("Saved data.pkl")