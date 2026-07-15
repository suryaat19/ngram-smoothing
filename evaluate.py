"""
1. Tunes hyperparameters (k, m, lambda1, context-dependent k) on the DEV split.
2. Evaluates final perplexity of every method on the held-out TEST split.
3. Computes "probability mass reserved for unseen continuations" per method.
4. Saves everything needed for plotting to results.pkl
"""
import math
import pickle
import numpy as np
from smoothing import (
    LMData, p_mle, p_laplace, p_addk, p_unigram_prior, p_good_turing,
    p_backoff_katz, p_linear_interp, p_context_dependent_interp, p_kneser_ney,
    p_witten_bell, witten_bell_lambda, p_modified_kneser_ney,
    p_stupid_backoff_raw, p_stupid_backoff_norm, stupid_backoff_Z,
    p_absolute_disc_mle_fallback,
    METHODS,
)

d = LMData()
print(f"V={d.V}  N={d.N}  distinct bigrams={len(d.bi)}  KN discount={d.kn_discount:.3f}")
print(f"Good-Turing fit: log(N_c) = {d.gt_fit[0]:.3f} + {d.gt_fit[1]:.3f}*log(c)")
print(f"Good-Turing unseen mass (global): {d.gt_p0_mass:.5f}")


def bigram_pairs(padded_sentences):
    """Drop the redundant extra <s> (sentences are triple-padded for trigram use);
    keep exactly one <s> as bigram start context, then yield consecutive pairs."""
    for s in padded_sentences:
        seq = s[1:]  # one <s> remains as start-of-sentence context
        for i in range(len(seq) - 1):
            yield seq[i], seq[i + 1]


def perplexity(prob_fn, pairs, eps=1e-12):
    logs = []
    zero_count = 0
    for w1, w2 in pairs:
        p = prob_fn(w1, w2)
        if p <= 0:
            zero_count += 1
            p = eps
        logs.append(math.log(p))
    M = len(logs)
    pp = math.exp(-sum(logs) / M)
    return pp, zero_count, M


dev_pairs = list(bigram_pairs(d.dev_padded))
test_pairs = list(bigram_pairs(d.test_padded))
print(f"Dev bigram predictions: {len(dev_pairs)}   Test bigram predictions: {len(test_pairs)}")

# Hyperparameter tuning on DEV
print("\nTuning Add-k")
k_grid = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
addk_curve = []
best_k, best_pp = None, float("inf")
for k in k_grid:
    pp, _, _ = perplexity(lambda w1, w2, k=k: p_addk(d, w1, w2, k), dev_pairs)
    addk_curve.append((k, pp))
    print(f"  k={k:<8} dev-PP={pp:.2f}")
    if pp < best_pp:
        best_pp, best_k = pp, k
d.best_k = best_k
print(f"  -> best k = {best_k}")

print("\nTuning Unigram Prior (m)")
m_grid = [1, 5, 25, 100, 500, 2000, d.V]
best_m, best_pp = None, float("inf")
for m in m_grid:
    pp, _, _ = perplexity(lambda w1, w2, m=m: p_unigram_prior(d, w1, w2, m), dev_pairs)
    print(f"  m={m:<8} dev-PP={pp:.2f}")
    if pp < best_pp:
        best_pp, best_m = pp, m
d.best_m = best_m
print(f"  -> best m = {best_m}")

print("\nTuning Linear Interpolation (lambda1)")
lam_grid = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
best_lam1, best_pp = None, float("inf")
for lam1 in lam_grid:
    pp, _, _ = perplexity(lambda w1, w2, lam1=lam1: p_linear_interp(d, w1, w2, lam1), dev_pairs)
    print(f"  lambda1={lam1:<6} dev-PP={pp:.2f}")
    if pp < best_pp:
        best_pp, best_lam1 = pp, lam1
d.best_lam1 = best_lam1
print(f"  -> best lambda1 = {best_lam1}")

print("\nTuning Context-Dependent Interpolation (k)")
cdk_grid = [1, 5, 20, 50, 100, 300, 1000]
best_cd_k, best_pp = None, float("inf")
for k in cdk_grid:
    pp, _, _ = perplexity(lambda w1, w2, k=k: p_context_dependent_interp(d, w1, w2, k), dev_pairs)
    print(f"  k={k:<6} dev-PP={pp:.2f}")
    if pp < best_pp:
        best_pp, best_cd_k = pp, k
d.best_cd_k = best_cd_k
print(f"  -> best k = {best_cd_k}")

# Final evaluation on TEST
print("\nFinal TEST perplexities")
results = {}
for name, fn in METHODS.items():
    pp, zeros, M = perplexity(lambda w1, w2, fn=fn: fn(d, w1, w2), test_pairs)
    results[name] = {"perplexity": pp, "zero_prob_events": zeros, "total_events": M}
    tag = "  <-- undefined/informative: some events have P=0" if zeros > 0 and name == "MLE" else ""
    print(f"  {name:<22} PP={pp:>12.2f}   zero-prob test events={zeros}/{M}{tag}")

# Probability mass reserved for UNSEEN continuations
print("\nProbability mass reserved for unseen continuations (avg over sample contexts) ")
freq_sorted = sorted(d.uni.items(), key=lambda x: -x[1])
sample_contexts = [w for w, c in freq_sorted if w not in ("<s>", "</s>", "<UNK>")][:12]
mass_reserved = {name: [] for name in METHODS}
for w1 in sample_contexts:
    unseen_words = [w for w in d.vocab if w not in d.followers[w1]]
    for name, fn in METHODS.items():
        total = sum(fn(d, w1, w2) for w2 in unseen_words)
        mass_reserved[name].append(total)
mass_reserved_avg = {name: float(np.mean(v)) for name, v in mass_reserved.items()}
for name, v in mass_reserved_avg.items():
    print(f"  {name:<22} avg unseen mass = {v:.4f}")

# Example bigram probability comparison
SPECIAL = {"<s>", "</s>", "<UNK>"}
real_bigrams = [(bg, c) for bg, c in d.bi.items() if bg[0] not in SPECIAL and bg[1] not in SPECIAL]
top_bigram = max(real_bigrams, key=lambda x: x[1])[0]
singleton_candidates = [bg for bg, c in real_bigrams if c == 1]
singleton_bigram = singleton_candidates[len(singleton_candidates) // 2]
w1_freq = top_bigram[0]
unseen_candidates = [w for w, c in freq_sorted[:200]
                      if w not in d.followers[w1_freq] and w != w1_freq and w not in SPECIAL]
unseen_bigram = (w1_freq, unseen_candidates[0])

example_bigrams = {
    f"frequent: {top_bigram}": top_bigram,
    f"seen-once: {singleton_bigram}": singleton_bigram,
    f"unseen: {unseen_bigram}": unseen_bigram,
}
print("\nExample bigram probabilities across methods")
example_probs = {}
for label, (w1, w2) in example_bigrams.items():
    example_probs[label] = {name: fn(d, w1, w2) for name, fn in METHODS.items()}
    print(f"  {label}  C={d.c_bi(w1,w2)}  C({w1})={d.c_uni(w1)}")
    for name, p in example_probs[label].items():
        print(f"      {name:<22} {p:.3e}")

# Sparsity growth: unigram vs bigram vs trigram
print("\nSparsity: fraction of TEST n-grams unseen in TRAIN (by order) ")
with open("data.pkl", "rb") as f:
    raw = pickle.load(f)
tri = raw["trigrams"]


def trigram_pairs(padded_sentences):
    for s in padded_sentences:
        for i in range(len(s) - 2):
            yield s[i], s[i + 1], s[i + 2]


test_uni_tokens = [w for s in d.test_padded for w in s[2:]]  # drop <s><s>
uni_unseen = sum(1 for w in test_uni_tokens if d.c_uni(w) == 0) / len(test_uni_tokens)
bi_unseen = sum(1 for w1, w2 in test_pairs if d.c_bi(w1, w2) == 0) / len(test_pairs)
test_tri = list(trigram_pairs(d.test_padded))
tri_unseen = sum(1 for w1, w2, w3 in test_tri if tri.get((w1, w2, w3), 0) == 0) / len(test_tri)
sparsity = {"unigram": uni_unseen, "bigram": bi_unseen, "trigram": tri_unseen}
print(f"  unigram unseen rate: {uni_unseen:.4f}")
print(f"  bigram  unseen rate: {bi_unseen:.4f}")
print(f"  trigram unseen rate: {tri_unseen:.4f}")


# KN Ablation: continuation prob. vs plain MLE fallback
print("\nKN Ablation: Kneser-Ney (continuation) vs Absolute Disc. (MLE fallback)")
kn_pp = results["Kneser-Ney"]["perplexity"]
abl_pp = results["Abs. Disc. (MLE fallback)"]["perplexity"]
print(f"  Kneser-Ney (P_cont):        test PP = {kn_pp:.2f}")
print(f"  Abs. Disc. (P_uni fallback): test PP = {abl_pp:.2f}")
print(f"  Improvement from continuation probability alone: {abl_pp - kn_pp:.2f} PP "
      f"({100*(abl_pp-kn_pp)/abl_pp:.1f}% relative)")

# find a "San Francisco"-style word: high raw unigram frequency but low
# continuation probability (i.e. it mostly follows just one or two contexts)
candidates = [w for w in d.vocab if w not in ("<s>", "</s>", "<UNK>") and d.uni[w] >= 15]
ratios = [(w, d.p_uni(w) / d.p_cont(w)) for w in candidates]
ratios.sort(key=lambda x: -x[1])
sf_word, sf_ratio = ratios[0]
sf_example = {
    "word": sf_word,
    "p_uni": d.p_uni(sf_word),
    "p_cont": d.p_cont(sf_word),
    "n_distinct_predecessors": len(d.preceders[sf_word]),
    "raw_count": d.uni[sf_word],
}
print(f"  Example word with P_uni >> P_cont: '{sf_word}'  "
      f"P_uni={sf_example['p_uni']:.5f}  P_cont={sf_example['p_cont']:.5f}  "
      f"(seen after only {sf_example['n_distinct_predecessors']} distinct word(s), "
      f"despite occurring {sf_example['raw_count']} times total)")

# Witten-Bell: data-driven lambda vs tuned Context-Dependent Interp
print("\nWitten-Bell lambda(w1) vs Context-Dependent Interp. lambda(w1)")
wb_sample = [w for w, c in freq_sorted if w not in SPECIAL and c >= 1]
wb_curve = []
for w1 in wb_sample:
    c1 = d.c_uni(w1)
    lam_wb = witten_bell_lambda(d, w1)
    lam_cdi = c1 / (c1 + d.best_cd_k)
    wb_curve.append((c1, lam_wb, lam_cdi))
print(f"  Computed lambda for {len(wb_curve)} contexts "
      f"(context counts ranging {min(c[0] for c in wb_curve)}-{max(c[0] for c in wb_curve)})")

# Stupid Backoff: raw score does NOT sum to 1
print("\nStupid Backoff: raw (unnormalized) score oversums probability mass")
sb_Z = {w1: stupid_backoff_Z(d, w1, alpha=0.4) for w1 in sample_contexts}
for w1, z in sb_Z.items():
    print(f"  Z({w1!r}) = {z:.4f}  (raw scores sum to {z:.3f}, not 1.0)")
print(f"  Mean Z over sample contexts: {np.mean(list(sb_Z.values())):.4f}")


# Save everything for plotting 
with open("results.pkl", "wb") as f:
    pickle.dump({
        "results": results,
        "mass_reserved_avg": mass_reserved_avg,
        "example_probs": example_probs,
        "example_bigrams": example_bigrams,
        "addk_curve": addk_curve,
        "sparsity": sparsity,
        "best_k": best_k, "best_m": best_m, "best_lam1": best_lam1, "best_cd_k": best_cd_k,
        "kn_discount": d.kn_discount,
        "mkn_discounts": d.mkn_discounts,
        "gt_fit": d.gt_fit,
        "Nc_raw": d.Nc_raw,
        "V": d.V, "N": d.N,
        "sf_example": sf_example,
        "wb_curve": wb_curve,
        "sb_Z": sb_Z,
    }, f)
print("\nSaved results.pkl")