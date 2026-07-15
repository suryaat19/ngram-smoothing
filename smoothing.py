"""
Implementations of n-gram smoothing techniques, operating at the BIGRAM level
P(w_i | w_{i-1}), so all eight methods can be compared on the
same probability estimation task.

Every formula here matches the formulas derived analytically beforehand:
  1. MLE (zero-probability baseline)
  2. Laplace (add-1)
  3. Add-k
  4. Unigram Prior Smoothing
  5. Good-Turing
  6. Backoff (Katz, using Good-Turing discounted counts)
  7. Linear Interpolation
  8. Context-Dependent Interpolation
  9. Kneser-Ney (Absolute Discounting + Continuation Probability)
  10. Witten-Bell Smoothing
  11. Modified Kneser-Ney (three discounts D1, D2, D3+)
  12. Stupid Backoff
  13. Absolute Discounting w/ MLE fallback (KN ablation -- isolates the
      contribution of continuation probability vs. plain discounting)
"""
import pickle
import math
import numpy as np
from collections import defaultdict


class LMData:
    """Precomputes everything the smoothing formulas need from raw counts."""

    def __init__(self, pkl_path="data.pkl"):
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)
        self.train_padded = d["train_padded"]
        self.dev_padded = d["dev_padded"]
        self.test_padded = d["test_padded"]
        self.vocab = d["vocab"]                 # predictable word types (excludes <s>)
        self.uni = d["unigrams"]                 # Counter, includes <s> (context-only)
        self.bi = d["bigrams"]                   # Counter[(w1,w2)]
        self.V = len(self.vocab)

        # N = total predictable tokens (excludes <s>, which is never generated)
        self.N = sum(c for w, c in self.uni.items() if w != "<s>")

        # distinct continuations of each context: followers(w1) = {w2 : C(w1,w2) > 0}
        self.followers = defaultdict(set)
        # distinct predecessors of each word: preceders(w2) = {w1 : C(w1,w2) > 0}
        self.preceders = defaultdict(set)
        for (w1, w2) in self.bi:
            self.followers[w1].add(w2)
            self.preceders[w2].add(w1)

        self.total_bigram_types = len(self.bi)                 # for continuation prob denominator
        self.total_bigram_tokens = sum(self.bi.values())        # for Good-Turing unseen mass

        # unigram MLE, P(w) = C(w)/N   (excludes <s> from being "generated")
        self._p_uni_cache = {w: c / self.N for w, c in self.uni.items() if w != "<s>"}

        # Good-Turing frequency-of-frequencies over BIGRAM counts
        self._fit_good_turing()

        # Kneser-Ney discount
        n1 = sum(1 for c in self.bi.values() if c == 1)
        n2 = sum(1 for c in self.bi.values() if c == 2)
        self.kn_discount = n1 / (n1 + 2 * n2) if (n1 + 2 * n2) > 0 else 0.75
        self.kn_discount = min(max(self.kn_discount, 0.1), 0.9)  # sane clip

        # continuation probability P_cont(w2) = |preceders(w2)| / total_bigram_types
        self._p_cont_cache = {
            w: len(self.preceders[w]) / self.total_bigram_types for w in self.vocab
        }
        
        # Modified Kneser-Ney: three discounts D1, D2, D3+ (Chen & Goodman)
        n3 = sum(1 for c in self.bi.values() if c == 3)
        n4 = sum(1 for c in self.bi.values() if c == 4)
        Y = n1 / (n1 + 2 * n2) if (n1 + 2 * n2) > 0 else 0.75
        D1 = 1 - 2 * Y * (n2 / n1) if n1 > 0 else 0.5
        D2 = 2 - 3 * Y * (n3 / n2) if n2 > 0 else 1.0
        D3p = 3 - 4 * Y * (n4 / n3) if n3 > 0 else 1.5
        self.mkn_discounts = (
            min(max(D1, 0.01), 0.99),
            min(max(D2, 0.01), 1.99),
            min(max(D3p, 0.01), 2.99),
        )
        # per-context counts of how many distinct followers fall in each count bucket
        self._ctx_n1 = defaultdict(int)
        self._ctx_n2 = defaultdict(int)
        self._ctx_n3p = defaultdict(int)
        for (w1, w2), c in self.bi.items():
            if c == 1:
                self._ctx_n1[w1] += 1
            elif c == 2:
                self._ctx_n2[w1] += 1
            else:
                self._ctx_n3p[w1] += 1
                
    def p_uni(self, w):
        return self._p_uni_cache.get(w, 1e-12)

    def p_cont(self, w):
        return self._p_cont_cache.get(w, 1e-12)

    def c_bi(self, w1, w2):
        return self.bi.get((w1, w2), 0)

    def c_uni(self, w1):
        return self.uni.get(w1, 0)

    # Good-Turing fitting
    def _fit_good_turing(self, kmax=8):
        """Frequency-of-frequencies N_c for bigram counts c=1..kmax+1,
        smoothed via log-log linear regression (Simple Good-Turing) to
        avoid the noisy/zero N_c at higher counts."""
        counts = np.array(list(self.bi.values()))
        Nc_raw = {c: int(np.sum(counts == c)) for c in range(1, kmax + 2)}
        self.Nc_raw = Nc_raw

        cs = np.array([c for c, n in Nc_raw.items() if n > 0], dtype=float)
        ns = np.array([n for c, n in Nc_raw.items() if n > 0], dtype=float)
        log_c, log_n = np.log(cs), np.log(ns)
        b, a = np.polyfit(log_c, log_n, 1)  # log(N_c) = a + b*log(c)
        self.gt_fit = (a, b)

        def Nc_smooth(c):
            return math.exp(a + b * math.log(c)) if c > 0 else Nc_raw.get(0, 0)

        self.Nc_smooth = Nc_smooth
        # adjusted (discounted) counts c* = (c+1) * S(c+1)/S(c), c = 1..kmax
        self.gt_cstar = {c: (c + 1) * Nc_smooth(c + 1) / Nc_smooth(c) for c in range(1, kmax + 1)}
        # total probability mass reserved for UNSEEN bigrams (given any context), globally
        N1 = Nc_raw.get(1, 0)
        self.gt_p0_mass = N1 / self.total_bigram_tokens


# 1. MLE  (baseline -- demonstrates the zero-probability problem)

def p_mle(d: LMData, w1, w2):
    c1 = d.c_uni(w1)
    if c1 == 0:
        return 0.0
    return d.c_bi(w1, w2) / c1


# 2. Laplace (add-1)
#    P(w2|w1) = (C(w1,w2) + 1) / (C(w1) + V)

def p_laplace(d: LMData, w1, w2):
    return (d.c_bi(w1, w2) + 1) / (d.c_uni(w1) + d.V)


# 3. Add-k
#    P(w2|w1) = (C(w1,w2) + k) / (C(w1) + k*V)

def p_addk(d: LMData, w1, w2, k=1.0):
    return (d.c_bi(w1, w2) + k) / (d.c_uni(w1) + k * d.V)


# 4. Unigram Prior Smoothing
#    P(w2|w1) = (C(w1,w2) + m*P(w2)) / (C(w1) + m)

def p_unigram_prior(d: LMData, w1, w2, m=1.0):
    return (d.c_bi(w1, w2) + m * d.p_uni(w2)) / (d.c_uni(w1) + m)


# 5. Good-Turing
#    Seen:   P = c* / C(w1)            (c* from GT regression)
#    Unseen: reserved global mass N1/N_bigramtokens, spread uniformly
#            across the unseen continuations of w1 (context-agnostic
#            redistribution -- this is exactly the limitation Katz
#            backoff fixes by using the unigram distribution instead).

def p_good_turing(d: LMData, w1, w2):
    c = d.c_bi(w1, w2)
    c1 = d.c_uni(w1)
    if c1 == 0:
        return 0.0
    if c > 0:
        cstar = d.gt_cstar.get(c, c)  # fall back to raw count if beyond fitted range
        return cstar / c1
    else:
        n_unseen_here = d.V - len(d.followers[w1])
        if n_unseen_here <= 0:
            return 1e-12
        return d.gt_p0_mass / n_unseen_here
 
# Shared helper: per-context unigram mass split into "mass covered by
# seen continuations" vs "mass left over for unseen continuations".
# Used by both Katz Backoff and Stupid Backoff (normalized variant).

def _context_uni_mass(d: LMData, w1, _cache={}):
    if w1 in _cache:
        return _cache[w1]
    seen_uni_mass = sum(d.p_uni(w2) for w2 in d.followers[w1])
    unseen_uni_mass = max(1.0 - seen_uni_mass, 1e-12)
    _cache[w1] = (seen_uni_mass, unseen_uni_mass)
    return seen_uni_mass, unseen_uni_mass
   

# 6. Backoff (Katz-style, using Good-Turing discounted counts)
#    Seen:   P = c*(w1,w2) / C(w1)
#    Unseen: P = alpha(w1) * P(w2) / sum_{w' unseen} P(w')

def _katz_context_stats(d: LMData, w1, _cache={}):
    if w1 in _cache:
        return _cache[w1]
    c1 = d.c_uni(w1)
    seen_mass = 0.0
    seen_uni_mass = 0.0
    for w2 in d.followers[w1]:
        c = d.c_bi(w1, w2)
        cstar = d.gt_cstar.get(c, c)
        seen_mass += cstar / c1
        seen_uni_mass += d.p_uni(w2)
    alpha = max(1.0 - seen_mass, 0.0)
    unseen_uni_mass = max(1.0 - seen_uni_mass, 1e-12)
    _cache[w1] = (alpha, unseen_uni_mass)
    return alpha, unseen_uni_mass


def p_backoff_katz(d: LMData, w1, w2):
    c1 = d.c_uni(w1)
    if c1 == 0:
        return 0.0
    c = d.c_bi(w1, w2)
    if c > 0:
        cstar = d.gt_cstar.get(c, c)
        return cstar / c1
    alpha, unseen_uni_mass = _katz_context_stats(d, w1)
    return alpha * d.p_uni(w2) / unseen_uni_mass


# 7. Linear Interpolation (fixed lambdas)
#    P(w2|w1) = lam1 * P_MLE(w2|w1) + lam2 * P(w2)

def p_linear_interp(d: LMData, w1, w2, lam1=0.7):
    lam2 = 1 - lam1
    return lam1 * p_mle(d, w1, w2) + lam2 * d.p_uni(w2)


# 8. Context-Dependent Interpolation
#    lambda1(w1) = C(w1) / (C(w1) + k)   -- trust bigram MLE more
#    when the context itself is well attested.

def p_context_dependent_interp(d: LMData, w1, w2, k=50.0):
    c1 = d.c_uni(w1)
    lam1 = c1 / (c1 + k)
    lam2 = 1 - lam1
    return lam1 * p_mle(d, w1, w2) + lam2 * d.p_uni(w2)


# 9. Kneser-Ney (Absolute Discounting Interpolation + Continuation Prob.)
#    P(w2|w1) = max(C(w1,w2)-d,0)/C(w1) + lambda(w1) * P_cont(w2)
#    lambda(w1) = (d/C(w1)) * |followers(w1)|

def p_kneser_ney(d: LMData, w1, w2):
    c1 = d.c_uni(w1)
    if c1 == 0:
        return 0.0
    disc = d.kn_discount
    c = d.c_bi(w1, w2)
    discounted_term = max(c - disc, 0.0) / c1
    lam = (disc / c1) * len(d.followers[w1])
    return discounted_term + lam * d.p_cont(w2)

# 10. Witten-Bell Smoothing
#     lambda(w1) = C(w1) / (C(w1) + T(w1)),  T(w1) = # distinct followers of w1
#     P(w2|w1) = lambda(w1)*P_MLE(w2|w1) + (1-lambda(w1))*P(w2)
#     -- the interpolation weight is DERIVED from data, no tuning needed.
#     Contrast with method 8 (Context-Dependent Interp.), which used the
#     same *shape* of formula but a hand-tuned constant k in place of T(w1).

def witten_bell_lambda(d: LMData, w1):
    c1 = d.c_uni(w1)
    T = len(d.followers[w1])
    return c1 / (c1 + T) if (c1 + T) > 0 else 0.0


def p_witten_bell(d: LMData, w1, w2):
    c1 = d.c_uni(w1)
    if c1 == 0:
        return 0.0
    lam1 = witten_bell_lambda(d, w1)
    return lam1 * p_mle(d, w1, w2) + (1 - lam1) * d.p_uni(w2)


# 11. Modified Kneser-Ney (Chen & Goodman, 1999)
#     Three separate discounts D1, D2, D3+ instead of one global d,
#     applied depending on the bigram's raw count (1, 2, or 3+).
#     This is what SRILM / KenLM use by default.

def p_modified_kneser_ney(d: LMData, w1, w2):
    c1 = d.c_uni(w1)
    if c1 == 0:
        return 0.0
    D1, D2, D3p = d.mkn_discounts
    c = d.c_bi(w1, w2)
    if c == 0:
        D = 0.0
    elif c == 1:
        D = D1
    elif c == 2:
        D = D2
    else:
        D = D3p
    discounted_term = max(c - D, 0.0) / c1
    lam = (D1 * d._ctx_n1[w1] + D2 * d._ctx_n2[w1] + D3p * d._ctx_n3p[w1]) / c1
    return discounted_term + lam * d.p_cont(w2)


# 12. Stupid Backoff (Brants et al., 2007) -- NOT a true probability
#     distribution: no discounting, fixed backoff weight alpha=0.4,
#     designed to be cheap at web scale rather than well-calibrated.
#     Seen:   S = C(w1,w2)/C(w1)      (this alone already sums to 1.0!)
#     Unseen: S = alpha * P(w2)       (extra, un-normalized mass on top)
#     We provide both the raw (literature-faithful, sums to >1) score and
#     a normalized version -- dividing by Z(w1) = 1 + alpha*unseen_uni_mass
#     -- purely so perplexity is comparable to the other methods here.

def p_stupid_backoff_raw(d: LMData, w1, w2, alpha=0.4):
    c1 = d.c_uni(w1)
    if c1 == 0:
        return 0.0
    c = d.c_bi(w1, w2)
    if c > 0:
        return c / c1
    return alpha * d.p_uni(w2)

def stupid_backoff_Z(d: LMData, w1, alpha=0.4):
    """Normalizer: seen mass is always exactly 1.0 (raw bigram MLE already
    sums to 1), so Z(w1) = 1 + alpha * (leftover unigram mass)."""
    _, unseen_uni_mass = _context_uni_mass(d, w1)
    return 1.0 + alpha * unseen_uni_mass


def p_stupid_backoff_norm(d: LMData, w1, w2, alpha=0.4):
    c1 = d.c_uni(w1)
    if c1 == 0:
        return 0.0
    Z = stupid_backoff_Z(d, w1, alpha)
    return p_stupid_backoff_raw(d, w1, w2, alpha) / Z


# 13. KN Ablation: Absolute Discounting with plain MLE fallback
#     Identical to Kneser-Ney EXCEPT the low-order term uses raw
#     unigram probability P(w2) instead of continuation probability
#     P_cont(w2). Comparing this against method 9 isolates exactly how
#     much of Kneser-Ney's advantage comes from continuation probability
#     specifically, versus from absolute discounting alone.

def p_absolute_disc_mle_fallback(d: LMData, w1, w2):
    c1 = d.c_uni(w1)
    if c1 == 0:
        return 0.0
    disc = d.kn_discount
    c = d.c_bi(w1, w2)
    discounted_term = max(c - disc, 0.0) / c1
    lam = (disc / c1) * len(d.followers[w1])
    return discounted_term + lam * d.p_uni(w2)   # <-- p_uni, not p_cont


METHODS = {
    "MLE": lambda d, w1, w2: p_mle(d, w1, w2),
    "Laplace": lambda d, w1, w2: p_laplace(d, w1, w2),
    "Add-k": lambda d, w1, w2: p_addk(d, w1, w2, k=d.best_k),
    "Unigram Prior": lambda d, w1, w2: p_unigram_prior(d, w1, w2, m=d.best_m),
    "Good-Turing": lambda d, w1, w2: p_good_turing(d, w1, w2),
    "Backoff (Katz)": lambda d, w1, w2: p_backoff_katz(d, w1, w2),
    "Linear Interp.": lambda d, w1, w2: p_linear_interp(d, w1, w2, lam1=d.best_lam1),
    "Context-Dep. Interp.": lambda d, w1, w2: p_context_dependent_interp(d, w1, w2, k=d.best_cd_k),
    "Witten-Bell": lambda d, w1, w2: p_witten_bell(d, w1, w2),
    "Stupid Backoff": lambda d, w1, w2: p_stupid_backoff_norm(d, w1, w2, alpha=0.4),
    "Abs. Disc. (MLE fallback)": lambda d, w1, w2: p_absolute_disc_mle_fallback(d, w1, w2),
    "Kneser-Ney": lambda d, w1, w2: p_kneser_ney(d, w1, w2),
    "Modified Kneser-Ney": lambda d, w1, w2: p_modified_kneser_ney(d, w1, w2),
}
