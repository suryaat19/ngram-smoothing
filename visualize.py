"""
visualize.py

Generates 10 plots visualizing the results of the N-gram smoothing analysis.
Reads metrics from results.pkl and saves PNGs to the 'plots' directory.
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from smoothing import LMData

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
})
OUT = "plots"

d = LMData()
with open("results.pkl", "rb") as f:
    R = pickle.load(f)

METHOD_COLORS = {
    "MLE": "#888888", "Laplace": "#d62728", "Add-k": "#ff7f0e",
    "Unigram Prior": "#bcbd22", "Good-Turing": "#17becf", "Backoff (Katz)": "#9467bd",
    "Linear Interp.": "#8c564b", "Context-Dep. Interp.": "#e377c2",
    "Witten-Bell": "#1f77b4", "Stupid Backoff": "#c49c94",
    "Abs. Disc. (MLE fallback)": "#98df8a", "Kneser-Ney": "#2ca02c",
    "Modified Kneser-Ney": "#0b5e0b",
}

# 1. Zipf's Law -- log-log rank vs frequency
freqs = sorted(d.uni.values(), reverse=True)
ranks = np.arange(1, len(freqs) + 1)
fig, ax = plt.subplots(figsize=(7, 5.5))
ax.loglog(ranks, freqs, ".", ms=3, color="#1f77b4", alpha=0.6, label="Observed word frequency")
C = freqs[0] * ranks[0]
ax.loglog(ranks, C / ranks, "--", color="#d62728", lw=1.8, label=r"Ideal Zipf: $f = C/rank$")
ax.set_xlabel("Rank (log scale)")
ax.set_ylabel("Frequency (log scale)")
ax.set_title("Zipf's Law")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/1_zipfs_law.png")
plt.close(fig)

# 2. Good-Turing log-log fit: N_c vs c
cs = np.array(sorted(R["Nc_raw"].keys()))
ns = np.array([R["Nc_raw"][c] for c in cs])
mask = ns > 0
a, b = R["gt_fit"]
fig, ax = plt.subplots(figsize=(7, 5.5))
ax.loglog(cs[mask], ns[mask], "o", ms=7, color="#2ca02c", label=r"Observed $N_c$ (bigram frequency-of-frequencies)")
c_line = np.linspace(cs.min(), cs.max(), 100)
ax.loglog(c_line, np.exp(a + b * np.log(c_line)), "--", color="#d62728", lw=1.8,
          label=fr"Fit: $\log N_c = {a:.2f} + {b:.2f}\log c$")
ax.set_xlabel("Count $c$ (log scale)")
ax.set_ylabel("$N_c$ = # bigram types with count $c$ (log scale)")
ax.set_title("Good-Turing Fit")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/2_good_turing_loglog.png")
plt.close(fig)

# 3. Example bigram probability comparison (log scale, grouped bars)
labels = list(R["example_bigrams"].keys())
methods = list(R["example_probs"][labels[0]].keys())
fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(labels))
width = 0.065
for i, m in enumerate(methods):
    vals = [max(R["example_probs"][lab][m], 1e-9) for lab in labels]
    ax.bar(x + (i - len(methods) / 2) * width, vals, width, label=m, color=METHOD_COLORS[m])
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels([l.split(": ")[0] + "\n" + l.split(": ")[1] for l in labels], fontsize=9)
ax.set_ylabel("P(w2 | w1)  (log scale)")
ax.set_title("Bigram Probability Comparison")
ax.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
fig.tight_layout()
fig.savefig(f"{OUT}/3_probability_comparison.png")
plt.close(fig)

# 4. Probability mass reserved for unseen continuations
mr = R["mass_reserved_avg"]
methods_sorted = sorted(mr.keys(), key=lambda m: mr[m])
fig, ax = plt.subplots(figsize=(9, 7.5))
vals = [mr[m] for m in methods_sorted]
colors = [METHOD_COLORS[m] for m in methods_sorted]
ax.barh(methods_sorted, vals, color=colors)
for i, v in enumerate(vals):
    ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
ax.set_xlabel("Avg. probability mass assigned to UNSEEN continuations")
ax.set_title("Reserved Mass for Unseen Bigrams")
fig.tight_layout()
fig.savefig(f"{OUT}/4_unseen_mass_reserved.png")
plt.close(fig)

# 5. Final test perplexity comparison (log scale to fit MLE)
res = R["results"]
methods_sorted = sorted(res.keys(), key=lambda m: res[m]["perplexity"])
vals = [res[m]["perplexity"] for m in methods_sorted]
colors = [METHOD_COLORS[m] for m in methods_sorted]
fig, ax = plt.subplots(figsize=(9, 7.5))
bars = ax.barh(methods_sorted, vals, color=colors)
ax.set_xscale("log")
ax.set_xlabel("Test-set Perplexity (log scale, lower = better)")
ax.set_title("Test Perplexity")
for i, (m, v) in enumerate(zip(methods_sorted, vals)):
    z = res[m]["zero_prob_events"]
    tag = f"  ({v:,.0f}, {z} zero-prob events)" if z > 0 else f"  ({v:,.1f})"
    ax.text(v * 1.05, i, tag, va="center", fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/5_perplexity_comparison.png")
plt.close(fig)

# 6. Add-k hyperparameter sweep
ks, pps = zip(*R["addk_curve"])
fig, ax = plt.subplots(figsize=(7, 5.5))
ax.plot(ks, pps, "o-", color="#ff7f0e")
ax.set_xscale("log")
ax.set_xlabel("k (log scale)")
ax.set_ylabel("Dev-set Perplexity")
ax.set_title("Add-k Hyperparameter Sweep")
best_k = R["best_k"]
best_pp = dict(R["addk_curve"])[best_k]
ax.plot(best_k, best_pp, "*", ms=20, color="#d62728", zorder=5, label=f"best k={best_k}")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/6_addk_sweep.png")
plt.close(fig)

# 7. Sparsity growth with n-gram order
sp = R["sparsity"]
fig, ax = plt.subplots(figsize=(6.5, 5.5))
orders = list(sp.keys())
vals = [sp[o] * 100 for o in orders]
bars = ax.bar(orders, vals, color=["#2ca02c", "#ff7f0e", "#d62728"])
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%", ha="center", fontsize=10)
ax.set_ylabel("% of test n-grams UNSEEN in training")
ax.set_title("Sparsity by N-gram Order")
ax.set_ylim(0, 100)
fig.tight_layout()
fig.savefig(f"{OUT}/7_sparsity_by_order.png")
plt.close(fig)

# 8. KN Ablation: does continuation probability actually help, and why?
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

ax = axes[0]
pair = ["Abs. Disc. (MLE fallback)", "Kneser-Ney"]
vals = [res[m]["perplexity"] for m in pair]
bars = ax.bar(pair, vals, color=[METHOD_COLORS[m] for m in pair], width=0.55)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}", ha="center", fontsize=11)
ax.set_ylabel("Test-set Perplexity")
ax.set_title("Abs. Disc. vs Kneser-Ney")
ax.set_xticks(range(len(pair)))
ax.set_xticklabels([m.replace(" (", "\n(") for m in pair], fontsize=9)

ax2 = axes[1]
sf = R["sf_example"]
word = sf["word"]
bars2 = ax2.bar(["P_uni(w)", "P_cont(w)"], [sf["p_uni"], sf["p_cont"]],
                 color=["#8c564b", "#2ca02c"], width=0.55)
for b, v in zip(bars2, [sf["p_uni"], sf["p_cont"]]):
    ax2.text(b.get_x() + b.get_width() / 2, v, f"{v:.5f}", ha="center", va="bottom", fontsize=10)
ax2.set_ylabel("Probability")
ax2.set_title(f"The 'San Francisco' Effect: '{word}'")
fig.suptitle("KN Ablation", fontsize=13)
fig.tight_layout()
fig.savefig(f"{OUT}/8_kn_ablation.png")
plt.close(fig)

# 9. Witten-Bell (data-driven lambda) vs tuned Context-Dependent Interp.
wb_curve = sorted(R["wb_curve"], key=lambda x: x[0])
c1s = np.array([c[0] for c in wb_curve])
lam_wb = np.array([c[1] for c in wb_curve])
lam_cdi = np.array([c[2] for c in wb_curve])
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(c1s, lam_wb, s=6, alpha=0.35, color="#1f77b4", label="Witten-Bell")
ax.plot(c1s, lam_cdi, "--", color="#e377c2", lw=2.2,
        label=f"Context-Dep. Interp. (k={R['best_cd_k']})")
ax.set_xscale("log")
ax.set_xlabel(r"$C(w_1)$ — context frequency (log scale)")
ax.set_ylabel(r"$\lambda_1(w_1)$ — weight on bigram MLE")
ax.set_title("Interpolation Weight: Witten-Bell vs Tuned")
ax.legend(fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig(f"{OUT}/9_witten_bell_vs_tuned.png")
plt.close(fig)

# 10. Stupid Backoff: raw score is not a valid probability distribution
sb_Z = R["sb_Z"]
contexts = list(sb_Z.keys())
zvals = list(sb_Z.values())
fig, ax = plt.subplots(figsize=(8, 5.5))
bars = ax.bar(contexts, zvals, color="#c49c94")
ax.axhline(1.0, color="#d62728", ls="--", lw=1.8, label="Z = 1.0 (valid probability distribution)")
for b, v in zip(bars, zvals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
ax.set_ylabel(r"$Z(w_1) = \sum_{w_2} S(w_2|w_1)$")
ax.set_title("Stupid Backoff Normalization")
ax.set_xticks(range(len(contexts)))
ax.set_xticklabels(contexts, rotation=30, ha="right")
ax.legend(fontsize=9)
ax.set_ylim(0, max(zvals) * 1.15)
fig.tight_layout()
fig.savefig(f"{OUT}/10_stupid_backoff_normalization.png")
plt.close(fig)

print("All 10 plots saved to", OUT)