# SOLON — learning language by compression

[![ci](https://github.com/Lulzx/solon/actions/workflows/ci.yml/badge.svg)](https://github.com/Lulzx/solon/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**No transformer. No backpropagation. No gradients anywhere.**

A working proof-of-concept for the thesis that *learning **is** compression*
(MDL / Solomonoff): the shortest reusable description of the data is also the
one that generalizes furthest beyond it. This is the single most relevant
principle for a low-data regime like the [BabyLM Challenge](https://babylm.github.io/)
(10M–100M words), where the bottleneck is sample efficiency, not compute.

```
python3 solon.py        # pure stdlib, runs in ~1s on a laptop
```

## The idea

A standard LM stores knowledge in billions of weights, updated slowly by
gradient descent, and cannot acquire a new word in one exposure. A child can.
SOLON drops the neural net entirely and learns the way a compressor does — by
counting and refactoring — yielding four behaviours, each a direct consequence
of "shorter code = better model":

| Stage | Mechanism | First principle |
|------|-----------|-----------------|
| **1. Predict** | Witten-Bell back-off model → calibrated bits per word | grammaticality = fewer bits; bits/word = a reading-time proxy |
| **2. Chunk** | RePair: replace the most frequent pair with a new symbol | constituents are whatever shrinks the corpus |
| **3. Abstract** | merge words that share (IDF-weighted) contexts into categories | the "dream" refactor — generalization manufactured offline |
| **4. Generalize** | a new word slots into a category from **one** context | productivity (the "wug" effect) with zero retraining |

## What the demo shows

On a toy world with word order **and** subject–verb agreement, SOLON:

- induces clean categories from raw text — `N.sg`, `N.pl`, `V.sg`, `V.pl`,
  `ADJ`, `DET` — separated even by **number**, with no labels;
- judges BLiMP-style minimal pairs at 100% (grammatical = fewer bits);
- learns the nonce word **`blicket`** from a single sentence (`the blicket runs`),
  files it under `N.sg`, and then correctly judges agreement and word order on
  sentences it has **never seen**:

```
                                          back-off LM      SOLON
the blicket sleeps . > the blicket sleep .     +0.2 b     +11.0 b
the blicket runs .   > blicket the runs .      +0.0 b     +20.9 b   (LM blind)
```

The back-off model is *blind* on a novel word (no n-gram contains it) and can
only guess from base rates (~0-bit margin). SOLON answers structurally, because
`blicket` inherited an entire category's grammar from one exposure.

## How this maps onto the full architecture (SOLON, the design)

This script is **rungs 1–3** of a larger, deliberately gradient-free design:

1. **Prediction by compression** ✔ (here: a back-off model; scales to PPM/CTW)
2. **A growing construction library** ✔ (here: RePair; scales to MDL grammar
   induction / Bayesian Model Merging with variable slots)
3. **Distributional abstraction / "dreaming"** ✔ (here: IDF-weighted
   complete-link clustering; scales to ADIOS-style equivalence classes)
4. **One-shot, test-time acquisition** ✔ — learning and inference are the same
   operation, so the model never stops learning, exactly like a child.

The induced grammar is **fully inspectable** — you can print the categories and
constructions it discovered. For a language-*acquisition* venue that
interpretability is worth as much as the score.

## Honest limitations

- **Toy corpus.** A synthetic mini-English. The mechanisms are real; the scale
  is not. Real text needs sub-word units, a chart/Earley parser over the induced
  grammar, and variable-slot constructions (not just flat categories).
- **Agreement is captured via adjacent class bigrams.** Long-distance
  dependencies (across embedded clauses) need the hierarchical / slot-binding
  parser — that is the next rung.
- **Clustering threshold** is tuned for this world (0.46, complete-linkage).
  At scale, replace the threshold with an MDL stopping criterion: merge iff it
  shortens the total description length.

## Sub-word edition: morphology by compression (`solon_morphology.py`)

Word-level tokens are blind to morphology — `wug` and `wugs` are unrelated
symbols. A **character-level** PPM (same Witten-Bell engine, order 6) learns the
*shape* of the language and inflects words it has never seen:

```
python3 solon_morphology.py        # ~2s on 1.2M chars of TinyStories
```

- **Wug test** — for the fully novel stem `wug`, the regular plural `-s` costs
  **5.2 bits** vs `-z` **28.5** / `-q` **29.5**. For `blicket`, `-s` = 3.1 bits.
  The rule generalized; it even respects phonotactics (noun-like stems prefer
  `-s`, verb-like stems prefer `-ed`/`-ing`).
- **Why sub-word** — the word-level model scores `wugs` and `wugz` *identically*
  (both OOV → blind); the char model prefers the real plural by 23 bits.
- **Held-out compression**: 1.70 bits/char on unseen text.
- **Honest limitation** — a left-to-right char model learns *how* to pluralize
  but not *when*: given "two ___" it keeps the shorter bare form, because the
  distant number cue is lost after backing off on the novel stem. Deciding *to*
  inflect lives in the **word-level categories** (`solon.py` §3). Form (sub-word)
  and *when* (word-level) are complementary — the full system needs both.

## Files

- `solon.py` — core system (toy corpus, predictor, RePair, category induction,
  one-shot learner). Pure Python standard library.
- `solon_tinystories.py` — the same pipeline on ~1M words of real TinyStories.
- `solon_morphology.py` — character-level PPM; the wug test and morphology.
  Run `pip install tqdm` for progress bars (optional; degrades gracefully).

## Scaling to real BabyLM data

Swap `make_corpus()` for a loader over the strict-small 10M-word corpus, move
the predictor to character/sub-word PPM (robust to morphology — real wug tests),
and add a CKY parser so grammaticality uses minimum-description-length parses
rather than class bigrams. The eval pipeline (BLiMP, EWOK, reading-time) drops
in 2026; bits-per-word is already the right currency for the reading-time fit.
