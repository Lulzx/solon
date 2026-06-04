"""
SOLON - Self-Organizing Library Of coNstructions
================================================

A language learner with NO transformer and NO backpropagation.

First principle: learning == compression (MDL / Solomonoff).
The hypothesis that compresses the data into the shortest reusable
program generalizes best. So instead of fitting a giant function with
gradient descent, SOLON:

  1. PREDICTS by compression  - a Witten-Bell back-off model gives
     calibrated next-symbol probabilities and a code length (bits).
     Grammaticality == fewer bits. (no parameters are trained)

  2. CHUNKS by compression     - RePair greedily replaces the most
     frequent adjacent pair with a new symbol, discovering constituents
     (a context-free construction library) purely to shrink the corpus.

  3. ABSTRACTS by distribution - words that share contexts are merged
     into equivalence classes (categories like noun / singular-verb).
     This is the "dream" refactor: it manufactures generalization power
     offline from already-seen text.

  4. LEARNS NEW WORDS IN ONE SHOT - a brand-new word seen a single time
     is slotted into a category by its one context, after which the
     construction grammar judges novel sentences containing it correctly
     -- where the pure statistical model is blind. This is productivity
     (the "wug" effect) with zero gradient steps.

Everything is counting and set operations. Run:  python3 solon.py
"""

import math
import random
import collections

try:
    from tqdm import tqdm
except ImportError:                                    # tqdm is optional
    def tqdm(it, **kw):
        return it

# ---------------------------------------------------------------------------
# 0. A toy world with real structure: word order + subject-verb agreement.
#    Number is strongly marked (most sentences intransitive) so the learner
#    can discover singular vs plural sub-categories distributionally.
# ---------------------------------------------------------------------------

DET_SING = ["the", "a"]
DET_PLUR = ["the"]
ADJS     = ["big", "small", "red", "happy"]
N_SING   = ["dog", "cat", "bird", "man", "woman"]
N_PLUR   = ["dogs", "cats", "birds", "men", "women"]
VI_SING  = ["runs", "sleeps", "sings"]
VI_PLUR  = ["run", "sleep", "sing"]
VT_SING  = ["sees", "likes", "chases"]
VT_PLUR  = ["see", "like", "chase"]
END = "."


def np_tokens(rng, number):
    det = rng.choice(DET_SING if number == "sing" else DET_PLUR)
    toks = [det]
    for _ in range(rng.choice([0, 0, 1, 1, 2])):      # 0-2 adjectives
        toks.append(rng.choice(ADJS))
    toks.append(rng.choice(N_SING if number == "sing" else N_PLUR))
    return toks


INTRANS_PROB = 0.85          # subjects dominate -> number is the strong signal


def sentence(rng):
    number = rng.choice(["sing", "plur"])
    toks = np_tokens(rng, number)
    if rng.random() < INTRANS_PROB:                    # intransitive (number-marked)
        toks.append(rng.choice(VI_SING if number == "sing" else VI_PLUR))
    else:                                              # transitive
        toks.append(rng.choice(VT_SING if number == "sing" else VT_PLUR))
        toks += np_tokens(rng, rng.choice(["sing", "plur"]))
    toks.append(END)
    return toks


def make_corpus(n, seed=0):
    rng = random.Random(seed)
    sents = [sentence(rng) for _ in range(n)]
    tokens = [t for s in sents for t in s]
    return sents, tokens


# ---------------------------------------------------------------------------
# 1. Prediction by compression: Witten-Bell interpolated back-off model.
#    Properly normalized at every order (a convex combination of normalized
#    distributions), so the bits it reports are a real code length.
# ---------------------------------------------------------------------------

class CompressionLM:
    def __init__(self, order=3):
        self.order = order
        self.ctx = [collections.defaultdict(collections.Counter)
                    for _ in range(order + 1)]
        self.vocab = set()

    def fit(self, tokens, progress=False):
        self.vocab.update(tokens)
        self.V = len(self.vocab) + 1                    # +1 for unseen words
        it = tqdm(range(len(tokens)), desc="[1] fit", unit="tok") if progress \
            else range(len(tokens))
        for i in it:
            w = tokens[i]
            for k in range(self.order + 1):
                if i - k < 0:
                    break
                self.ctx[k][tuple(tokens[i - k:i])][w] += 1

    def prob(self, history, w):
        p = 1.0 / self.V                               # order -1: uniform
        for k in range(self.order + 1):
            ctx = tuple(history[len(history) - k:]) if k else ()
            counter = self.ctx[k].get(ctx)
            if not counter:
                continue                               # no evidence -> weight 0
            n = sum(counter.values())
            t = len(counter)
            lam = n / (n + t)                          # Witten-Bell weight
            p = lam * (counter.get(w, 0) / n) + (1 - lam) * p
        return p

    def bits(self, history, w):
        return -math.log2(self.prob(history, w))

    def sentence_bits(self, sent):
        return [(w, self.bits(sent[:i], w)) for i, w in enumerate(sent)]

    def perplexity(self, sents, progress=False):
        total_bits, total_tok = 0.0, 0
        for s in (tqdm(sents, desc="[1] ppl", unit="sent") if progress else sents):
            for i, w in enumerate(s):
                total_bits += self.bits(s[:i], w)
                total_tok += 1
        return 2 ** (total_bits / total_tok)


# ---------------------------------------------------------------------------
# 2. Chunking by compression: RePair builds a construction library.
# ---------------------------------------------------------------------------

def repair(tokens, max_rules=120, progress=False):
    seq = list(tokens)
    rules = {}
    nid = 0
    bar = tqdm(total=max_rules, desc="[2] repair", unit="rule") if progress else None
    while len(rules) < max_rules:
        pairs = collections.Counter(zip(seq, seq[1:]))
        if not pairs:
            break
        (a, b), f = pairs.most_common(1)[0]
        if f < 5:                                      # not worth a rule
            break
        nt = ("NT", nid); nid += 1
        rules[nt] = (a, b)
        out, i = [], 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                out.append(nt); i += 2
            else:
                out.append(seq[i]); i += 1
        seq = out
        if bar:
            bar.update(1)
    if bar:
        bar.close()
    return rules, seq


def expand(sym, rules):
    if isinstance(sym, tuple) and sym[0] == "NT":
        a, b = rules[sym]
        return expand(a, rules) + expand(b, rules)
    return [sym]


# ---------------------------------------------------------------------------
# 3. Abstraction by distribution: merge words that share contexts into
#    equivalence classes (categories). The "dream" refactor.
# ---------------------------------------------------------------------------

def context_vector(word, left, right):
    v = collections.Counter()
    for c, n in left[word].items():
        v[("L", c)] += n
    for c, n in right[word].items():
        v[("R", c)] += n
    return v


def cosine(a, b):
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[k] * b[k] for k in common)
    na = math.sqrt(sum(x * x for x in a.values()))
    nb = math.sqrt(sum(x * x for x in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def induce_classes(tokens, min_count=8, thresh=0.46, top_k=None, progress=False):
    left = collections.defaultdict(collections.Counter)
    right = collections.defaultdict(collections.Counter)
    for i, w in enumerate(tokens):
        # the sentence boundary is punctuation, not lexical context: if kept,
        # its huge frequency swamps the discriminative signal (e.g. it would
        # make every clause-final verb look alike regardless of number).
        if i > 0 and tokens[i - 1] != END:
            left[w][tokens[i - 1]] += 1
        if i < len(tokens) - 1 and tokens[i + 1] != END:
            right[w][tokens[i + 1]] += 1
    freq = collections.Counter(tokens)
    words = [w for w in set(tokens)
             if w != END and sum(left[w].values()) + sum(right[w].values()) >= min_count]
    if top_k:                                          # cluster only frequent words
        words = sorted(words, key=lambda w: -freq[w])[:top_k]
    raw = {w: context_vector(w, left, right) for w in words}
    # IDF weighting: a context like "the" that precedes almost every word is
    # uninformative and would dominate cosine; a context like "runs" that only
    # follows singular nouns is highly discriminative. Down-weight the common
    # ones (this is what lets number survive the shared determiner context).
    df = collections.Counter()
    for v in raw.values():
        for f in v:
            df[f] += 1
    n = len(words)
    idf = {f: math.log(1 + n / c) for f, c in df.items()}
    vecs = {w: collections.Counter({f: c * idf[f] for f, c in v.items()})
            for w, v in raw.items()}
    words.sort(key=lambda w: -sum(raw[w].values()))    # frequent words seed classes
    # Complete-linkage clustering: a word joins a class only if it is similar
    # to EVERY member. This prevents single-link "chaining" (everything follows
    # "the", so single-link would merge the whole vocabulary into one blob).
    classes = []
    assigned = {}
    for w in tqdm(words, desc="[3] cluster", unit="w") if progress else words:
        best, best_sim = None, thresh
        for ci, members in enumerate(classes):
            s = min(cosine(vecs[w], vecs[m]) for m in members)   # complete link
            if s >= best_sim:
                best_sim, best = s, ci
        if best is None:
            assigned[w] = len(classes)
            classes.append([w])
        else:
            assigned[w] = best
            classes[best].append(w)
    return classes, assigned, vecs, idf


# ---------------------------------------------------------------------------
# 4. Grammaticality + one-shot word learning via the construction grammar.
#    A class-bigram code length: grammatical class sequences are frequent
#    (few bits); ungrammatical ones are rare (many bits). A new word slotted
#    into a class inherits the grammar of the whole class -- instant
#    productivity, no retraining.
# ---------------------------------------------------------------------------

class ConstructionGrammar:
    def __init__(self, tokens, assigned):
        self.assigned = dict(assigned)
        labs = [self.label(w) for w in tokens]
        self.bi = collections.Counter(zip(labs, labs[1:]))
        self.uni = collections.Counter(labs)
        self.V = len(set(labs)) + 1

    def label(self, w):
        return ("C", self.assigned[w]) if w in self.assigned else ("W", w)

    def bits(self, sent):
        labs = [self.label(w) for w in sent]
        total = 0.0
        for a, b in zip(labs, labs[1:]):
            p = (self.bi[(a, b)] + 1) / (self.uni[a] + self.V)   # add-1 smoothing
            total += -math.log2(p)
        return total

    def learn_word_oneshot(self, word, one_context_sentence, vecs, idf):
        """Assign a brand-new word to the class whose members share its
        single observed context. One exposure, no gradients."""
        i = one_context_sentence.index(word)
        v = collections.Counter()
        if i > 0 and one_context_sentence[i - 1] != END:
            v[("L", one_context_sentence[i - 1])] += idf.get(("L", one_context_sentence[i - 1]), 1.0)
        if i < len(one_context_sentence) - 1 and one_context_sentence[i + 1] != END:
            v[("R", one_context_sentence[i + 1])] += idf.get(("R", one_context_sentence[i + 1]), 1.0)
        # compare against each class CENTROID (sum of member context vectors).
        # Centroid, not best-member: a single frequent member has a huge norm
        # that would suppress cosine even when the contexts genuinely match.
        centroids = collections.defaultdict(collections.Counter)
        for w, c in self.assigned.items():
            if w in vecs:
                centroids[c].update(vecs[w])
        best, best_sim = None, 0.0
        for c, cen in centroids.items():
            sim = cosine(v, cen)
            if sim > best_sim:
                best_sim, best = sim, c
        if best is not None:
            self.assigned[word] = best
        return best, best_sim


# ---------------------------------------------------------------------------
# 5. Minimal pairs (BLiMP-style): grammatical vs corrupted.
# ---------------------------------------------------------------------------

def minimal_pairs(rng, n=200):
    pairs = []
    for _ in range(n):
        s = sentence(rng)
        body = s[:-1]                                   # drop "."
        # find the main verb (first verb after the subject noun)
        vidx = None
        for i, w in enumerate(body):
            if w in VI_SING + VI_PLUR + VT_SING + VT_PLUR:
                vidx = i; break
        if vidx is None:
            continue
        kind = rng.choice(["agreement", "order"])
        if kind == "agreement":
            v = body[vidx]
            swap = {**dict(zip(VI_SING, VI_PLUR)), **dict(zip(VI_PLUR, VI_SING)),
                    **dict(zip(VT_SING, VT_PLUR)), **dict(zip(VT_PLUR, VT_SING))}
            bad = body.copy(); bad[vidx] = swap[v]
            pairs.append(("agreement", body + [END], bad + [END]))
        else:                                           # swap determiner & noun
            bad = body.copy()
            if len(bad) >= 2:
                bad[0], bad[1] = bad[1], bad[0]
                pairs.append(("order", body + [END], bad + [END]))
    return pairs


def judge_accuracy(scorer, pairs):
    """A pair is correct if the grammatical sentence gets fewer bits."""
    by_kind = collections.defaultdict(lambda: [0, 0])
    for kind, good, bad in pairs:
        ok = scorer(good) < scorer(bad)
        by_kind[kind][0] += ok
        by_kind[kind][1] += 1
    return by_kind


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def line(c="-"):
    print(c * 70)


def guess_label(members):
    sets = [("DET", set(DET_SING + DET_PLUR)), ("ADJ", set(ADJS)),
            ("N.sg", set(N_SING)), ("N.pl", set(N_PLUR)),
            ("V.sg", set(VI_SING + VT_SING)), ("V.pl", set(VI_PLUR + VT_PLUR))]
    best, score = "?", 0
    for name, s in sets:
        ov = len(set(members) & s)
        if ov > score:
            score, best = ov, name
    return best


def main():
    print()
    line("=")
    print("SOLON  -  learning language by compression (no net, no backprop)")
    line("=")

    train_sents, train_tokens = make_corpus(4000, seed=1)
    test_sents, _ = make_corpus(500, seed=99)
    print(f"train: {len(train_sents)} sentences / {len(train_tokens)} tokens   "
          f"vocab: {len(set(train_tokens))}")

    # --- 1. prediction by compression -------------------------------------
    line()
    print("[1] PREDICTION BY COMPRESSION (Witten-Bell back-off, just counts)")
    lm = CompressionLM(order=3)
    lm.fit(train_tokens)
    print(f"    held-out perplexity: {lm.perplexity(test_sents):.2f}")
    demo = ["the", "big", "dog", "runs", "."]
    print(f"    per-word surprisal (bits) -- a reading-time proxy:")
    for w, b in lm.sentence_bits(demo):
        bar = "#" * int(b)
        print(f"        {w:<6} {b:5.2f}  {bar}")

    # --- 2. chunking by compression ---------------------------------------
    line()
    print("[2] CONSTRUCTION LIBRARY (RePair chunks that shrink the corpus)")
    rules, _ = repair(train_tokens)
    chunks = sorted(((len(expand(nt, rules)), expand(nt, rules)) for nt in rules),
                    reverse=True)
    seen = set()
    shown = 0
    for size, ch in chunks:
        key = tuple(ch)
        if size >= 2 and key not in seen and END not in ch:
            seen.add(key)
            print(f"        [{' '.join(map(str, ch))}]")
            shown += 1
        if shown >= 8:
            break

    # --- 3. abstraction by distribution -----------------------------------
    line()
    print("[3] INDUCED CATEGORIES (words merged by shared context = the 'dream')")
    classes, assigned, vecs, idf = induce_classes(train_tokens)
    classes_sorted = sorted(enumerate(classes), key=lambda x: -len(x[1]))
    for ci, members in classes_sorted:
        if len(members) >= 2:
            print(f"        {guess_label(members):<5} :: {' '.join(sorted(members))}")

    # --- 4. grammaticality via compression --------------------------------
    line()
    print("[4] GRAMMATICALITY = FEWER BITS  (BLiMP-style minimal pairs)")
    pairs = minimal_pairs(random.Random(7), n=400)
    cg = ConstructionGrammar(train_tokens, assigned)
    for name, scorer in [("statistical (back-off LM)", lambda s: lm.sentence_total(s)),
                         ("construction grammar      ", cg.bits)]:
        pass
    # statistical scorer
    lm_score = lambda s: sum(b for _, b in lm.sentence_bits(s))
    for name, scorer in [("statistical back-off LM ", lm_score),
                         ("construction grammar    ", cg.bits)]:
        acc = judge_accuracy(scorer, pairs)
        parts = "  ".join(f"{k}: {v[0]/v[1]*100:4.0f}%" for k, v in sorted(acc.items()))
        print(f"        {name}  {parts}")

    # --- 5. one-shot word learning ----------------------------------------
    line()
    print("[5] ONE-SHOT WORD LEARNING (productivity / the 'wug' effect)")
    new = "blicket"
    one_shot = ["the", new, "runs", "."]            # the ONLY time it is ever seen
    print(f"    new word '{new}' is shown exactly once:  {' '.join(one_shot)}")
    cls, sim = cg.learn_word_oneshot(new, one_shot, vecs, idf)
    members = [w for w, c in assigned.items() if c == cls]
    print(f"    -> slotted into category {guess_label(members)} "
          f"(cosine {sim:.2f}): {' '.join(sorted(members))}")

    # novel minimal pairs the word never appeared in
    novel = [
        ("the blicket sleeps .", "the blicket sleep ."),     # agreement
        ("the man sees the blicket .", "the man see the blicket ."),
        ("the blicket runs .", "blicket the runs ."),        # word order
    ]
    print(f"    judging sentences containing '{new}' that were NEVER in training")
    print(f"    (margin = bits(bad) - bits(good); >0 means correct, ~0 = guessing):")
    print(f"        {'minimal pair':<44}{'back-off LM':>14}{'SOLON':>9}")
    for good, bad in novel:
        g, b = good.split(), bad.split()
        lm_m = lm_score(b) - lm_score(g)
        cg_m = cg.bits(b) - cg.bits(g)
        print(f"        {good:<26}>  {bad:<16}{lm_m:>+10.1f} b{cg_m:>+8.1f} b")
    print("    (the back-off LM has no n-gram with 'blicket', so on word order it")
    print("     is BLIND, and on agreement it can only fall back to base-rate")
    print("     frequency -- guessing. SOLON judges all three structurally, because")
    print("     'blicket' inherited a whole category's grammar from ONE exposure.)")

    line("=")
    print("compression -> chunks -> categories -> one-shot productivity.")
    print("No transformer. No gradients. Learning was just finding the shorter code.")
    line("=")
    print()


if __name__ == "__main__":
    main()
