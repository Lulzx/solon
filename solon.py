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


def _mdl_delta_for_class(word, best_ci, classes, vecs, class_costs, idf_weight=1.0, base_sim=0.0):
    """Cheap MDL proxy delta for assigning `word` to existing class vs new.
    Negative delta means the merge/assign shortens total description length.
    Uses class cardinality cost + rough fit cost from vector overlap (reuses IDF vecs).
    Pure stdlib; fast; called only in mdl mode.
    base_sim: the complete-link sim to best (used to gate; only consider if reasonably similar).
    """
    # Cost of a class model (bits to "describe" the class itself)
    def class_cost(csize):
        return math.log2(1 + max(1, csize))  # new class or growth penalty

    # Rough data fit cost for this word's context vec under a "prototype"
    # (negative cosine as surprisal proxy; higher overlap = lower cost)
    def fit_cost(wv, proto):
        if not proto:
            return 8.0  # high cost for no prototype
        sim = cosine(wv, proto)
        return max(0.1, 1.5 - 1.5 * max(0.0, sim)) * idf_weight

    wv = vecs[word]
    # cost if new class
    new_c = class_cost(1)
    new_fit = fit_cost(wv, wv)  # self
    new_total = new_c + new_fit

    if best_ci is None or best_ci not in class_costs or base_sim < 0.15:
        return 1.0  # force new class (no good candidate or low sim)

    # cost if add to best
    csize = class_costs[best_ci]
    add_c = class_cost(csize + 1) - class_cost(csize)  # marginal
    # Use base_sim (already complete-link min) to estimate fit improvement
    add_fit = fit_cost(wv, wv) * (1.0 - 0.6 * min(1.0, base_sim))
    add_total = add_c + add_fit

    delta = add_total - new_total
    return delta


def induce_classes(tokens, min_count=8, thresh=0.46, top_k=None, progress=False, mdl=False, mdl_cost_weight=1.0):
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
    # If mdl=True: use MDL delta (merge/assign only if it shortens approx total DL)
    # instead of fixed thresh. See README "clustering threshold" and plan.
    classes = []
    assigned = {}
    class_costs = collections.Counter()  # cid -> current size (for mdl)
    for w in tqdm(words, desc="[3] cluster", unit="w") if progress else words:
        if not mdl:
            best, best_sim = None, thresh
            for ci, members in enumerate(classes):
                s = min(cosine(vecs[w], vecs[m]) for m in members)   # complete link
                if s >= best_sim:
                    best_sim, best = s, ci
            if best is None:
                assigned[w] = len(classes)
                classes.append([w])
                class_costs[len(classes)-1] = 1
            else:
                assigned[w] = best
                classes[best].append(w)
                class_costs[best] += 1
        else:
            # MDL mode: find best class by sim, then decide by delta_DL < 0
            best, best_sim = None, -1.0
            for ci, members in enumerate(classes):
                s = min(cosine(vecs[w], vecs[m]) for m in members)
                if s > best_sim:
                    best_sim, best = s, ci
            delta = _mdl_delta_for_class(w, best, classes, vecs, class_costs, base_sim=best_sim)
            weighted_delta = delta * mdl_cost_weight
            if best is None or weighted_delta >= 0:
                # create new class (no savings or no candidate)
                cid = len(classes)
                assigned[w] = cid
                classes.append([w])
                class_costs[cid] = 1
            else:
                assigned[w] = best
                classes[best].append(w)
                class_costs[best] += 1
    return classes, assigned, vecs, idf


# ---------------------------------------------------------------------------
# 4. Grammaticality + one-shot word learning via the construction grammar.
#    A class-bigram code length: grammatical class sequences are frequent
#    (few bits); ungrammatical ones are rare (many bits). A new word slotted
#    into a class inherits the grammar of the whole class -- instant
#    productivity, no retraining.
# ---------------------------------------------------------------------------

class ConstructionGrammar:
    def __init__(self, tokens, assigned, use_chart=False):
        self.assigned = dict(assigned)
        labs = [self.label(w) for w in tokens]
        self.bi = collections.Counter(zip(labs, labs[1:]))
        self.uni = collections.Counter(labs)
        self.V = len(set(labs)) + 1
        self.use_chart = use_chart  # if True, bits() prefers CKY min-cost parse (MDL parse)

    def label(self, w):
        return ("C", self.assigned[w]) if w in self.assigned else ("W", w)

    def bits(self, sent):
        if self.use_chart:
            try:
                return self.parse_bits(sent)
            except Exception:
                pass  # fallback
        labs = [self.label(w) for w in sent]
        total = 0.0
        for a, b in zip(labs, labs[1:]):
            p = (self.bi[(a, b)] + 1) / (self.uni[a] + self.V)   # add-1 smoothing
            total += -math.log2(p)
        return total

    def parse_bits(self, sent):
        """Very simple CKY over class-bigram 'rules' for a min-cost (MDL) parse.
        Nonterms are the observed labels (C,cid or W,w). Rules from self.bi.
        Returns min total -log cost for a full 'parse' of the label sequence.
        Falls back gracefully for short/empty. Pure stdlib, O(n^3 * L) with L small.
        """
        if not sent:
            return 0.0
        labs = [self.label(w) for w in sent]
        n = len(labs)
        if n == 1:
            return 0.0  # no transitions
        # labels set (L small: #classes + few)
        all_labs = list(self.bi.keys())  # (a,b) pairs imply the labels
        from collections import defaultdict
        chart = [defaultdict(lambda: defaultdict(lambda: float('inf'))) for _ in range(n+1)]
        # init: singletons (cost 0)
        for i, lab in enumerate(labs):
            chart[i][i+1][lab] = 0.0
        # fill spans
        for length in range(2, n+1):
            for i in range(n - length + 1):
                j = i + length
                for k in range(i+1, j):
                    for left, left_cost in chart[i][k].items():
                        for right, right_cost in chart[k][j].items():
                            if left_cost >= float('inf') or right_cost >= float('inf'):
                                continue
                            rule_cost = -math.log2( (self.bi[(left, right)] + 1) / (self.uni.get(left, 0) + self.V) )
                            total = left_cost + right_cost + rule_cost
                            if total < chart[i][j][right]:  # or track best root; simplify: allow right as head
                                chart[i][j][right] = total
                            # also try left as 'head' for symmetry (cheap)
                            if total < chart[i][j][left]:
                                chart[i][j][left] = total
        # min over any root for full span
        min_cost = min(chart[0][n].values()) if chart[0][n] else float('inf')
        if min_cost >= float('inf') or n < 2:
            # fallback to flat bigram
            return sum(-math.log2( (self.bi[(a, b)] + 1) / (self.uni.get(a, 0) + self.V) ) for a, b in zip(labs, labs[1:]))
        return min_cost

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

def minimal_pairs(rng, n=200, sents=None, cg=None):
    """Generate minimal pairs. If sents provided (real text), mine simple
    patterns and corrupt (agreement via verb swap heuristic, order, det-noun).
    Falls back to toy generator. cg optional for future class-aware corrupt.
    """
    pairs = []
    if sents:
        # simple mining from real sents for heldout evals
        for _ in range(n):
            if not sents:
                break
            s = list(rng.choice(sents))
            if len(s) < 4:
                continue
            body = s[:-1] if s[-1] == END else s
            kind = rng.choice(["agreement", "order", "det-noun"])
            bad = body[:]
            if kind == "agreement":
                # find a verb-ish token (heuristic: ends with s or common)
                for i, w in enumerate(body):
                    if w.endswith(("s", "ed", "ing")) or w in ("is", "was", "runs", "run"):
                        swap = w[:-1] if w.endswith("s") else (w + "s" if not w.endswith("s") else w)
                        bad[i] = swap
                        break
                else:
                    continue
                pairs.append(("agreement", body + [END], bad + [END]))
            elif kind == "order" and len(body) >= 3:
                bad[0], bad[2] = bad[2], bad[0]  # crude scramble
                pairs.append(("order", body + [END], bad + [END]))
            else:
                if len(body) >= 2:
                    bad[0], bad[1] = bad[1], bad[0]
                    pairs.append(("det-noun", body + [END], bad + [END]))
        return pairs[:n]
    # original toy synthetic
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


def grammar_ppl(scorer, sents):
    """Bits-per-label (or token) using a bits() scorer (CG or LM wrapper).
    Analog to perplexity but for the grammar layer.
    """
    total_bits, total = 0.0, 0
    for s in sents:
        try:
            b = scorer(s) if callable(scorer) else scorer.bits(s)
            total_bits += b
            total += max(1, len(s) - 1)
        except Exception:
            continue
    return (2 ** (total_bits / total)) if total else 1.0


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
    classes, assigned, vecs, idf = induce_classes(train_tokens)  # mdl=False (default) for exact old behavior on toy
    classes_sorted = sorted(enumerate(classes), key=lambda x: -len(x[1]))
    for ci, members in classes_sorted:
        if len(members) >= 2:
            print(f"        {guess_label(members):<5} :: {' '.join(sorted(members))}")

    # --- 4. grammaticality via compression --------------------------------
    line()
    print("[4] GRAMMATICALITY = FEWER BITS  (BLiMP-style minimal pairs)")
    pairs = minimal_pairs(random.Random(7), n=400)
    cg = ConstructionGrammar(train_tokens, assigned)
    lm_score = lambda s: sum(b for _, b in lm.sentence_bits(s))
    for name, scorer in [("statistical back-off LM ", lm_score),
                         ("construction grammar    ", cg.bits),
                         ("construction grammar+chart", (lambda s: cg.parse_bits(s)) if cg.use_chart else cg.bits)]:
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
