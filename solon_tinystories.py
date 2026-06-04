"""
SOLON on TinyStories (~1M words) -- real text, still no net, no backprop.

Loads ~1M words of TinyStories, then runs the same compression-as-learning
pipeline from solon.py:
    predict (back-off bits) -> chunk (RePair) -> abstract (categories)
    -> one-shot word learning.

Usage:
    python3 solon_tinystories.py [path_to_tinystories.txt] [n_words]

Defaults to ./tinystories-valid.txt and 1_000_000 words.
"""

import re
import sys
import time
import math
import random
import collections

import solon                       # reuse the core: CompressionLM, repair, etc.
from solon import (CompressionLM, repair, expand, induce_classes,
                   ConstructionGrammar, guess_label, END)

WORD = re.compile(r"[a-z]+'?[a-z]*")
SENT_SPLIT = re.compile(r"[.!?]+")


def load_tinystories(path, n_words):
    """Tokenize into sentences of word tokens, each ending in END.
    Lowercased; punctuation dropped except sentence boundaries."""
    text = open(path, encoding="utf-8").read().lower()
    text = text.replace("<|endoftext|>", ". ")          # story break = boundary
    sents, tokens = [], []
    for raw_sent in SENT_SPLIT.split(text):
        ws = WORD.findall(raw_sent)
        if not ws:
            continue
        s = ws + [END]
        sents.append(s)
        tokens.extend(s)
        if len(tokens) >= n_words:
            break
    return sents, tokens


def banner(c="-"):
    print(c * 72)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "tinystories-valid.txt"
    n_words = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000

    banner("=")
    print("SOLON on TinyStories  -  learning real text by compression")
    print("                         (no transformer, no backprop)")
    banner("=")

    t0 = time.time()
    sents, tokens = load_tinystories(path, n_words)
    split = int(len(sents) * 0.97)
    train_sents, test_sents = sents[:split], sents[split:]
    train_tokens = [t for s in train_sents for t in s]
    vocab = set(train_tokens)
    print(f"loaded {len(tokens):,} tokens / {len(sents):,} sentences "
          f"/ vocab {len(vocab):,}   ({time.time()-t0:.1f}s)")

    # --- 1. prediction by compression ------------------------------------
    banner()
    print("[1] PREDICTION BY COMPRESSION (Witten-Bell back-off, order 3)")
    t0 = time.time()
    lm = CompressionLM(order=3)
    lm.fit(train_tokens, progress=True)
    ppl = lm.perplexity(test_sents[:400], progress=True)
    print(f"    held-out perplexity: {ppl:.1f}   "
          f"(bits/word = {math.log2(ppl):.2f})   ({time.time()-t0:.1f}s)")
    demo = "once upon a time there was a little girl".split() + [END]
    print("    per-word surprisal (bits) -- a reading-time proxy:")
    for w, b in lm.sentence_bits(demo):
        print(f"        {w:<8} {b:5.2f}  {'#'*int(b)}")

    # --- 2. construction library -----------------------------------------
    banner()
    print("[2] CONSTRUCTION LIBRARY (RePair chunks that shrink the corpus)")
    t0 = time.time()
    rep_tokens = train_tokens[:300_000]
    rules, _ = repair(rep_tokens, max_rules=400, progress=True)
    chunks = sorted(((len(expand(nt, rules)), expand(nt, rules)) for nt in rules),
                    reverse=True)
    seen, shown = set(), 0
    for size, ch in chunks:
        key = tuple(ch)
        if size >= 3 and END not in ch and key not in seen:
            seen.add(key)
            print(f"        [{' '.join(ch)}]")
            shown += 1
        if shown >= 12:
            break
    print(f"    ({time.time()-t0:.1f}s)")

    # --- 3. induced categories -------------------------------------------
    banner()
    print("[3] INDUCED CATEGORIES (top words merged by shared context)")
    t0 = time.time()
    classes, assigned, vecs, idf = induce_classes(
        train_tokens, min_count=40, thresh=0.20, top_k=300, progress=True)
    # show the largest, most coherent classes
    freq = collections.Counter(train_tokens)
    classes_sorted = sorted(classes, key=lambda c: -sum(freq[w] for w in c))
    shown = 0
    for members in classes_sorted:
        if len(members) >= 4:
            top = sorted(members, key=lambda w: -freq[w])[:12]
            print(f"        ({len(members):>3}) {' '.join(top)}")
            shown += 1
        if shown >= 10:
            break
    print(f"    ({time.time()-t0:.1f}s)")

    # --- 4. one-shot word learning ---------------------------------------
    banner()
    print("[4] ONE-SHOT WORD LEARNING (productivity)")
    cg = ConstructionGrammar(train_tokens, assigned)
    nonce = [("zorp", "the zorp was happy".split()),
             ("glip", "she wanted to glip".split()),
             ("blicket", "he saw a blicket".split())]
    for new, frame in nonce:
        cls, sim = cg.learn_word_oneshot(new, frame, vecs, idf)
        if cls is None:
            print(f"    '{new}' (in: {' '.join(frame)}) -> no class matched")
            continue
        members = [w for w, c in assigned.items() if c == cls]
        top = sorted(members, key=lambda w: -freq[w])[:10]
        print(f"    '{new}'  seen once in: \"{' '.join(frame)}\"")
        print(f"        -> {guess_label(members)}-like class (cos {sim:.2f}): {' '.join(top)}")

    # --- 5. grammaticality on hand-built minimal pairs -------------------
    banner()
    print("[5] GRAMMATICALITY = FEWER BITS (hand-built minimal pairs)")
    pairs = [
        ("agreement", "she was very happy", "she were very happy"),
        ("agreement", "the dog runs fast", "the dog run fast"),
        ("det-noun",  "he saw a cat", "he saw a cats"),
        ("order",     "once upon a time", "time a upon once"),
        ("pronoun",   "the girl said she was sad", "the girl said she were sad"),
    ]
    score = lambda s: sum(b for _, b in lm.sentence_bits(s.split() + [END]))
    correct = 0
    for kind, good, bad in pairs:
        m = score(bad) - score(good)
        ok = m > 0
        correct += ok
        print(f"        {kind:<10} {good:<24} > {bad:<24} {m:+6.1f} b  {'ok' if ok else 'X'}")
    print(f"    accuracy: {correct}/{len(pairs)}")

    banner("=")
    print("Real text, ~1M words. Categories, phrases and one-shot generalization")
    print("emerged from counting and refactoring alone -- no gradients.")
    banner("=")


if __name__ == "__main__":
    main()
