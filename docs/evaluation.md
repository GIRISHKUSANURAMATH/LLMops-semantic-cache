# Evaluation

Two separate questions:

1. **Performance** - how much latency and how many LLM calls does the cache save on a workload?
2. **Quality** - at a given threshold, how often does it serve a *wrong* cached answer?

Both scripts are deterministic (mock LLM, seeded workload, built-in hashing embedder), so
anyone can reproduce the numbers below. Numbers were produced by actually running the scripts;
none are hand-written.

**Benchmark environment:** Python 3.13.18, Linux x86_64 (Ubuntu 24.04 container), 2 vCPU
Intel Xeon @ 2.10 GHz, 7 GB RAM. Run on 2026-10-31.

## 1. Performance: without cache vs with semantic cache

```bash
python scripts/benchmark.py
```

The workload is 200 requests drawn (seed 42) from the evaluation dataset with a Zipf-like
popularity distribution - a few questions asked often, in several phrasings. The mock LLM
sleeps a fixed 50 ms per call to stand in for network + generation time.

| metric | without cache | with semantic cache |
|---|---:|---:|
| LLM calls | 200 | 34 |
| cache hit rate | 0.0% | 83.0% |
| &nbsp;&nbsp;of which: hits on a differently worded query | 0 | 30 |
| mean latency (ms) | 50.51 | 8.77 |
| p50 latency (ms) | 50.48 | 0.14 |
| p95 latency (ms) | 50.70 | 50.83 |
| total wall time (s) | 10.12 | 1.76 |
| estimated tokens avoided | 0 | 5640 |
| estimated cost saved (USD) | 0.000000 | 0.002750 |

How to read this honestly:

- **136 of the 166 hits were exact repeats** (after normalisation). Only 30 were genuine
  paraphrase hits. An exact-match cache would already capture most of this workload's
  benefit; the semantic layer adds the remaining 30.
- The hit rate is a property of the **workload**, not the software. Real traffic with fewer
  repeats will see far less. Measure your own traffic with `/v1/metrics` before claiming
  savings.
- A hit costs ~0.1-0.6 ms in-process (embedding + exact search over a small index). p95 is
  unchanged because the slowest 5% are still misses.
- Cost figures use `COST_PER_1K_INPUT_TOKENS` / `COST_PER_1K_OUTPUT_TOKENS` and the mock's
  ~4-chars-per-token estimate. They are **estimates**, not billing data.

## 2. Quality: threshold vs. wrong answers

```bash
python scripts/evaluate_threshold.py --show-pairs
```

Dataset: `scripts/data/semantic_pairs.json`, 36 hand-labelled pairs:

- **16 equivalent** - reusing the cached answer is correct ("What is Docker?" / "Can you
  explain what Docker is?")
- **12 related** - same topic, needs a different answer ("How do I reset my password?" /
  "How do I reset my password if I lost access to my email?")
- **8 unrelated** - different topics

For each pair, query B is the incoming request and query A the cached entry.
*Recall* = share of equivalent pairs served from cache. *Precision* = share of served pairs
that were actually equivalent. A false positive is a wrong answer delivered to a user.

Results with the default `hashing` embedder:

| threshold | recall (equivalent served) | precision | false positives (related / unrelated) |
|---:|---:|---:|---:|
| 0.70 | 75% | 71% | 5 (5 / 0) |
| 0.75 | 69% | 69% | 5 (5 / 0) |
| 0.80 | 62% | 77% | 3 (3 / 0) |
| 0.85 | 62% | 91% | 1 (1 / 0) |
| 0.90 | 56% | 100% | 0 (0 / 0) |
| 0.95 | 50% | 100% | 0 (0 / 0) |

Observations:

- At **0.90** (the default) there were no wrong answers on this dataset, at the price of only
  serving 56% of the paraphrases. This is a deliberate bias: a missed hit costs one LLM call;
  a false hit gives a user a wrong answer.
- The closest call was **"Convert Celsius to Fahrenheit" vs "Convert Fahrenheit to Celsius" at
  0.899** - opposite meanings, same words. Bag-of-words embeddings are nearly blind to word
  order; the bigram features are the only thing keeping it under 0.90.
- "What is machine learning?" vs "What is machine learning used for?" scores 0.816. At a
  threshold of 0.80 it would be a false hit (the demo shows this with `--threshold 0.8`).
- Unrelated pairs never came close (all < 0.05). The danger zone is always **related**
  queries, which is exactly where similarity-based caching is weakest.
- Synonyms are invisible to the lexical embedder: "What is a car?" vs "Describe an
  automobile" scores 0.0, and "Who wrote Hamlet?" vs "Who is the author of Hamlet?" 0.48.
  A neural embedder (`EMBEDDING_PROVIDER=sentence-transformers`) should raise recall on these;
  re-run this script with it and re-tune the threshold, because score distributions differ
  between models. *That run is not included here: the model download was not possible in the
  environment that produced these numbers.*

36 pairs is a sanity check, not a statistically meaningful benchmark. Before production use,
build a labelled set from your own traffic.

## The core risk, stated plainly

**Semantic similarity is not identical intent.** A response can be incorrectly reused when
two queries are close in embedding space but need different answers - negations, swapped
arguments, an added constraint ("...if I lost access to my email?"), a different entity
("capital of France" vs "capital of Germany"), or time-sensitive questions. Mitigations in
this project:

- Configurable, conservative default threshold (0.90), documented as application-specific.
- Context key: model, system prompt, temperature, max_tokens and `namespace` must match.
- `use_cache=false` per request for anything that must be fresh or exact.
- High-temperature requests bypass the cache.
- TTL so stale answers age out.
- `POST /v1/cache/search` to inspect near-matches when tuning.

Mitigations **not** implemented (future work): a re-ranker / LLM judge that verifies a
candidate hit, entity or negation checks, and per-route thresholds.
