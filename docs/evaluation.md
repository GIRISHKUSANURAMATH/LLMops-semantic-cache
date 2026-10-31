# Evaluation

This document answers three questions, and keeps them separate:

1. **What kind of matching is happening?** Exact-match vs lexical similarity vs neural similarity.
2. **Performance:** how many LLM calls and how much latency does the cache save, and *which kind
   of matching* produced those savings?
3. **Quality:** at a given threshold, how often does the cache serve a **wrong** answer?

Every number below was produced by running the scripts in `scripts/`. None are hand-written.
Re-run them to reproduce; they are deterministic (mock LLM, seeded workload).

**Environment for the numbers below:** Python 3.13.18, Linux x86_64 (Ubuntu 24.04 container),
2 vCPU Intel Xeon @ 2.10 GHz, 7 GB RAM, 2026-10-31. Not yet re-run on Windows; see
`windows_verification.log` after running `verify_windows.ps1`.

---

## 1. Three kinds of matching

| | Exact-match caching | Lexical similarity (default) | Neural semantic similarity (optional) |
|---|---|---|---|
| What "same" means | Same text after normalisation (case, whitespace, trailing punctuation) | Embedding of the **words** is close: shared content words, stems, word pairs, character trigrams | Embedding of the **meaning** is close (learned from data) |
| In this project | Happens automatically: identical normalised text has cosine similarity 1.0 | `EMBEDDING_PROVIDER=hashing` | `EMBEDDING_PROVIDER=sentence-transformers` |
| "What is ML?" / "Explain ML to me" | miss | **hit** (framing words like "explain", "what", "is" are ignored) | hit |
| "What is a car?" / "Describe an automobile" | miss | **miss** (0.0: no shared words) | expected hit (not yet measured: see section 4) |
| "Celsius to Fahrenheit" / "Fahrenheit to Celsius" | miss | 0.899: nearly a **wrong** hit (word order mostly ignored) | model-dependent |
| Needs | nothing | nothing (pure NumPy, offline, deterministic) | PyTorch + a ~90 MB model download once |

**The default is lexical, not semantic in the full sense.** It exists so that the whole project,
including all tests and CI, runs offline with zero setup and gives identical numbers on every
machine. It is a reasonable baseline for "same question, different filler words", and it
fails on synonyms and paraphrases that share no words. Do not describe the default
configuration as "understanding meaning".

How the lexical embedder works (`src/llmops_cache/embeddings/hashing.py`): lower-case and
tokenise, drop stop-words and question-framing words ("what", "explain", "describe", "please"...),
crudely stem, then hash each content word (weight 1.0), adjacent word pair (0.5) and character
trigram (0.25) into a 512-dimensional signed vector and L2-normalise. Cosine similarity of
two such vectors measures weighted word overlap.

---

## 2. Performance benchmark

```bash
python scripts/benchmark.py
```

### Method

- Workload: 200 requests drawn (seed 42) with a Zipf-like distribution over the topics in
  `scripts/data/semantic_pairs.json`. Each topic has one or more phrasings; equivalent
  phrasings share a topic. A few topics are asked often, many rarely.
- Every request carries a hidden **topic label** (which answer it really needs), so each cache
  hit can be checked for correctness.
- The mock LLM sleeps a fixed 50 ms per call (stands in for network + generation time).
- Three strategies on the same stream: no cache (measured), exact-match cache (computed: one
  LLM call per distinct normalised phrasing), semantic cache (measured).

### Results (default lexical embedder, threshold 0.90)

Workload composition:

| requests | distinct phrasings | exact repeats | reworded (topic seen, new wording) | new topic |
|---:|---:|---:|---:|---:|
| 200 | 45 | 155 (77.5%) | 10 (5.0%) | 35 |

LLM calls:

| strategy | LLM calls | calls avoided vs no cache |
|---|---:|---:|
| no cache | 200 | 0 |
| exact-match cache (one call per distinct phrasing) | 45 | 155 |
| semantic cache (measured) | 40 | 160 |

Where the semantic cache's hits came from:

| outcome | count | share of requests |
|---|---:|---:|
| exact-repeat hits (an exact-match cache gets these too) | 155 | 77.5% |
| reworded hits: new wording, correct answer (**the semantic benefit**) | 5 | 2.5% |
| false-positive hits (wrong answer served) | 0 | 0.0% |
| first-time misses (unavoidable) | 35 | 17.5% |
| missed rewordings (same topic, similarity below threshold) | 5 | 2.5% |
| **overall cache hit rate, including exact repeats** | 160 | 80.0% |

Latency:

| metric | no cache | semantic cache |
|---|---:|---:|
| mean (ms) | 50.53 | 10.27 |
| p50 (ms) | 50.38 | 0.15 |
| p95 (ms) | 50.54 | 50.77 |
| total wall time (s) | 10.13 | 2.06 |

Estimated tokens avoided: 4,948; estimated cost saved: $0.002467 (configured per-1K-token
prices x the mock's ~4-characters-per-token estimate; **not billing data**).

### How to read this honestly

- **80% is the overall cache hit rate in this benchmark, including exact repeats. It is not a
  semantic hit rate.** 155 of the 160 hits would also have been served by a plain exact-match
  cache.
- The semantic layer's own contribution here is **5 extra hits (2.5% of requests)**: 45 LLM
  calls with exact matching versus 40 with semantic matching.
- Of the 10 genuinely reworded requests, the lexical embedder served 5 correctly and missed 5.
- A hit costs ~0.1-0.6 ms in-process. p95 is unchanged because the slowest 5% are still misses.
- The hit rate is a property of the **workload**. This one is deliberately repetitive. Measure
  your own traffic (`GET /v1/metrics`) before claiming any savings.

An earlier version of this document reported "83% hit rate, 30 paraphrase hits". That
benchmark derived topics from the dataset in a way that put one phrasing ("What is
photosynthesis?") under two topics and classified hits by the cached entry's wording rather
than the request's history. Both bugs are fixed; the numbers above supersede it.

### Threshold sweep on the same workload

| threshold | LLM calls | exact hits | reworded hits | false positives | missed rewordings |
|---:|---:|---:|---:|---:|---:|
| 0.80 | 36 | 147 | 5 | **12** | 4 |
| 0.85 | 39 | 155 | 6 | 0 | 4 |
| 0.90 | 40 | 155 | 5 | 0 | 5 |
| 0.95 | 40 | 155 | 5 | 0 | 5 |

At 0.80 the cache saves 4 more LLM calls than at 0.90 and serves **12 wrong answers** to do it.
Examples from the 0.80 run (`python scripts/benchmark.py --threshold 0.8`):

- asked "How do I install numpy?" -> served the answer cached for "How do I install numpy on
  macOS?" (similarity 0.817)
- asked "How do I delete a git branch?" -> served the answer cached for "How do I delete a
  remote git branch?" (0.819)

Note the compounding effect: once "How do I install numpy?" was wrongly matched, it was not
stored as its own entry, so **every later identical question got the same wrong answer** until
the entry expired. False positives are not one-off events.

---

## 3. Quality evaluation on labelled pairs

```bash
python scripts/evaluate_threshold.py --show-pairs
```

Dataset: `scripts/data/semantic_pairs.json`, 36 hand-labelled pairs:

- **16 equivalent:** reusing the answer is correct ("What is Docker?" / "Can you explain what
  Docker is?")
- **12 related:** same topic, needs a different answer ("How do I reset my password?" / "How do I
  reset my password if I lost access to my email?")
- **8 unrelated:** different topics

For each pair, query B is the incoming request and query A the cached entry.
*Recall* = share of equivalent pairs that would be served from cache. *Precision* = share of
served pairs that were really equivalent. A false positive is a wrong answer delivered to a user.

Default lexical embedder:

| threshold | recall (equivalent served) | precision | false positives (related / unrelated) |
|---:|---:|---:|---:|
| 0.70 | 75% | 71% | 5 (5 / 0) |
| 0.75 | 69% | 69% | 5 (5 / 0) |
| 0.80 | 62% | 77% | 3 (3 / 0) |
| 0.85 | 62% | 91% | 1 (1 / 0) |
| 0.90 | 56% | 100% | 0 (0 / 0) |
| 0.95 | 50% | 100% | 0 (0 / 0) |

Observations:

- Unrelated pairs never came close (all < 0.05). The danger zone is always **related** queries.
- Closest call at 0.90: "Convert Celsius to Fahrenheit" vs "Convert Fahrenheit to Celsius",
  **0.899**. Opposite meanings, same words. Only the word-pair (bigram) features keep it under
  the threshold.
- "What is machine learning?" vs "What is machine learning used for?" scores 0.816; at 0.80 it
  is served the wrong answer. `python examples/demo.py --threshold 0.8` shows exactly this.
- Synonyms are invisible to the lexical embedder: "What is a car?" vs "Describe an automobile"
  = **0.000**; "Who wrote Hamlet?" vs "Who is the author of Hamlet?" = 0.484.
- 36 pairs is a sanity check, not a statistically meaningful benchmark. Before production use,
  build a labelled set from your own traffic.

---

## 4. Neural embeddings: status

The `sentence-transformers` provider is implemented and unit-tested **with a fake model**
(`tests/unit/test_sentence_transformer.py`): model loading on a pinned device, dimension
detection, L2 normalisation, cosine similarity, clear errors for a missing package or a failed
model download, and an end-to-end cache hit on a synonym pair.

**It has not yet been run against the real model.** The environment that produced the numbers
above cannot reach the Hugging Face Hub, so the model could not be downloaded. Real similarity
values for "What is a car?" / "Describe an automobile." and "What is machine learning?" /
"Explain machine learning." are recorded by `verify_windows.ps1` (NEURAL EMBEDDING section of
`windows_verification.log`) when it runs on a machine with internet access. Until then:
**NEURAL MODEL: NOT VERIFIED.**

To evaluate it yourself:

```bash
python -m pip install -e ".[sentence-transformers]"
python scripts/neural_check.py
python scripts/evaluate_threshold.py --embedding-provider sentence-transformers
python scripts/benchmark.py --embedding-provider sentence-transformers --latency-ms 0
```

Expect a **different similarity scale**: neural models score unrelated sentences well above 0
and paraphrases below 1.0, so the threshold that is right for the lexical embedder (0.90) is
not automatically right for a neural one. Re-tune it with the scripts above.

---

## 5. Threshold trade-off

| Lower threshold | Higher threshold |
|---|---|
| More hits, fewer LLM calls | Fewer hits, more LLM calls |
| More wrong answers (false positives) | More missed rewordings (false negatives) |
| A false positive repeats for the whole TTL | A false negative costs one extra LLM call |

The asymmetry is why the default is conservative: a miss costs money, a false hit costs
correctness. **There is no universally correct threshold.** It depends on the embedding model,
the domain, and how costly a wrong answer is. 0.90 is a starting point for the lexical
embedder on this dataset only.

## 6. The core risk, stated plainly

**Semantic similarity is not identical intent.** A response can be incorrectly reused when two
queries are close in embedding space but need different answers: negations, swapped arguments
(Celsius/Fahrenheit), an added constraint ("...on macOS", "...if I lost access to my email"),
a different entity ("capital of France" vs "capital of Germany"), or time-sensitive questions.

Mitigations implemented:

- Conservative, configurable default threshold (0.90), documented as application-specific.
- Cache identity (context key): provider + endpoint, model, system prompt, temperature,
  max_tokens, embedding model and `namespace` must all match before similarity is even compared.
- `use_cache=false` per request for anything that must be fresh or exact.
- Requests above `CACHE_MAX_TEMPERATURE` bypass the cache.
- TTL so wrong or stale answers eventually age out; `DELETE /v1/cache/{id}` to evict one now.
- `POST /v1/cache/search` to inspect near-matches when tuning.

Not implemented (future work): verifying a candidate hit with a cross-encoder or small LLM
judge, entity/negation checks, per-route thresholds.

## 7. Limitations of this evaluation

- Synthetic workload and a 36-pair dataset written by the author; both are small and biased
  toward short FAQ-style questions.
- Mock LLM with fixed latency; real LLM latency varies widely, so real savings in milliseconds
  will differ.
- Token and cost figures are estimates.
- Single process, single machine; no load testing.
- Neural results not yet measured (section 4).
