# API reference

Interactive docs are served by the running app at `/docs` (Swagger UI) and `/redoc`; the raw
schema is at `/openapi.json`. Examples below are real responses from the server running with
the mock LLM (`MOCK_LLM_LATENCY_MS=800`).

If `API_KEYS` is set, every `/v1/*` route and `/metrics` require the header
`X-API-Key: <key>`. `/health` stays open for load-balancer probes.

All errors share one shape:

```json
{"error": "validation_error", "message": "body.message: String should have at least 1 character", "request_id": "43c09a827bf343d0"}
```

| Status | `error` | When |
|---|---|---|
| 401 | `unauthorized` | `API_KEYS` set and header missing/wrong |
| 404 | `not_found` | Unknown route or cache entry id |
| 413 | `payload_too_large` | Body > `MAX_REQUEST_BYTES` or message > `MAX_MESSAGE_CHARS` |
| 422 | `validation_error` | Malformed JSON, missing/extra fields, out-of-range values |
| 500 | `internal_error` | Unexpected server error (details only in logs) |
| 502 | `provider_unavailable` | LLM rejected the request (non-retryable) |
| 503 | `provider_unavailable` / `embedding_unavailable` | LLM timeout/5xx/429 |

Rate limiting (429) is **not** implemented in the service; put it in your API gateway.

---

## `GET /health`

```json
{"status":"ok","version":"0.1.0","llm_provider":"mock","embedding_provider":"hashing-v1-512","entries_count":0}
```

## `POST /v1/chat`

Request fields:

| Field | Type | Default | Notes |
|---|---|---|---|
| `message` | string | required | Non-blank, max `MAX_MESSAGE_CHARS` |
| `model` | string | `"default"` | `"default"` -> `LLM_DEFAULT_MODEL` |
| `system_prompt` | string \| null | null | Part of the cache key |
| `temperature` | float 0-2 | 0.0 | Part of the cache key; above `CACHE_MAX_TEMPERATURE` bypasses cache |
| `max_tokens` | int | 512 | Part of the cache key |
| `namespace` | string \| null | null | `[A-Za-z0-9_.:-]{1,64}`; isolates tenants / template versions |
| `use_cache` | bool | true | `false` forces an LLM call, nothing stored |
| `ttl_seconds` | int \| null | null | Per-entry TTL override |

**Cache MISS** (first request):

```bash
curl -s -X POST http://127.0.0.1:8000/v1/chat -H "Content-Type: application/json" -d "{\"message\": \"What is machine learning?\"}"
```

```json
{
  "response": "[mock:mock-1] This is a simulated answer to: What is machine learning? (response id aa133a3d)",
  "cache_hit": false,
  "source": "llm",
  "similarity": null,
  "latency_ms": 801.217,
  "model": "mock-1",
  "cache_entry_id": "e1dd2f3151a748d89061414123c2279f",
  "usage": {"input_tokens": 7, "output_tokens": 24},
  "estimated_cost_saved_usd": 0.0,
  "cache_bypass_reason": null
}
```

**Cache HIT** (different wording, same meaning):

```bash
curl -s -X POST http://127.0.0.1:8000/v1/chat -H "Content-Type: application/json" -d "{\"message\": \"Explain machine learning to me.\"}"
```

```json
{
  "response": "[mock:mock-1] This is a simulated answer to: What is machine learning? (response id aa133a3d)",
  "cache_hit": true,
  "source": "semantic_cache",
  "similarity": 1.0,
  "latency_ms": 0.602,
  "model": "mock-1",
  "cache_entry_id": "e1dd2f3151a748d89061414123c2279f",
  "usage": {"input_tokens": 7, "output_tokens": 24},
  "estimated_cost_saved_usd": 0.00001545,
  "cache_bypass_reason": null
}
```

(The hashing embedder scores this pair 1.0 because, after removing framing words like
"explain", "what", "is", "to", "me", both reduce to "machine learning".)

`source` is one of `llm`, `semantic_cache`, or `coalesced` (served by waiting on an identical
in-flight request). The chat response deliberately does **not** include the cached entry's
original query text: in a shared cache that would leak another user's prompt.

## `POST /v1/cache/search`

Shows what the cache *would* match, without calling the LLM or counting a hit. Useful for
threshold tuning. Accepts the same identity fields as `/v1/chat` plus optional `threshold`
and `top_k`. Returns stored query text, so protect it with `API_KEYS` in shared deployments.

```bash
curl -s -X POST http://127.0.0.1:8000/v1/cache/search -H "Content-Type: application/json" -d "{\"message\": \"What is machine learning used for?\"}"
```

```json
{"threshold":0.9,"matches":[{"entry_id":"e1dd2f3151a748d89061414123c2279f","query":"What is machine learning?","similarity":0.8165,"would_hit":false,"expired":false,"hit_count":1,"created_at":1793440826.0762491,"expires_at":1793444426.0762491}]}
```

## `GET /v1/cache/stats`

```json
{"entries_count":1,"max_entries":10000,"expired_entries":0,"total_hits_served":1,"similarity_threshold":0.9,"ttl_seconds":3600,"embedding_provider":"hashing-v1-512","embedding_dimensions":512,"llm_provider":"mock"}
```

## `DELETE /v1/cache` / `DELETE /v1/cache/expired` / `DELETE /v1/cache/{entry_id}`

Clear everything, purge only expired entries, or delete one entry (404 if unknown).

```json
{"removed":1}
```

## `GET /v1/metrics`

```json
{"requests_total":2,"cache_hits":1,"cache_misses":1,"coalesced_requests":0,"cache_bypassed":0,"errors":0,"hit_rate":0.5,"llm_calls":1,"llm_calls_avoided":1,"estimated_tokens_avoided":31,"estimated_cost_saved_usd":0.000015,"average_latency_ms":400.909,"average_hit_latency_ms":0.602,"average_miss_latency_ms":801.217,"entries_count":1,"note":"Cost and token figures are estimates based on configured per-token prices."}
```

`hit_rate = (hits + coalesced) / (hits + coalesced + misses)`; bypassed requests are excluded.
Counters are per process and reset on restart.

## `GET /metrics` (Prometheus)

Text exposition format. Series: `llmops_requests_total{outcome}`,
`llmops_request_latency_seconds{outcome}` (histogram), `llmops_llm_calls_total`,
`llmops_estimated_tokens_avoided_total`, `llmops_estimated_cost_saved_usd_total`,
`llmops_cache_entries`.

```
llmops_requests_total{outcome="miss"} 1.0
llmops_requests_total{outcome="hit"} 1.0
llmops_llm_calls_total 1.0
```
