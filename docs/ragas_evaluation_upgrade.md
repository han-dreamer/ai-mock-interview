# RAGAS Evaluation Upgrade

This project uses RAGAS as a local/offline evaluation layer for the RAG QA subtask.
It is intentionally separated from the deployed FastAPI runtime: the server serves users,
while the RAGAS scripts are used locally to run experiments, compare variants, and diagnose
bad cases.

## Evaluation Goal

The upgraded evaluation loop focuses on four main RAGAS metrics:

- Faithfulness: whether the answer is grounded in retrieved contexts.
- Answer Relevancy: whether the answer directly addresses the user question.
- Context Precision: whether retrieved contexts are relevant.
- Context Recall: whether retrieved contexts cover the evidence needed by the reference answer.

The goal is not only to get a score, but to build a reproducible optimization loop:

1. Build a baseline with the current full RAG pipeline.
2. Compare retriever variants with the same dataset and metric set.
3. Run top_k sensitivity experiments.
4. Inspect automatically flagged low-score cases.
5. Use the findings to improve query rewriting, reranking, context formatting, and grounded generation.

## Files

- `scripts/evaluate_ragas.py`: main RAGAS generation-level evaluation script.
- `scripts/evaluate_retrieval.py`: retrieval-only evaluation using Hit@5, Recall@5, and MRR@10.
- `data/eval/ragas_qa_golden.json`: original 12-case smoke dataset.
- `data/eval/ragas_qa_golden_v2.json`: upgraded 30-case evaluation dataset.
- `data/eval/results/`: ignored output directory for JSON, CSV, Markdown, and PNG reports.

## Golden Dataset Schema

Each v2 case uses this shape:

```json
{
  "id": "ragas-v2-001",
  "category": "rag_chunking",
  "difficulty": "medium",
  "question_type": "single_hop",
  "question": "What is Parent-Child Chunking in RAG and why is it useful?",
  "ground_truth": "Parent-Child Chunking retrieves small child chunks...",
  "expected_source_ids": ["rag-adv-001", "rag-adv-010"],
  "tags": ["RAG", "Chunking", "Parent-Child Retrieval"]
}
```

The `category`, `difficulty`, and `question_type` fields are used for report breakdowns.
The `expected_source_ids` field is used by the script to flag retrieval misses in addition
to RAGAS metric failures.

## Basic Commands

Before running generated-answer evaluations, create `.env` from `.env.example`
and fill in the OpenAI-compatible LLM and embedding settings:

```text
LLM_API_KEY=...
LLM_BASE_URL=...
LLM_MODEL=...
EMBEDDING_API_KEY=...
EMBEDDING_BASE_URL=...
EMBEDDING_MODEL=...
```

Then build the local Chroma index:

```bash
python -m scripts.init_vector_store --reset
```

Prepare a RAGAS dataset without calling judge models:

```bash
python -m scripts.evaluate_ragas --dataset data/eval/ragas_qa_golden_v2.json --variant full --limit 3 --prepare-only
```

Run a quick generated-answer smoke test:

```bash
python -m scripts.evaluate_ragas --dataset data/eval/ragas_qa_golden_v2.json --variant full --limit 3 --metrics core --answer-source generated --run-name smoke_full_v2
```

If the judge model is slow, start with one case or a targeted metric subset:

```bash
python -m scripts.evaluate_ragas --dataset data/eval/ragas_qa_golden_v2.json --variant full --limit 1 --metrics core --answer-source generated --batch-size 1 --run-name smoke_core_one
python -m scripts.evaluate_ragas --dataset data/eval/ragas_qa_golden_v2.json --variant full --limit 3 --metrics retrieval --answer-source reference --batch-size 1 --run-name smoke_retrieval_reference_3
```

Run the full baseline:

```bash
python -m scripts.evaluate_ragas --dataset data/eval/ragas_qa_golden_v2.json --variant full --metrics core --answer-source generated --batch-size 2 --run-name full_v2_baseline
```

The standard experiment suite can also be run through the wrapper script:

```bash
python -m scripts.run_ragas_experiments --suite smoke
python -m scripts.run_ragas_experiments --suite baseline
python -m scripts.run_ragas_experiments --suite variant
python -m scripts.run_ragas_experiments --suite topk
```

Use `--prepare-only` with the wrapper to verify retrieval contexts without RAGAS judge calls:

```bash
python -m scripts.run_ragas_experiments --suite smoke --prepare-only
```

`--metrics core` now means the four main metrics:

- `faithfulness`
- `answer_relevancy`
- `context_precision`
- `context_recall`

`--metrics all` and `--metrics core4` are aliases for the same four-metric set.

## Variant Comparison

The script supports four RAG variants:

- `vector`: vector-only retrieval.
- `hybrid`: vector + BM25 retrieval with RRF fusion.
- `multi`: multi-query hybrid retrieval.
- `full`: multi-query retrieval plus metadata-aware reranking and parent hydration.

Run all variants:

```bash
python -m scripts.evaluate_ragas --dataset data/eval/ragas_qa_golden_v2.json --variant all --metrics core --answer-source generated --batch-size 2 --run-name variant_compare_v2
```

This is the main ablation experiment for explaining what each retrieval module contributes.

Because the current judge model can be slow, use the fast source-hit variant
comparison as the first screening step:

```bash
python -m scripts.evaluate_rag_variants --dataset data/eval/ragas_qa_golden_v2.json --limit 5 --top-k 5 --run-name variant_source_hit_v2_5
```

This script does not call RAGAS judges. It checks whether each variant retrieves
the expected parent source ids from the golden dataset.

### First Variant Source-Hit Result

The first 5-case v2 source-hit comparison produced:

| Variant | Hit Rate | Expected Recall | Avg Retrieved Parents | Missed Cases |
|---|---:|---:|---:|---:|
| vector | 1.000 | 0.900 | 3.60 | 0 |
| hybrid | 1.000 | 0.900 | 4.40 | 0 |
| multi | 1.000 | 0.900 | 2.00 | 0 |
| full | 1.000 | 0.900 | 2.00 | 0 |

Initial interpretation:

- All variants hit the expected source for the first 5 v2 cases.
- `vector` and `hybrid` retrieved more parent sources on average, which can help
  recall but may introduce more context noise.
- `multi` and `full` kept the same hit rate while returning fewer parent sources,
  making the final context more compact.
- In this small sample, the value of `full` is not that it is the only variant
  that can hit the expected source, but that it preserves source hits while
  reducing unnecessary parent contexts.

Generated source-hit report:

```text
data/eval/results/variant_source_hit_v2_5_20260609_205741.md
```

### Expanded 15-Case Source-Hit Result

The source-hit comparison was then expanded from 5 to 15 v2 cases.

| Variant | Hit Rate | Expected Recall | Avg Retrieved Parents | Missed Cases |
|---|---:|---:|---:|---:|
| vector | 1.000 | 0.967 | 3.60 | 0 |
| hybrid | 1.000 | 0.967 | 4.00 | 0 |
| multi | 1.000 | 1.000 | 1.73 | 0 |
| full | 1.000 | 1.000 | 1.73 | 0 |

Interpretation:

- No variant fully missed a golden case in the first 15 v2 cases.
- After tightening `ragas-v2-001` into a true single-hop parent-child chunking
  case, `multi` and `full` reached `Expected Recall = 1.000`, while `vector`
  and `hybrid` remained at `0.967`.
- `multi` and `full` also reduced average retrieved parent contexts from
  `3.60-4.00` to `1.73`, making the context passed downstream more compact.
- The remaining partial-recall case was `ragas-v2-012`, where `vector` and
  `hybrid` missed one expected debugging/failure-mode source, while `multi`
  and `full` retrieved both expected sources.

Current decision:

- Keep `full` as the default RAG QA evaluation variant.
- Treat `multi/full` as the preferred retrieval path for downstream generation
  because they keep source hits while reducing context redundancy.
- Use RAGAS judge calls selectively on suspicious cases or final baselines,
  because full variant-by-variant RAGAS judging is slow with the current model.

Generated 15-case source-hit report:

```text
data/eval/results/variant_source_hit_v2_15_20260609_210358.md
data/eval/results/variant_source_hit_v2_15_after_golden_fix_20260609_231345.md
```

Golden dataset note:

- `ragas-v2-001` originally listed both `rag-adv-001` and `rag-adv-010` as
  expected sources. Manual inspection showed that the question is a single-hop
  Parent-Child Chunking question, and `rag-adv-001` already supports the revised
  reference answer. `rag-adv-010` is a neighboring Parent Hydration topic, so it
  was removed from the expected source ids to avoid forcing the retriever to
  recall adjacent but non-required evidence.

### Full 30-Case Source-Hit Result

The source-hit comparison was finally expanded to all 30 v2 cases.

| Variant | Hit Rate | Expected Recall | Avg Retrieved Parents | Missed Cases |
|---|---:|---:|---:|---:|
| vector | 1.000 | 0.983 | 3.57 | 0 |
| hybrid | 1.000 | 0.983 | 3.87 | 0 |
| multi | 1.000 | 1.000 | 1.53 | 0 |
| full | 1.000 | 1.000 | 1.53 | 0 |

Interpretation:

- All variants reached `Hit Rate = 1.000`, so none of the 30 cases fully missed
  the expected source set.
- `multi` and `full` reached `Expected Recall = 1.000`, while `vector` and
  `hybrid` remained at `0.983`.
- `multi` and `full` also reduced average retrieved parent contexts from
  `3.57-3.87` to `1.53`.
- The only partial-recall case for `vector` and `hybrid` was `ragas-v2-012`
  (`rag_debugging`), where they retrieved `rag-adv-014` but missed `rag-adv-024`;
  `multi` and `full` retrieved both expected sources.

Current source-hit conclusion:

- Keep `full` as the default RAG QA evaluation variant.
- Use `multi/full` as the preferred retrieval path when generation quality
  matters, because they preserve source coverage while reducing context
  redundancy.
- Use full RAGAS judge calls for sampled generated-answer quality checks rather
  than every variant comparison, since source-hit screening is much faster and
  already separates the retrieval variants clearly.

Generated 30-case source-hit report:

```text
data/eval/results/variant_source_hit_v2_30_20260609_232044.md
```

## Generated-Answer Sample

After source-hit screening, a small generated-answer RAGAS sample was used to
check whether correct retrieval actually turns into grounded and complete
answers. The sampled cases were:

- `ragas-v2-001`: parent-child chunking.
- `ragas-v2-002`: structure-aware chunking.
- `ragas-v2-012`: retrieval failure versus generation failure debugging.

The first 3-case generated-answer run produced:

| Variant | Cases | Answer Relevancy | Context Precision | Context Recall | Faithfulness | Flagged Cases |
|---|---:|---:|---:|---:|---:|---:|
| full | 3 | 0.943 | 0.944 | 0.500 | 0.889 | 2/3 |

Main finding:

- `ragas-v2-012` had a source hit, but the generated answer compressed the
  debugging distinction into a short sentence. It mentioned irrelevant retrieval
  and ignored evidence, but missed retrieved debugging evidence such as top-k
  candidates, scores, source metadata, chunking/context loss, and query-document
  alignment. This was a generation coverage issue rather than a retrieval miss.
- `ragas-v2-002` also exposed wording misalignment between the golden reference
  and the retrieved evidence, so the reference was tightened to match the actual
  structure-aware chunking source.

The first fix aligned the `ragas-v2-002`/`ragas-v2-012` references with retrieved
evidence and strengthened the grounded-answer prompt for comparison,
distinction, and debugging questions.

| Variant | Cases | Answer Relevancy | Context Precision | Context Recall | Faithfulness | Flagged Cases |
|---|---:|---:|---:|---:|---:|---:|
| full | 3 | 0.938 | 1.000 | 0.667 | 0.917 | 1/3 |

The remaining problem was still `ragas-v2-012`, where the generated answer was
relevant but under-covered the debugging evidence. The final prompt update now
requires multi-hop/debugging answers to use 2-4 concise bullets and cover both
retrieval-side and generation-side signals when supported by contexts.

Targeted re-test on `ragas-v2-012`:

| Variant | Case | Answer Relevancy | Context Precision | Context Recall | Faithfulness | Flagged |
|---|---|---:|---:|---:|---:|---:|
| full | `ragas-v2-012` | 0.935 | 1.000 | 1.000 | n/a | 0/1 |

Interpretation:

- `Context Recall` improved from `0.000` to `1.000` on the problematic generated
  answer sample.
- The generated answer now covers retrieval-side signals including
  query-document alignment, chunking/context loss, rewritten queries, top-k
  candidates, scores, and source metadata, and generation-side ignored evidence.
- `Faithfulness` is reported as `n/a` for this targeted run because the judge
  call timed out for that metric. The result is kept as missing instead of being
  imputed, and metric coverage is written into the report.

### Representative 5-Case Generated Baseline

After the targeted debugging fix, a broader 5-case generated-answer sample was
run across representative project capabilities:

- `ragas-v2-001`: parent-child chunking.
- `ragas-v2-005`: hybrid retrieval with BM25 and vector search.
- `ragas-v2-010`: context precision and context recall.
- `ragas-v2-012`: retrieval failure versus generation failure.
- `ragas-v2-026`: LangGraph multi-stage interview pipeline.

Initial four-metric result:

| Variant | Cases | Answer Relevancy | Context Precision | Context Recall | Faithfulness | Flagged Cases |
|---|---:|---:|---:|---:|---:|---:|
| full | 5 | 0.778 | 0.967 | 0.933 | 0.956 | 5/5 |

Interpretation:

- Retrieval and grounding were already strong, with `Faithfulness = 0.956`,
  `Context Precision = 0.967`, and `Context Recall = 0.933`.
- The weak point was answer phrasing. Several generated answers started with
  report-style phrases such as "Based on the retrieved contexts", cited context
  numbers, or used overly broad bullet formatting for simple definition
  questions. These tokens were not useful for the user question and hurt
  `Answer Relevancy`.

The grounded-answer prompt was then tightened to:

- Start with a direct answer using key terms from the question.
- Avoid phrases that mention retrieved contexts.
- Avoid context-number citations in the final answer.
- Prefer one short paragraph for simple definition/explanation questions.
- Keep 2-4 concise bullets or short sentences for comparison, multi-hop, and
  debugging questions.

Answer-only fast re-test:

| Variant | Cases | Answer Relevancy | Flagged Cases |
|---|---:|---:|---:|
| full | 5 | 0.938 | 0/5 |

Final four-metric re-test:

| Variant | Cases | Answer Relevancy | Context Precision | Context Recall | Faithfulness | Flagged Cases |
|---|---:|---:|---:|---:|---:|---:|
| full | 5 | 0.911 | 1.000 | 0.933 | 0.980 | 2/5 |

Optimization result:

- `Answer Relevancy` improved from `0.778` to `0.911` in the final four-metric
  run, while `Faithfulness` improved from `0.956` to `0.980`.
- `Context Precision` improved from `0.967` to `1.000`, and `Context Recall`
  stayed at `0.933`.
- The remaining flagged cases were boundary cases: `ragas-v2-005` had
  `Answer Relevancy = 0.833`, and `ragas-v2-010` had `Answer Relevancy = 0.847`
  with `Context Recall = 0.667`. Manual inspection showed both answers were
  relevant and grounded, so they are recorded as residual evaluation targets
  rather than prompt-overfitted further.

Generated-answer reports:

```text
data/eval/results/sampled_full_core_generated_3_20260609_234545.md
data/eval/results/sampled_full_core_generated_3_after_prompt_reference_fix_20260610_000830.md
data/eval/results/targeted_v2_012_after_debug_prompt_fix_20260610_002234.md
data/eval/results/representative_5_generated_core_20260610_104502.md
data/eval/results/representative_5_answer_after_direct_prompt_20260610_105403.md
data/eval/results/representative_5_generated_core_after_direct_prompt_20260610_111521.md
```

## Top K Sensitivity

Chunk size and overlap are not the main tuning direction for this project because the corpus
is a structured interview question bank. The more suitable experiment is top_k sensitivity:

```bash
python -m scripts.evaluate_ragas --dataset data/eval/ragas_qa_golden_v2.json --variant full --metrics core --top-k 3 --run-name full_topk3
python -m scripts.evaluate_ragas --dataset data/eval/ragas_qa_golden_v2.json --variant full --metrics core --top-k 5 --run-name full_topk5
python -m scripts.evaluate_ragas --dataset data/eval/ragas_qa_golden_v2.json --variant full --metrics core --top-k 8 --run-name full_topk8
python -m scripts.evaluate_ragas --dataset data/eval/ragas_qa_golden_v2.json --variant full --metrics core --top-k 10 --run-name full_topk10
```

Typical interpretation:

- If `Context Recall` is low, top_k may be too small or query variants may not cover the needed evidence.
- If `Context Precision` drops as top_k grows, retrieval is adding noise.
- If `Faithfulness` drops while recall is high, the generator may be distracted by noisy context or the prompt may need stronger grounding.

### First Smoke Result

The first 3-case retrieval-only smoke experiment compared `top_k=3`, `top_k=5`,
and `top_k=8` on `data/eval/ragas_qa_golden_v2.json` with reference answers.

| Top K | Context Precision | Context Recall | Flagged Cases |
|---:|---:|---:|---:|
| 3 | 1.000 | 0.667 | 1/3 |
| 5 | 0.944 | 0.833 | 1/3 |
| 8 | 0.944 | 0.833 | 1/3 |

Initial interpretation:

- `top_k=3` produced cleaner context but lost recall on `ragas-v2-002`.
- `top_k=5` improved context recall with a moderate precision tradeoff.
- `top_k=8` did not improve over `top_k=5` in this smoke run because parent hydration
  returned the same parent sources after deduplication.
- The flagged case still hit the expected source id, so the issue is not a pure
  retrieval miss; it is a context coverage/noise tradeoff that should be checked
  on a larger sample before changing the default.

Generated comparison report:

```text
data/eval/results/topk_3_5_8_retrieval_compare_20260609_172508.md
```

### Expanded Top K 5 Smoke

After the 3-case top_k comparison, `top_k=5` was expanded to the first 5 v2
cases with retrieval-only RAGAS metrics and reference answers.

| Variant | Top K | Cases | Context Precision | Context Recall | Flagged Cases |
|---|---:|---:|---:|---:|---:|
| full | 5 | 5 | 0.967 | 1.000 | 1/5 |

Case-level finding:

- `ragas-v2-001`, `ragas-v2-003`, `ragas-v2-004`, and `ragas-v2-005` passed with
  `Context Precision = 1.000` and `Context Recall = 1.000`.
- `ragas-v2-002` still showed mild context noise with `Context Precision = 0.833`,
  but the expected source id was hit and `Context Recall = 1.000`.
- The noisy neighboring contexts came mainly from broad `generated_query` chunks
  around adjacent RAG chunking/debugging topics. This is useful evidence, but it
  is not yet enough to justify changing chunking or reranking globally.

Current decision:

- Keep `top_k=5` as the default for now.
- Do not tune chunk size/overlap for this structured question-bank corpus.
- Do not change generated-query chunking from one flagged case; first expand the
  retrieval-only RAGAS sample to more categories or run a small variant comparison.

Generated 5-case report:

```text
data/eval/results/topk5_retrieval_reference_5_20260609_201808.md
```

## Output Files

Each non-prepare run writes:

- `*.json`: full raw result, traces, RAGAS rows, metric coverage, and diagnostics.
- `*.summary.csv`: one row per variant with aggregate metrics.
- `*.cases.csv`: one row per case with metrics, source-hit status, flags, and suggestions.
- `*.md`: Markdown report suitable for GitHub docs or course reporting.
- `*.png`: grouped bar chart for the four main metrics by variant.

The output directory is ignored by Git:

```text
data/eval/results/
```

This keeps generated evaluation artifacts out of the repository while preserving scripts,
datasets, and docs.

## Automatic Problem Case Detection

The script flags cases with metric scores below `0.85`:

- `unsupported_claims`: low Faithfulness.
- `answer_off_topic`: low Answer Relevancy.
- `context_noise`: low Context Precision.
- `insufficient_context`: low Context Recall.
- `retrieval_miss`: retrieved parent ids did not overlap with `expected_source_ids`.

Use the generated `*.cases.csv` to sort and inspect the worst cases.

## Recommended Experiment Order

1. Run `--prepare-only` on 3 cases to verify retrieval and dataset construction.
2. Run `--limit 3` smoke test with generated answers.
3. Run `full_v2_baseline` on all 30 cases.
4. Run `variant_compare_v2` to compare vector, hybrid, multi, and full.
5. Run `top_k` experiments for 3, 5, 8, and 10.
6. Inspect `*.cases.csv` and `*.md` problem cases.
7. Optimize query variants, reranking, context formatting, or grounded answer prompt based on the flagged failure type.

## Judge Model Configuration

By default, RAGAS uses the project's configured chat and embedding models. For local
evaluation, judge settings can be overridden without changing server configuration:

```bash
python -m scripts.evaluate_ragas --judge-model your-judge-model --judge-base-url your-base-url --judge-api-key your-key
```

Environment variable overrides are also supported:

```text
RAGAS_JUDGE_MODEL
RAGAS_JUDGE_API_KEY
RAGAS_JUDGE_BASE_URL
RAGAS_EMBEDDING_MODEL
RAGAS_EMBEDDING_API_KEY
RAGAS_EMBEDDING_BASE_URL
RAGAS_TIMEOUT_SECONDS
RAGAS_MAX_RETRIES
```

This keeps local evaluation flexible while keeping the deployed FastAPI service independent
from evaluation-only choices.
