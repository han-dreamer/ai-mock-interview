# RAG Design Notes

This project uses RAG for interview question planning, even with a small initial
question bank, because the question bank is treated as an expandable and
auditable knowledge source rather than a static prompt appendix.

Current corpus after expansion:

- 202 parent questions / knowledge items
- 2,523 parent-child retrieval chunks
- Coverage includes Python, system design, machine learning, AI trends, AI application engineering, advanced RAG patterns, computer networks, operating systems, databases, Redis, and message queues

Embedding mode:

- The system uses the configured remote OpenAI-compatible embedding API.
- The indexing script embeds all chunks before resetting the Chroma collection, so a provider failure will not wipe an existing index.

## Offline Indexing

The question bank is structured JSON, so the indexer uses structure-aware
parent-child chunking instead of fixed-size text windows.

For every parent question, the offline script creates child chunks:

- `question_stem`: the interview question itself
- `question_summary`: question plus compact answer/follow-up context
- `answer_point`: one chunk per reference answer point
- `follow_up`: one chunk per follow-up direction
- `generated_query`: rule-generated interviewer-style search queries

Each child chunk gets a contextual header with category, difficulty, skill tags,
chunk type, and parent question. The vector store indexes child chunks, but each
chunk keeps the full parent question metadata for hydration after retrieval.

## Retrieval Flow

1. Build 3-5 deterministic retrieval queries from the interview state.
   - Practice mode uses high-weight JD skills.
   - Round 1 uses JD skills plus resume-project and resume-JD match signals.
   - Round 2 uses JD skills plus Round 1 weak areas, system design, and AI-agent/RAG breadth.
2. Run hybrid retrieval for each query.
   - Vector search captures semantic similarity.
   - BM25 captures exact framework, API, and technical term matches.
   - Reciprocal Rank Fusion combines vector and BM25 rankings.
3. Fuse multi-query child-chunk results.
   - Repeated chunk hits across queries receive higher scores.
   - Generated-query chunks help bridge JD/resume wording and question-bank wording.
4. Apply metadata-aware reranking.
   - Required JD skill overlap receives a bonus.
   - Resume matched/missing skills receive a smaller bonus.
   - Round 2 favors system-design and AI-trend categories.
   - Medium/hard references are favored for professional rounds.
5. Hydrate child chunks back to parent questions.
   - The planner sees full question context, not isolated fragments.
   - Source chunk ids/types are preserved for traceability.
6. Deduplicate and diversify final parent questions.
   - Category and difficulty caps keep the prompt balanced.
7. Pass retrieved references to the question planner with source ids.
   - The planner can attach `source_ids` and `source_categories` to generated questions.
   - Invalid source ids are removed before the plan enters the interview state.

## Why Not Prompt Stuffing

The current corpus is small, but prompt stuffing makes every planning call longer
and less controllable as the corpus grows. RAG keeps the prompt focused on the
most relevant references, supports category filtering later, and allows simple
offline evaluation with golden retrieval cases.

## Evaluation

`scripts/evaluate_retrieval.py` compares:

- vector-only retrieval
- hybrid retrieval
- multi-query hybrid retrieval
- parent-hydrated multi-query hybrid retrieval

Metrics:

- `Hit@5`: whether at least one expected source appears in the top 5
- `Recall@5`: fraction of expected sources retrieved in the top 5
- `MRR@10`: rank-sensitive score for the first expected source in the top 10

Expected ids are parent question ids. When child chunks are retrieved, evaluation
maps `chunk_id -> parent_id` before computing metrics.

Run after the vector store is initialized:

```bash
python -m scripts.init_vector_store --reset
python -m scripts.evaluate_retrieval
```
