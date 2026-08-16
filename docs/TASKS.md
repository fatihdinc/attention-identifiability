# Synthetic tasks

Each example contains 64 context records and one query. Latent records are 128-dimensional and contain two 32-dimensional keys (`A` and `B`), a 32-way label, an 8-way category, a scalar priority, a 10-way task code, a constant coordinate, and reserved coordinates. Independent seed-specific orthogonal transforms mix context and query coordinates before the model sees them.

| # | Task | Required operation |
|---|---|---|
| 1 | Exact A lookup | Retrieve the label of the record whose A-key exactly matches the query A-key. |
| 2 | Noisy A lookup | Retrieve using an A-query correlated with the target A-key by the locked coefficient `rho=0.9`. |
| 3 | Partial A lookup | Retrieve from an A-query in which only 16 of 32 coordinates are observed. |
| 4 | B lookup | Retrieve by an exact match in the independent B-key space. |
| 5 | Two-key lookup | Identify the unique record matching both A and B; separate distractors match only A or only B. |
| 6 | Category-filtered lookup | Match A while using the query category to reject an A-matched distractor from another category. |
| 7 | Highest priority | Among records in the queried category, output the label with maximum priority. |
| 8 | Lowest priority | Among records in the queried category, output the label with minimum priority. |
| 9 | Category majority vote | Within the queried category, output the majority label among seven relevant records. |
| 10 | Three-item vote | Attend to three A-key voters encoded by their aggregate query and output their majority label in the presence of correlated distractors. |

The generator is implemented in `src/identifiability_llm/ten_task_attention.py`. All examples are produced online and deterministically; the repository does not require an external dataset.

