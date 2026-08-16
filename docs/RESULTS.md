# Results and interpretation

## Main reconstruction benchmark

The final ten-panel figures compare query-input, query-output, key-input, and key-output Gram reconstructions against both original effective-matrix SVD and support-projected SVD, plus learned functional low-rank replacements trained for 5,000 and 10,000 optimizer steps. The support projection uses the complete non-null query-input and key-input Gram eigenspaces before applying the rank $K$ SVD truncation. Cumulative projected-SVD power remains recorded in the generated tables but is not plotted.

Mean task-level $K_{95}$ is 18.5 for support-projected SVD, 18.9 for query-output Gram, 20.2 for either trained horizon, 20.3 for query-input Gram, 37.9 for both original SVD of $M$ and key-output Gram, and 57.7 for key-input Gram. Query-output ties support-projected SVD on seven tasks and trails it by only one or two ranks on the remaining three. The gap between original and support-projected SVD shows that much of the original matrix's disadvantage arises from exact input-null directions rather than task relevance within the active support.

The support-projected SVD is a post-hoc fairness baseline. Its query support ranks are 8, 33, 40, or 65 by task, and its key support rank is 104 throughout. At full supported rank it preserves the raw teacher accuracy exactly across all 20 seeds and ten tasks; the largest held-out score discrepancy from floating-point evaluation is $7.63\times10^{-6}$.

Training for 10,000 rather than 5,000 steps reduced the selected normalized logit loss in 1,482 of 2,000 seed/task/rank conditions and improved mean accuracy by 0.00339. It did not change mean $K_{95}$ (20.2) or mean $K_{99}$ (25.0), so the longer horizon modestly improves fit without changing the overall behavioral-rank conclusion.

## Task-transfer control

The average full-model test accuracy was 0.942. At rank $K=24$:

| Method | Matched source | Mean mismatched source | Difference |
|---|---:|---:|---:|
| Query-input Gram | 0.890 | 0.407 | 0.483 |
| Query-output Gram | 0.902 | 0.444 | 0.458 |
| Key-input Gram | 0.488 | 0.487 | 0.001 |
| Key-output Gram | 0.523 | 0.523 | 0.000 |

At $K=32$, query-input reached 0.938 and query-output reached 0.942 when matched, while mismatched-source performance remained 0.463 and 0.507, respectively. The positive query-side matched advantage occurred across all ten evaluation tasks.

The advantage peaks at intermediate rank and must vanish at $K=128$, where every complete projector is the identity. The result therefore concerns the ordering and efficiency of low-dimensional directions, not different full-rank maps.

## Qualified conclusion

Matching the explicit task one-hot does not rescue transfer from an incorrect source-task query distribution. This supports the claim that the query-input Gram—and its propagated query-output counterpart—captures task-specific low-dimensional structure beyond the explicit task-code coordinates.

The key/context-side Grams show no corresponding specificity, consistent with the tasks largely sharing context statistics. The two query-side results are algebraically related through $G_{M^\top q}=M^\top G_qM$, so they should be treated as two views of the same query-dependent structure rather than independent discoveries.

These experiments establish behavioral sufficiency of selected low-rank subspaces in this controlled synthetic model. They do not establish unique parameter recovery, causal necessity of individual eigenvectors, or immediate generalization to natural-language transformers.
