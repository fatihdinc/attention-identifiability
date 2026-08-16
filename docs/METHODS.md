# Methods

## Effective attention-score matrix

The model computes

$$
q'=W_Qq,\qquad k'=W_Kx,
$$

and the unnormalized attention score

$$
s(q,x)=\gamma (W_Qq)^\top(W_Kx)=q^\top Mx,
\qquad M=\gamma W_Q^\top W_K.
$$

Changing $W_Q\mapsto AW_Q$ and $W_K\mapsto A^{-\top}W_K$ preserves $M$. Consequently, the experiment reconstructs $M$, rather than treating the two factors as separately identifiable.

## Four Gram constructions

Queries and context/key inputs are column vectors. Expectations are empirical averages over a task-specific calibration set.

### Query-input Gram

$$
G_q=\mathbb{E}[qq^\top].
$$

Let $U_{q,K}$ contain its leading $K$ eigenvectors and $P_{q,K}=U_{q,K}U_{q,K}^\top$. The reconstruction is

$$
\widehat M_{q\text{-in},K}=P_{q,K}M.
$$

### Query-output Gram

The query after propagation through the bilinear map is $M^\top q$:

$$
G_{M^\top q}=\mathbb{E}[(M^\top q)(M^\top q)^\top]
=M^\top G_qM.
$$

With leading-eigenspace projector $P_{M^\top q,K}$,

$$
\widehat M_{q\text{-out},K}=MP_{M^\top q,K}.
$$

### Key-input Gram

$$
G_x=\mathbb{E}[xx^\top],\qquad
\widehat M_{k\text{-in},K}=MP_{x,K}.
$$

### Key-output Gram

The key/context input after propagation is $Mx$:

$$
G_{Mx}=\mathbb{E}[(Mx)(Mx)^\top]=MG_xM^\top,
$$

and

$$
\widehat M_{k\text{-out},K}=P_{Mx,K}M.
$$

The code audits symmetry, positive semidefiniteness, projector idempotence, the two propagated-Gram identities, and the requested rank bound.

## Baselines

### Original and support-projected SVD

For $M=U\Sigma V^\top$, the original rank $K$ SVD baseline is

$$
\widehat M_{\mathrm{original},K}
=U_{:K}\Sigma_{:K}V_{:K}^\top.
$$

This is the optimal rank $K$ Frobenius approximation to the complete trained matrix, including its input-null directions.

Let $U_q$ and $U_x$ span the non-null eigenspaces of the query-input and key-input Grams, using the fixed cutoff $\lambda>10^{-10}\lambda_{\max}$. Define

$$
P_q=U_qU_q^\top,\qquad P_x=U_xU_x^\top,\qquad
M_{\mathrm{supp}}=P_qMP_x.
$$

Rather than factorizing a numerically lifted 128-by-128 matrix, the implementation computes

$$
B=U_q^\top M U_x=\widetilde U\Sigma\widetilde V^\top
$$

and lifts its rank $K$ truncation:

$$
\widehat M_{\mathrm{supp},K}
=U_q\widetilde U_{:K}\Sigma_{:K}\widetilde V_{:K}^\top U_x^\top.
$$

The generated tables also retain cumulative power from the supported spectrum:

$$
C_{\mathrm{supp}}(K)
=\frac{\sum_{i\le K}\sigma_i(B)^2}{\sum_i\sigma_i(B)^2}.
$$

The non-null support ranks are unchanged for relative cutoffs from $10^{-8}$ through $10^{-12}$. The query support—and, empirically in these runs, the supported operator—has rank 8, 33, 40, or 65 depending on task; the key support has rank 104 for every task. This baseline was added post hoc after the exact null input directions were identified. The original SVD of $M$ remains in the locked parent results and is shown alongside the support-projected curve.

The final main figures show both SVD accuracy curves. The cumulative-power diagnostic remains available in the generated tables but is not plotted.

### Learned functional low rank

A rank $K$ factorization $A_KB_K$ replaces $M$ while the teacher's value, output, and readout paths remain frozen. Optimization minimizes teacher-logit mean-squared error on deterministic calibration examples; ground-truth class labels are not used for this optimization. The same nested trajectory is evaluated at 5,000 and 10,000 optimizer steps.

## Evaluation

Every reconstructed matrix is inserted into the exact direct $M$ forward pass. The primary metric is held-out classification accuracy on 4,096 examples per seed and task. Secondary metrics include accuracy retention, attention KL divergence, centered score MSE, matrix error, and numerical rank. Matrix error for the support-projected SVD is measured relative to $M_{\mathrm{supp}}$, while a separate held-out audit requires the full support projection to preserve the scores and accuracy of raw $M$.

The analytic rank grid is $K=0,1,\ldots,50,60,70,80,90,100,110,120,128$. Learned ranks are $0,1,2,4,8,16,24,32,48,64,96,128$.

## Task-transfer control

For every ordered source/evaluation pair $(t,s)$:

1. Generate the locked calibration examples for source task $t$.
2. Replace only the query's task-code block with task $s$'s one-hot code while computing the four Grams.
3. Build each nested rank $K$ projector from those Grams.
4. Evaluate on the ordinary, untouched held-out test examples from task $s$.

Targets, test queries, test task codes, contexts, and labels are never replaced. The diagonal $t=s$ is audited against the parent reconstruction experiment.
