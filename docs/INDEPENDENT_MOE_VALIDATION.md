# Independent MoE validation report guide

The unified GitHub Actions workflow splits the collected Persian corpus **by whole document before tokenization**. It trains on the training partition and evaluates the one-layer checkpoint on held-out documents. The split manifest records document hashes and verifies zero overlap.

The validation JSON reports:
- trained MoE MSE/RMSE, relative L2 error, cosine similarity, normalized MSE, and improvement against a zero-output baseline;
- a random-router baseline using the same channel-sharded experts;
- repeated original FFN vs trained MoE forward-only timing on identical held-out token vectors, including routing/dispatch overhead;
- process RSS and peak RSS (plus CUDA peak allocation if used);
- actual parameter counts from loaded text-backbone modules and a clearly labelled hypothetical all-layer MoE estimate.

Interpretation cautions:
- This is still a **one-FFN research checkpoint**, not a converted full language model.
- The all-layer active-parameter figure is an architecture-derived estimate and does not prove the full model retains reasoning or Persian generation quality.
- The full vocabulary logits projection is reported separately because it can dominate compute even when the transformer body is sparse.
- Process peak RSS is process-wide, not isolated memory for a single layer.
- A 64-document run is a small validation; repeat with larger, diverse held-out corpora before making strong quality claims.


## Matched active-budget screening

The unified pipeline also runs `scripts/sweep_moe_budgets.py` on the same whole-document train/evaluation split. It evaluates 8/16/32/48 experts with top-2 routing and a bounded, deterministic activation sample to keep CPU time finite. The JSON records the parameter budget for each variant, held-out layer-output imitation metrics, and single-layer forward throughput. It also reports the non-FFN core estimate, which indicates whether 450M/400M can be reached by changing expert count alone.

This sweep is deliberately a screening step. Since it predicts the all-layer budget from a one-layer trained experiment and measures only one FFN's output imitation, it cannot establish full-model Persian quality, reasoning retention, end-to-end token speed, or Android RAM. A candidate must pass broader language-level tests before any architecture is selected.
