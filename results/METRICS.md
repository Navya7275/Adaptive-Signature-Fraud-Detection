# Results

All numbers below are measured on **held-out writers only** — the
writer-disjoint 20% validation split of the training corpus. No writer
in these tables appears anywhere in training, so this reflects
generalization to people the model has never seen.

Per-dataset rows are *subsets of that same split*, not fresh splits of
each dataset. (Re-splitting an individual dataset leaks training writers
into its validation set and inflates AUC to 0.98+ — avoided here.)

## Verification performance

| Evaluation set | Writers | Pairs | ROC-AUC | EER | Accuracy |
|---|---:|---:|---:|---:|---:|
| All held-out writers | 201 | 12,060 | 0.884 | 18.4% | 82.1% |
| CEDAR (English) | 16 | 960 | 0.926 | 14.4% | 85.9% |
| SYNTH (generated) | 27 | 1,620 | 0.878 | 21.5% | 81.9% |
| BHSig260 (Hindi) | 31 | 1,860 | 0.955 | 11.0% | 90.1% |
| sign_data (real) | 127 | 7,620 | 0.879 | 19.9% | 80.3% |

## Score separation

| Evaluation set | Genuine similarity | Forged similarity | Gap |
|---|---:|---:|---:|
| All held-out writers | 0.939 | 0.800 | **0.139** |
| CEDAR (English) | 0.940 | 0.824 | **0.116** |
| SYNTH (generated) | 0.982 | 0.802 | **0.180** |
| BHSig260 (Hindi) | 0.954 | 0.761 | **0.194** |
| sign_data (real) | 0.927 | 0.807 | **0.120** |

## Operating point (deployed threshold = 0.9 cosine)

| Evaluation set | False Accept Rate | False Reject Rate |
|---|---:|---:|
| All held-out writers | 21.1% | 14.9% |
| CEDAR (English) | 16.0% | 12.3% |
| SYNTH (generated) | 40.9% | 0.0% |
| BHSig260 (Hindi) | 14.3% | 5.9% |
| sign_data (real) | 19.2% | 20.6% |

Borderline scores inside the escalation band are routed to human review
rather than auto-rejected, so the false-reject figures above are an upper
bound on user-visible rejections.

## Figures

![ROC curves](roc_curves.png)

![Score distributions](score_distributions.png)

![Training curves](training_curves.png)
