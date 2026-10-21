# K-means-filtered SVM vs class-balanced LightGBM for NSL-KDD intrusion detection

Motivation: The paper "Hybrid machine learning technique for intrusion detection system" (2015, https://repo.uum.edu.my/id/eprint/15600/) combines K-means clustering with an SVM classifier, with preprocessing, to detect network intrusions. This experiment re-implements that hybrid as a baseline and checks whether a simpler, class-balanced gradient-boosting model with a cross-validated decision threshold does better.

Question: Can a class-balanced LightGBM with a CV-tuned threshold beat a K-means-filtered SVM hybrid at detecting intrusions on NSL-KDD while producing fewer false alarms?

Dataset: NSL-KDD, using the public KDDTrain+ (training) and KDDTest+ (test) files from the defcom17/NSL_KDD GitHub repository. This is the same dataset as the paper. A fixed 20,000-row subsample of the training file is used for both models.

Baseline vs improved:
- Baseline: K-means (k=20) fit on training rows, rows whose label disagrees with their cluster's majority are removed, then an RBF SVM is trained.
- Improved: class-balanced LightGBM on the same features.
- For both, the decision threshold is chosen by 5-fold CV on the training subsample to maximise F1, then applied once to KDDTest+.

Main metric: malicious-class F1. Secondary: detection rate and false alarm rate.

## How to run

```
pip install -r requirements.txt
python analysis.py
```

Results: written to results.json (and figures/baseline_vs_improved.png).

## Results (latest run)

In my experiment on NSL-KDD's KDDTest+, a standard K-means-filtered RBF SVM hybrid reached malicious-class F1 0.690 while my class-balanced LightGBM reached 0.757, though its false-alarm rate was slightly higher (0.029 vs 0.026).

| Metric | Value |
|---|---|
| baseline_f1 | 0.6903 |
| baseline_score_std | 0.003083 |
| baseline_detection_rate | 0.5377 |
| baseline_false_alarm_rate | 0.0265 |
| improved_f1 | 0.7567 |
| improved_score_std | 0.000687 |
| improved_detection_rate | 0.622 |
| improved_false_alarm_rate | 0.02911 |
| baseline_score | 0.6903 |
| improved_score | 0.7567 |
| improvement | 0.06633 |

Run on 2026-10-21. Full output: [results.json](results.json) · [run.log](run.log)

Notes: KDDTest+ is used as the test set as specified, so the split is the predefined NSL-KDD split (about 15% test), not a 20-30% resplit. Training uses a 20,000-row subsample of KDDTrain+. The threshold is chosen on out-of-fold training scores only. Std is the population std over 3 seeds. The SVM is the main runtime cost. The test set has known distribution shift (novel attacks), and results are not a claim about the original 2015 paper's numbers, which are not stated.

## Reference

This project is motivated by the following paper:

> H Mohamad Tahir, W Hasan, A Md Said, NH Zakaria (2015). *Hybrid machine learning technique for intrusion detection system*. repo.uum.edu.my. https://repo.uum.edu.my/id/eprint/15600/

Paper author: Prof. Assoc. Prof. Dr. Hatim Mohamad Tahir, Universiti Utara Malaysia (hatim@uum.edu.my)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
