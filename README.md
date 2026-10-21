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

## Reference

This project is motivated by the following paper:

> H Mohamad Tahir, W Hasan, A Md Said, NH Zakaria (2015). *Hybrid machine learning technique for intrusion detection system*. repo.uum.edu.my. https://repo.uum.edu.my/id/eprint/15600/

Paper author: Prof. Assoc. Prof. Dr. Hatim Mohamad Tahir, Universiti Utara Malaysia (hatim@uum.edu.my)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
