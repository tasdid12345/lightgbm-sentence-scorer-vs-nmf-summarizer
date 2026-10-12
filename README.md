# LightGBM sentence scorer vs NMF top-sentence summarizer (CNN/DailyMail)

Question: can a LightGBM sentence scorer trained on sentence features beat a per-document NMF sentence-selection baseline on ROUGE-1 F1?

Method is unchanged: article-level 70/30 split of about 700 CNN/DM test-split articles, greedy ROUGE-1 oracle labels, NMF top-sentence baseline vs LightGBM top-3 selection, seeds 42-44, 3-point grid searched with CV on training articles only.

Data-source fix: the `ccdv/cnn_dailymail` configs `default` and `3.0.0` both returned 404 from the datasets-server. The script now falls back to `abisee/cnn_dailymail` config `3.0.0`, which contains the same CNN/DailyMail 3.0.0 articles. The source actually used is recorded in `results.json`.

Run: `python analysis.py` (CPU only, well under 20 minutes). Writes `results.json` and `figures/rouge1_bar.png`.

## Reference

This project is motivated by the following paper:

> NK Anh, NK Toi, NV Linh (2014). *An interpretable method for text summarization based on simplicial non-negative matrix factorization*. Proceedings of the 5th Symposium on. https://dl.acm.org/doi/abs/10.1145/2676585.2676604

Paper author: Prof. Nguyen Thi Kim Anh, Hanoi University of Science and Technology (anhnk@soict.hust.edu.vn)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
