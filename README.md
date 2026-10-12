# LightGBM sentence scorer vs NMF top-sentence summarizer (CNN/DailyMail)

Question: can a LightGBM sentence scorer trained on sentence features beat a per-document NMF sentence-selection baseline on ROUGE-1 F1?

Method is unchanged: article-level 70/30 split of about 700 CNN/DM test-split articles, greedy ROUGE-1 oracle labels, NMF top-sentence baseline vs LightGBM top-3 selection, seeds 42-44, 3-point grid searched with CV on training articles only.

Data-source fix: the `ccdv/cnn_dailymail` configs `default` and `3.0.0` both returned 404 from the datasets-server. The script now falls back to `abisee/cnn_dailymail` config `3.0.0`, which contains the same CNN/DailyMail 3.0.0 articles. The source actually used is recorded in `results.json`.

Run: `python analysis.py` (CPU only, well under 20 minutes). Writes `results.json` and `figures/rouge1_bar.png`.

## Results (latest run)

On 210 held-out CNN/DailyMail 3.0.0 test articles, ROUGE-1 F1 was 25.9% for a standard per-document NMF sentence-selection summarizer versus 30.2% for my LightGBM top-3 sentence scorer, a gain of 4.3 points.

| Metric | Value |
|---|---|
| baseline_score | 0.2591 |
| improved_score | 0.3025 |
| improvement | 0.04334 |
| baseline_score_std | 0.001659 |
| improved_score_std | 0.002559 |
| baseline_rouge2_f1 | 0.07482 |
| improved_rouge2_f1 | 0.1194 |
| improvement_rouge2_f1 | 0.04457 |
| cv_best_rouge1 | 0.2941 |
| selected_num_leaves | 7 |

Run on 2026-10-12. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Single run of data split (seed 42); seeds 42-44 vary NMF init and LightGBM. Split is 70/30 of ~700 articles rather than exactly 400/200 to satisfy the 20-30% test-size rule. Articles come from the CNN/DM test split (abisee/cnn_dailymail, config 3.0.0), not the paper's dataset. Baseline is a stand-in for simplicial NMF. Std is population std over 3 seeds. Only a 3-point hyper-parameter grid was searched, using CV on training articles. Dataset source is resolved at runtime: ccdv/cnn_dailymail configs 'default' and '3.0.0' returned 404 on the datasets-server, so it falls back to abisee/cnn_dailymail 3.0.0 (same articles).

## Reference

This project is motivated by the following paper:

> NK Anh, NK Toi, NV Linh (2014). *An interpretable method for text summarization based on simplicial non-negative matrix factorization*. Proceedings of the 5th Symposium on. https://dl.acm.org/doi/abs/10.1145/2676585.2676604

Paper author: Prof. Nguyen Thi Kim Anh, Hanoi University of Science and Technology (anhnk@soict.hust.edu.vn)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
