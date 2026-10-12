# LightGBM sentence scorer vs NMF top-sentence summarizer (CNN/DailyMail, ROUGE-1 F1)

## Run

```
pip install -r requirements.txt
python analysis.py
```

This writes `results.json` and `figures/rouge1_bar.png`.

## Fix applied

The previous run crashed with HTTP 404 when fetching the test split from the
datasets-server, because the config name `3.0.0` does not exist for
`ccdv/cnn_dailymail`. `fetch_rows` now tries the config names `default` and then
`3.0.0`, and uses the first one that returns data. The dataset, split, baseline,
improved model, evaluation code and test split are unchanged.

## Method

- Data: about 700 articles from the CNN/DailyMail test split, split 70/30 at the article level (seed 42).
- Baseline: per-document NMF on a TF-IDF sentence-term matrix. It picks the top-weighted sentence per topic, then fills up to 3 sentences by total weight.
- Improved: LightGBM trained on greedy ROUGE-1 oracle labels, using sentence features. It selects the 3 sentences with the highest predicted probability.
- Hyper-parameters are chosen by 3-fold CV on training articles only.
- Metric: ROUGE-1 F1 on the held-out articles, averaged over seeds 42–44.

## Reference

This project is motivated by the following paper:

> NK Anh, NK Toi, NV Linh (2014). *An interpretable method for text summarization based on simplicial non-negative matrix factorization*. Proceedings of the 5th Symposium on. https://dl.acm.org/doi/abs/10.1145/2676585.2676604

Paper author: Prof. Nguyen Thi Kim Anh, Hanoi University of Science and Technology (anhnk@soict.hust.edu.vn)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
