# NMF Baseline vs LightGBM Sentence Scorer for News Summarization

Interpretable extractive summarization methods based on matrix factorization, such as the simplicial NMF approach in "An interpretable method for text summarization based on simplicial non-negative matrix factorization" (2014), pick sentences using topic structure that a reader can inspect. A standard per-document NMF selector is a simple version of this idea. This experiment asks whether a supervised sentence scorer that uses NMF topic loadings as features can outperform that selector.

**Question:** Can a LightGBM sentence scorer trained on sentence features beat a standard per-document NMF sentence-selection summarizer on ROUGE-1 F1 for CNN/DailyMail news articles?

**Dataset:** CNN/DailyMail 3.0.0 (`ccdv/cnn_dailymail`), test-split rows fetched from the Hugging Face datasets-server API. This is not the paper's own dataset (`same_as_paper: false`).

**Baseline:** per-document NMF on the TF-IDF sentence-term matrix. It picks the top-weighted sentence for each of 3 topics and selects 3 sentences. This is a stand-in for simplicial NMF, not a faithful reimplementation.

**Improved model:** a LightGBM classifier on sentence features (position, length, TF-IDF centrality, per-document NMF topic loadings), trained on greedy ROUGE-1 oracle labels. It selects the 3 highest-probability sentences. Both methods share the same sentences, vocabulary, and ROUGE code.

## How to run

```bash
pip install -r requirements.txt
python analysis.py
```

Results are written to `results.json` (and `figures/rouge1_bar.png`).

## Reference

This project is motivated by the following paper:

> NK Anh, NK Toi, NV Linh (2014). *An interpretable method for text summarization based on simplicial non-negative matrix factorization*. Proceedings of the 5th Symposium on. https://dl.acm.org/doi/abs/10.1145/2676585.2676604

Paper author: Prof. Nguyen Thi Kim Anh, Hanoi University of Science and Technology (anhnk@soict.hust.edu.vn)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
