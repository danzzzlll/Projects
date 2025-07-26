## 🧠 What Was Done in the Notebook

- Implemented a **custom reranker class** using `llama-index`'s `BaseNodePostprocessor`.
- Trained a simple **CatBoostClassifier** on sample data with multiple features.
- Serialized the model using `joblib` and integrated it into a reranking pipeline.
- Used the reranker to adjust scores of retrieved documents based on learned CatBoost relevance predictions.

The reranker can be plugged into any `llama-index` query engine postprocessing step for better relevance ranking.
