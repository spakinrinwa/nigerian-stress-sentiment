# Data

The questionnaire texts are **not** in this repository. Participants consented to the use of their anonymised answers
for research and publication, and short expressions can be recognisable to peers, so the texts are shared only on
request, under the conditions in the article's Data availability statement.

To reproduce the results, place the restricted files in `data/restricted/` (or point the `DATA_DIR` environment
variable at another folder):

| File | Contents | Produced by |
| --- | --- | --- |
| `sentiment_benchmark.csv` | One row per record: identifiers, prompt, original expression, the English interpretation the models scored, language, construct, rule code (`Priority`), referent and code-switching fields | Coding workbook export |
| `Human_Sentiment_Annotation_3741.csv` | Reviewed reference labels for the original and English forms, before harmonisation | Annotation |
| `predictions_raw/<Model>__<Representation>__<Mode>.csv` | Classifier and TextBlob predictions (8 files) | `inference/run_classifiers_and_textblob.py` |
| `qwen/Qwen_<Representation>_<ZeroShot or Cultural>.csv` | Qwen generated labels and candidate-scoring probabilities (4 files) | `inference/qwen_label_scoring_colab.py` |

`data/public/labels_and_predictions.csv` (no text) is written by `tools/make_public_labels.py` and is published
only if the ethics committee agrees.
