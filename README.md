# Different labels for the same expression?

Code for the article *Different labels for the same expression? Representation sensitivity of sentiment models on
original and English-interpreted stress expressions from Nigerian university students* (Akinrinwa, submitted to
Computer Speech & Language).

Each of 3,741 questionnaire responses from 314 Nigerian university students is held as the original expression
(Nigerian English, Nigerian Pidgin or Yoruba) and as a researcher-prepared English interpretation, each with its own
reviewed reference label. Three experiments compare how often model labels change between the two forms
(Experiment I), how well each system agrees with the reference labels (Experiment II), and how classifiers trained
on those labels generalise to respondents and expressions withheld from training (Experiment III).

## Repository layout

| Folder | Contents |
| --- | --- |
| `inference/` | `run_classifiers_and_textblob.py` (Twitter RoBERTa, Twitter XLM-R, AfriSenti AfroXLM-R, TextBlob) and `qwen_label_scoring_colab.py` (Qwen2.5-3B-Instruct, zero-shot and cultural prompts) |
| `scripts/` | Analysis scripts, numbered in the order they run |
| `tools/` | `make_public_labels.py`, which writes the label-only file |
| `data/` | Where the restricted input files go; see `data/README.md` |
| `outputs/` | Created by `run_all.py` |

## Reproducing the results

```
pip install -r environment.txt
python run_all.py              # about 30 minutes on a laptop CPU; add --skip-exp3 to omit Experiment III
```

| Script | Produces | In the article |
| --- | --- | --- |
| `01_harmonise_labels.py` | Harmonised reference labels and `label_harmonisation_log.csv` | Section 3.6 |
| `02_build_predictions.py` | `predictions.csv`, with every input matched to its record by SHA-256 | Section 3.12 |
| `03_experiments_1_2.py` | Shift rates, cross-system agreement, accuracy with intervals, Δm, subgroups, sensitivity analyses | Sections 4.2–4.6 |
| `04_gloss_and_macroF1_intervals.py` | Translation-type and gloss-type results, macro-F1 intervals | Tables A.4, A.8, A.9 |
| `05_robustness_checks.py` | Fixed-reference and unique-pair Δm, respondent-gloss exclusion, paired XLM-R–AfriSenti contrast, prompt distributions | Section 4.4, Table A.5 |
| `06_reference_symmetry_tests.py` | Cluster-robust McNemar–Bowker and Stuart–Maxwell tests on the reference labels | Section 4.2 |
| `07_model_symmetry_tests.py` | The same tests for every system | Section 4.3 |
| `08_experiment3_supervised.py` | Naive Bayes, linear SVM and MLP under participant-disjoint and participant- and text-disjoint training | Table 8 |
| `09_experiment3_controls.py` | Size-matched control and paired contrasts | Section 4.8 |
| `10_build_tables.py` | `tables.json`: Tables 4, 6 and A.1–A.9 as printed | Results, Appendix A |
| `11`–`14` | Figures 2–7, A.1 and the graphical abstract | Figures |

Without the restricted data, the label-only file (if published) is enough to check Experiments I and II.

## Statistical notes

Intervals are 95% percentile intervals from a participant-cluster bootstrap (1,000 replicates, seed 42). The symmetry
tests use the Wald statistic d′V⁻¹d, with V the sum over respondents of the outer product of each respondent's
contribution to d (Durkalski et al., 2003); without clustering they reproduce the textbook statistics.

## Licence and citation

Code: MIT licence. Documentation and any published label file: CC BY 4.0. Please cite the article and the archived
release (see `CITATION.cff`).

## Contact

Sadura Priscilla Akinrinwa · spakinrinwa@gmail.com · ORCID 0000-0002-0775-1128
