# What did a dementia classifier learn from OASIS-2?

**Mingpu (Andy) Wu · Indian Springs School research project**

In high school, I compared nine machine-learning models on clinical and MRI measurements from the Open Access Series of Imaging Studies. My original poster reported 95% test accuracy for XGBoost. Returning to that result raised a question I could investigate with the surviving poster and data: how much did the model learn from the brain measurements, and how much could it infer from a clinical rating already in the table? [Original research poster](poster/Original-poster.pptx)

The poster's feature-ranking table gave its highest rank to two MRI measurements, a cognitive test score (MMSE), and the Clinical Dementia Rating (CDR). The OASIS researchers themselves used CDR to characterize dementia. A model that receives the rating therefore has access to part of the information behind the label. Repeated visits to the same person also make it possible for a visit-level split to place that person in both training and testing. [OASIS study publication](https://doi.org/10.1162/jocn.2009.21407)

## What the renewal found

The historical XGBoost matrix contains 106 correct classifications in 112 test visits. The new run came within one classification of that count when it used CDR and the other three top-ranked inputs. Seventy participants appeared on both sides of the reconstructed visit split. The original notebook and split assignments were lost, so the close counts do not establish that the original experiment was reproduced exactly.

| XGBoost evaluation | Correct | Accuracy | Positive-class F1 |
| --- | ---: | ---: | ---: |
| Historical poster, as reported | 106/112 | 95% (rounded) | Averaging unspecified; reported F1 0.95 |
| Reconstructed visit split, with CDR | 107/112 | 95.5% | 0.952 |
| Same visit split, withholding CDR | 82/112 | 73.2% | 0.694 |

For a second experiment, I used one baseline visit per person and asked whether the inputs could distinguish a recorded CDR of zero from a rating above zero. Each person was held out once in five-fold cross-validation. XGBoost classified **125/150 (83.3%)** correctly with the cognitive score and both MRI measurements, **129/150 (86.0%)** with the cognitive score alone, and **74/150 (49.3%)** with the two MRI measurements alone. Always choosing the training majority classified 85/150 correctly. In this sample, adding those two MRI summaries to MMSE did not improve XGBoost's accuracy. These experiments evaluate recorded dementia ratings within one cohort. They do not establish a clinical diagnosis or predict who will develop Alzheimer's disease.

## Reproduce the analysis

The Python code was written anew with OpenAI Codex assistance. The published run used Python 3.13.9. From the repository root, run the following commands in a Python environment:

```sh
python -m pip install -r requirements.txt
python get_data.py
python analyze.py data/oasis_longitudinal.csv
```

The downloader verifies the pinned public CSV. An existing local copy can be supplied directly to `analyze.py`. The output contains aggregate metrics for nine model families and a majority baseline under five conditions. The original presentation and its transcribed table are in [poster/](poster/). The dataset attribution, assumptions and limitations are in [METHODS.md](METHODS.md). All renewed results are in [results/reanalysis.csv](results/reanalysis.csv).
