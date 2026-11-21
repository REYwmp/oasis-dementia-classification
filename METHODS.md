# Data, labels and evaluation

## Which table survived?

The poster names longitudinal OASIS T1-weighted scans and lists MR Delay, CDR, MMSE, eTIV, nWBV and ASF. The recovered OASIS-2 table matches that description, with 373 visits from 150 people. The original source URL and working copy are unknown. The renewal uses a version-pinned copy of the tabular data. No MRI image files are needed. [Washington University dataset](https://sites.wustl.edu/oasisbrains/home/oasis-2/) · [Study publication](https://doi.org/10.1162/jocn.2009.21407) · [Kaggle distribution](https://www.kaggle.com/datasets/jboysen/mri-and-alzheimers) · [Exact CSV used here](https://github.com/NizarIslah/dementia-prediction/blob/de76175f53465fa0167dceb8cf57aabf717005c0/oasis_longitudinal.csv)

## Reading the original poster

The poster reports a 70/30 split, discusses z-score normalization, stratified cross-validation, Boruta and PCA, and shows nine models. Four features have Boruta rank 1 in the displayed table: Clinical Dementia Rating (CDR), Mini-Mental State Examination (MMSE), estimated total intracranial volume (eTIV) and normalized whole-brain volume (nWBV). The surviving material does not specify the exact final feature set, conversion rule, row assignments, feature-selection fit, tuned model settings or F1 averaging rule. The renewal treats the four displayed features as a fixed historical clue. It does not refit Boruta or PCA.

The XGBoost panel displays the two rows [57, 0] and [6, 49]. Its diagonal gives 106/112 correct (94.64%). Taking class 1 as positive gives an F1 of 0.9423, while macro F1 rounds to 0.95. The displayed matrix image is reused in the logistic regression, naïve Bayes and XGBoost panels although their summary-table F1 values differ. The archived table preserves what the poster reports. [Original poster](poster/Original-poster.pptx) · [Transcribed results](poster/reported-metrics.csv)

## How the new comparisons work

**Visit-level reconstruction.** The table contains 190 Nondemented, 146 Demented and 37 Converted visits. Combining Demented and Converted as class 1 yields the poster's 50.9%/49.1% class proportions. This is a plausible reconstruction of the historical coding. It gives some pre-conversion visits a positive label. After sorting by subject and visit, a stratified 70/30 split with seed 42 gives 261 training and 112 test visits. Seventy people appear on both sides. Two runs use exactly the same rows, one with all four rank-one features and one excluding CDR. These runs illustrate how a label-related input affects the score on this particular split.

**Participant-level reanalysis.** Keep each person's first visit and classify its recorded global CDR as zero or above zero. This uses the visit rating even when a participant's longitudinal group label differs. There are 85 zero and 65 positive ratings. Withhold CDR, group, identifiers and subsequent observations from predictors. Use the same five shuffled stratified folds (seed 42) for all models and feature sets. Each of the 150 people contributes one out-of-fold prediction. Compare two MRI summaries plus MMSE, MRI summaries alone, and MMSE alone.

**Fitting and metrics.** Fit median imputation and standardization inside each training partition. The stack uses logistic regression, random forest and SVM base learners, each with its own preprocessing inside the inner five-fold cross-validation. Hyperparameters are fixed in code. The majority-class baseline is fit on each training partition. Positive F1 refers to class 1. Macro F1 gives the two classes equal weight. Baseline-visit metrics pool all out-of-fold predictions. The aggregate CSV includes all four confusion-matrix counts for every run. [Analysis](analyze.py) · [Results](results/reanalysis.csv)

**Interpretation.** The original OASIS study used clinical assessments, including CDR, to characterize dementia. The two evaluation protocols have different targets, units and validation designs. Comparisons of feature sets within one protocol hold the tested observations fixed. The MRI-only inputs are two global measurements. The results do not measure the performance of a system using the full scans. The cohort and sample size limit generalization to other settings.

**Acknowledgment.** Data were provided by OASIS. OASIS-2 longitudinal principal investigators: D. Marcus, R. Buckner, J. Csernansky and J. Morris. Support included P50 AG05681, P01 AG03991, P01 AG026276, R01 AG021910, P20 MH071616 and U24 RR021382. [OASIS-2 acknowledgments](https://sites.wustl.edu/oasisbrains/home/oasis-2/)
