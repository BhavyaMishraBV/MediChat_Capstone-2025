# Data and Preprocessing

This section shows how the training data was selected and prepared. **No datasets are distributed in this repository.** Each one has its own license and terms, so please obtain them from the original sources listed below.

## What `preprocessing.py` does

[`preprocessing.py`](preprocessing.py) takes the raw DrugBank and disease-symptom files and produces the prepared files that [the training scripts](../02_training_and_selection/) read:

1. **Keyword filtering.** DrugBank entries are kept when their indication mentions a muscle-related term (muscle, muscular, dystrophy, atrophy, myopathy, or the acronyms SMA, LGMD, DMD). The disease-symptom file is filtered the same way on the disease name, giving the muscle-level subset.
2. **Symptom merging.** The Kaggle file stores one symptom per column (`Symptom_1`, `Symptom_2`, ...). These are cleaned of stray whitespace and merged into one comma-separated string per row.
3. **Exploration helper.** `--inspect-hf` downloads and lists the columns of the Hugging Face datasets (PubMedQA and SQuAD) used in training.

The script consolidates three of my original working scripts. While consolidating I fixed an unfinished line, a wrong column name, a substring match that caught unrelated words (for example "plasma" for "sma"), and a row-dropping step that would have removed most of the symptom data. The header of the script lists each change. I tested the consolidated script on small synthetic files; it has not been run on the real datasets in its consolidated form.

## Dataset credits

Thank you to the creators and maintainers of these datasets. I filtered, merged and reformatted most of them for this project, so any errors in the prepared data are mine, not theirs.

**Used for training**

| Dataset | Used for | Source |
|---|---|---|
| **DrugBank** | Muscle-related drug entries (name, description, indication, mechanism of action) | [drugbank.com](https://www.drugbank.com). Wishart, D. S., et al. (2018). DrugBank 5.0: a major update to the DrugBank database for 2018. *Nucleic Acids Research*, 46(D1), D1074–D1082. |
| **Kaggle Disease-Symptom dataset** | Disease and symptom pairs, used for fine-tuning and for the model comparison test | Itachi 9604 (2022). [Disease-Symptom Description Dataset](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset) [Data set]. Kaggle. |
| **PubMedQA** (PubMed-derived, `pqa_artificial`) | Yes/no biomedical question answering | Jin et al., *PubMedQA: A Dataset for Biomedical Research Question Answering*, EMNLP 2019. Loaded from the Hugging Face Hub. |
| **SQuAD** | General question-answering fine-tuning | Rajpurkar et al., *SQuAD: 100,000+ Questions for Machine Comprehension of Text*, 2016. Loaded from the Hugging Face Hub. |

Also thanks to the creators of the base models used in training: `deepset/roberta-base-squad2` (deepset), BioGPT (Microsoft), DistilGPT-2 and BERT.

**Downloaded but not part of the final model:** DailyDialog was tried as general conversation data for DistilGPT-2 and was left out of the final model.

**Consulted as a guide only (not used for training):** datasets on Duchenne muscular dystrophy (DMD), limb-girdle muscular dystrophy (LGMD) and spinal muscular atrophy (SMA), and MIMIC-III (Johnson, A. E. W., et al. (2016). MIMIC-III, a freely accessible critical care database. *Scientific Data*, 3, 160035). I reviewed them to understand medical terminology and how clinical information is described. None of them was used to train or evaluate a model.

**Not part of the chatbot:** BodyParts2D/Anatomography, MalaCards and DDXPlus were used by the frontend team for the human body model in the wider MediChat project. They were not used in the chatbot process described in this repository. See the team repository for details.

**Created by me, not published:** custom test documents for evaluating the models, and a short custom question-answer set built with reference to the datasets above. These are not released because of the licenses of their sources.

## Prepared file format

If you obtain the raw data yourself, these are the files the training scripts expect:

| File | Columns |
|---|---|
| `drugbank_prepared.csv` | `name`, `description`, `indication`, `mechanism-of-action` |
| `disease_symptoms_prepared.csv` | `Disease`, `symptoms` (comma-separated) |
| `disease_symptoms_muscle_filtered.csv` | same as above, muscle-related diseases only |
