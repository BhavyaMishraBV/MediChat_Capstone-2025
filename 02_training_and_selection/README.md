# Training and Model Selection

This section covers how the chatbot's model was trained and why RoBERTa was selected. It has two scripts:

| File | What it is |
|---|---|
| [`local_training.py`](local_training.py) | Stage 1: lightweight fine-tuning of four candidate models on a laptop. |
| [`cluster_training_artemis.py`](cluster_training_artemis.py) | Stages 2 and 3: extended fine-tuning of the selected model on the Kakos cluster, producing `model_3_27_Bhavya`. |

## A note on the code

Both scripts are **consolidated and partly reconstructed**. The original training was done in separate files and in several short passes. In liew of better representation, presentation of the training process and choices, the consolidation was a better format. In short:

- `local_training.py` is a representative consolidation of my staged local runs. It has not been re-run end to end.
- In `cluster_training_artemis.py`, stage 2 comes from my original cluster script but two stages are combined in one script.

## Stage 1: local training of four candidates

Candidates: **DistilGPT-2**, **BioGPT**, a **BERT-base-uncased classifier** (labelled "ClinicalBERT" in my original files; the checkpoint used was `bert-base-uncased`), and **RoBERTa** (`deepset/roberta-base-squad2`).

Fine-tuning every model on every dataset at once would have needed far more memory than a laptop provides, so training was split into short passes on limited data. Each pass continued from the previous one's saved checkpoint. `local_training.py` keeps that structure: one stage per dataset, each loading the last stage's checkpoint and saving its own.

Datasets used for training: SQuAD, PubMedQA, DrugBank (filtered for muscle-related entries) and a Kaggle disease-symptom dataset. Other datasets were consulted, and custom test documents were created. No dataset is distributed here (see the data section for credits).

## Model selection

I compared the candidates on a small test over 100 entries from a muscle-level filtered subset of the disease-symptom data:

1. **Disease to symptoms:** given a disease, predict three symptoms. It counts as a hit when two match.
2. **Symptoms to disease:** given all listed symptoms (at least three), predict the disease. It counts as a hit when the predicted disease matches.

Hits were tallied across the 200 passes, with a local target of 50 or more.

**Result:** RoBERTa came closest, with 31 hits, and gave the best predictions for the training resources it needed. BioGPT was not chosen because its training cost and prediction rate did not match the output the chatbot needed. So I extended RoBERTa's training on the cluster, which raised its tally to 56 on the same test.

This is a project-level sanity check on a small filtered set, not a formal benchmark: it shows RoBERTa was the most promising candidate for the resources available, not that it is the best possible model. The test script is not included because it is part of my ongoing research on matching and validating model outputs on biomedical terms.

## Stages 2 and 3: extended training on the cluster (Artemis)

The cluster work ran on Artemis, my assigned node on the Kakos Center for Scientific Computing cluster at Manhattan University. The full path from start to deployed model:

| Stage | Where | What |
|---|---|---|
| 1 | Local | `deepset/roberta-base-squad2` fine-tuned on the disease-symptom data, saved as `fine_tuned_roberta`. |
| 2 | Cluster | PubMedQA yes/no classification on top of the stage 1 weights. |
| 3 | Cluster | Short question-answering fine-tune on a small custom medical QA dataset, producing `model_3_27_Bhavya`. |

The custom QA set was built in consultation with all the datasets above. It is not published because of licensing, and `cluster_training_artemis.py` documents the format it expects.

The chatbot loads `model_3_27_Bhavya` as a question-answering model ([chatbot section](../03_chatbot_design/)).

## Scope

This was built at an engineering level, to deliver an end-to-end product in one semester. I am continuing this line of work in my own research.
