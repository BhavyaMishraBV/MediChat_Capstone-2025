"""
Data filtering and preprocessing for the MediChat chatbot training data.

Consolidates three of my original scripts into one:
    filtering_Datasets.py   keyword filter for muscle-related rows
    biogptPreprocess.py     merging the Symptom_1..Symptom_n columns into one string
    datasetsDownload.py     downloading / inspecting the Hugging Face datasets

Input files (NOT distributed; obtain them from the original sources, see README):
    --drugbank-csv   a cleaned DrugBank export with columns:
                     name, description, indication, mechanism-of-action
    --disease-csv    the Kaggle disease-symptom file with columns:
                     Disease, Symptom_1, Symptom_2, ... (one row per symptom set)

Output files (written to --out-dir; these are what ../02_training_and_selection reads):
    drugbank_prepared.csv                  muscle-related drugs (keyword filter on "indication")
    disease_symptoms_prepared.csv          every disease row, symptoms merged into one string
    disease_symptoms_muscle_filtered.csv   the muscle-keyword subset of the file above

Changes made while consolidating (the original scripts are not distributed):
    * filtering_Datasets.py had an unfinished line (`muscle_keywords.append(`) and
      filtered the disease file on a column called "name", which that file does not
      have; the disease name column is "Disease".
    * Short acronyms (SMA, LGMD, DMD) are now matched as whole words. As plain
      substrings, "sma" also matches words like "plasma" or "cytoplasm".
    * Symptom values are stripped of the leading spaces present in the raw file, and
      rows are no longer dropped just because they have fewer than the maximum number
      of symptoms (an earlier version used dropna() on the whole table, which would
      remove most rows).
    * The earlier BioGPT script tokenized only the symptom column, so DrugBank rows
      became the string "nan". Here each table keeps its own text, and the training
      script formats it.

Usage:
    python preprocessing.py --drugbank-csv raw/cleaned_drugbank.csv \
        --disease-csv raw/dataset.csv --out-dir prepared_data
    python preprocessing.py --inspect-hf        # only list columns of the HF datasets
"""

import argparse
import os
import re

import pandas as pd

# Whole-word terms vs. substring terms (see "changes" above).
MUSCLE_SUBSTRINGS = ["muscle", "muscular", "dystrophy", "atrophy", "myopathy"]
MUSCLE_ACRONYMS = ["sma", "lgmd", "dmd"]

DRUGBANK_COLUMNS = ["name", "description", "indication", "mechanism-of-action"]


def muscle_pattern():
    parts = [re.escape(k) for k in MUSCLE_SUBSTRINGS]
    parts += [rf"\b{a}\b" for a in MUSCLE_ACRONYMS]
    return "|".join(parts)


def filter_muscle_related(df, column):
    """Keep rows whose `column` mentions a muscle-related keyword (case-insensitive)."""
    mask = df[column].astype("string").str.contains(muscle_pattern(), case=False, na=False, regex=True)
    return df[mask]


def prepare_disease_symptoms(df):
    """One row per original record: Disease and a comma-joined symptoms string."""
    df = df.dropna(subset=["Disease"]).copy()
    df["Disease"] = df["Disease"].astype("string").str.strip()

    symptom_cols = [c for c in df.columns if c.lower().startswith("symptom_")]
    symptom_cols.sort(key=lambda c: int(c.split("_")[1]))
    for c in symptom_cols:
        df[c] = df[c].astype("string").str.strip()

    df["symptoms"] = df[symptom_cols].apply(
        lambda row: ", ".join(s for s in row if pd.notna(s) and s), axis=1
    )
    df = df[df["symptoms"] != ""]
    return df[["Disease", "symptoms"]].reset_index(drop=True)


def prepare_drugbank(df):
    """Muscle-related drugs; missing description/mechanism become empty strings."""
    df = df[DRUGBANK_COLUMNS].dropna(subset=["name", "indication"]).copy()
    df = filter_muscle_related(df, "indication")
    return df.fillna("").reset_index(drop=True)


def inspect_hf_datasets():
    """Download and list the columns of the Hugging Face datasets used in training."""
    from datasets import load_dataset

    pubmedqa = load_dataset("pubmed_qa", "pqa_artificial")
    squad = load_dataset("squad")
    print("PubMedQA columns:", pubmedqa.column_names)
    print("SQuAD columns:", squad.column_names)
    # An 80/20 split of PubMedQA's train set (seed 42) was used while exploring.
    split = pubmedqa["train"].train_test_split(test_size=0.2, seed=42)
    print("PubMedQA split sizes:", {k: len(v) for k, v in split.items()})


def main():
    p = argparse.ArgumentParser(description="Filter and prepare the MediChat training data")
    p.add_argument("--drugbank-csv")
    p.add_argument("--disease-csv")
    p.add_argument("--out-dir", default="prepared_data")
    p.add_argument("--inspect-hf", action="store_true",
                   help="Only download and list columns of the Hugging Face datasets.")
    a = p.parse_args()

    if a.inspect_hf:
        inspect_hf_datasets()
        return
    if not (a.drugbank_csv and a.disease_csv):
        p.error("--drugbank-csv and --disease-csv are required (or use --inspect-hf)")

    os.makedirs(a.out_dir, exist_ok=True)

    drugbank = prepare_drugbank(pd.read_csv(a.drugbank_csv))
    drugbank.to_csv(os.path.join(a.out_dir, "drugbank_prepared.csv"), index=False)

    disease = prepare_disease_symptoms(pd.read_csv(a.disease_csv))
    disease.to_csv(os.path.join(a.out_dir, "disease_symptoms_prepared.csv"), index=False)

    disease_muscle = filter_muscle_related(disease, "Disease")
    disease_muscle.to_csv(os.path.join(a.out_dir, "disease_symptoms_muscle_filtered.csv"), index=False)

    print(f"DrugBank (muscle-related): {len(drugbank)} rows")
    print(f"Disease-symptom rows: {len(disease)} | muscle-filtered: {len(disease_muscle)}")
    print(f"Saved to {a.out_dir}")


if __name__ == "__main__":
    main()
