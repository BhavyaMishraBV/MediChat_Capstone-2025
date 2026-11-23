"""
Local fine-tuning of the four candidate models (stage 1 of the training process).

Candidates:
    distilgpt2   causal LM        (distilgpt2)
    biogpt       causal LM        (microsoft/biogpt)
    bert         classifier       (bert-base-uncased; labelled "ClinicalBERT" in my
                                   original files, but the checkpoint used was
                                   bert-base-uncased)
    roberta      question answering (deepset/roberta-base-squad2)

Provenance (please read):
  This is a CONSOLIDATED, REPRESENTATIVE script. The real training was run in two or
  three lightweight passes per model, in separate files, because fine-tuning every
  model on every dataset in one go needs far more RAM than a laptop has. The staged
  structure is kept here: each stage loads the previous stage's checkpoint and saves
  its own, so a run can be stopped and resumed between stages.
  Differences from the originals, made while consolidating:
    * Uses the Hugging Face Trainer instead of a hand-written loop (the original loop
      did not build start/end positions for RoBERTa question answering).
    * RoBERTa starts from deepset/roberta-base-squad2 (the original consolidated file
      mistakenly listed roberta-base).
    * The DailyDialog stage is dropped; it was not part of the final model.
    * Disease-symptom QA formatting (see make_disease_symptom_qa) is a representative
      reconstruction; the exact original format was not preserved.
    * Not re-run end to end after consolidation.

Inputs (produced by the preprocessing step in ../01_data_and_preprocessing):
    --drugbank-csv   columns: name, description, indication, mechanism-of-action
    --symptoms-csv   columns: Disease, symptoms   (symptoms = comma-joined string)
Datasets pulled from the Hugging Face Hub: pubmed_qa (pqa_artificial), squad.
The prepared CSV files are not distributed.

Usage:
    python local_training.py --model roberta --drugbank-csv data/drugbank_prepared.csv \
        --symptoms-csv data/disease_symptoms_prepared.csv --max-samples 2000
"""

import argparse
import os

import pandas as pd
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BertForSequenceClassification,
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
    RobertaForQuestionAnswering,
    RobertaTokenizerFast,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

EPOCHS = 3
BATCH_SIZE = 8
MAX_LENGTH = 512
SEED = 42


def training_args(output_dir, lr):
    return TrainingArguments(
        output_dir=os.path.join(output_dir, "_trainer"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=lr,
        weight_decay=0.01,
        logging_steps=100,
        save_strategy="no",       # stages save explicitly below
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=SEED,
    )


# ==========================================================
# Dataset loading (small samples: limited local resources)
# ==========================================================
def cap(dataset, max_samples):
    if max_samples and len(dataset) > max_samples:
        dataset = dataset.shuffle(seed=SEED).select(range(max_samples))
    return dataset


def load_text_stages(drugbank_csv, symptoms_csv, max_samples):
    """Four datasets rendered as plain text for the causal language models."""
    drug = pd.read_csv(drugbank_csv).dropna()
    drug_text = [
        f"{r['name']} | {r['description']} | {r['indication']} => {r['mechanism-of-action']}"
        for _, r in drug.iterrows()
    ]
    sym = pd.read_csv(symptoms_csv).dropna()
    sym_text = [f"{r['Disease']} => {r['symptoms']}" for _, r in sym.iterrows()]

    pubmed = cap(load_dataset("pubmed_qa", "pqa_artificial", split="train"), max_samples)
    pubmed_text = [f"Question: {r['question']} Answer: {r['long_answer']}" for r in pubmed]

    squad = cap(load_dataset("squad", split="train"), max_samples)
    squad_text = [
        f"Question: {r['question']} Context: {r['context']} Answer: {r['answers']['text'][0]}"
        for r in squad
    ]

    stages = {
        "drugbank": drug_text,
        "disease_symptoms": sym_text,
        "pubmedqa": pubmed_text,
        "squad": squad_text,
    }
    return {name: Dataset.from_dict({"text": texts[:max_samples or None]})
            for name, texts in stages.items()}


# ==========================================================
# Causal LMs: DistilGPT-2 and BioGPT, one dataset stage after another
# ==========================================================
def train_causal_lm(model_name, short_name, stages, out_root):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    checkpoint = model_name
    for stage_name, ds in stages.items():
        print(f"\n[{short_name}] stage: {stage_name} ({len(ds)} examples)")
        model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
        tokenized = ds.map(
            lambda b: tokenizer(b["text"], truncation=True, max_length=MAX_LENGTH),
            batched=True, remove_columns=["text"],
        )
        out_dir = os.path.join(out_root, f"{short_name}_{stage_name}")
        Trainer(model=model, args=training_args(out_dir, 5e-5),
                train_dataset=tokenized, data_collator=collator).train()
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        checkpoint = out_dir          # next stage continues from this one
        print(f"[{short_name}] saved {out_dir}")


# ==========================================================
# BERT classifier: PubMedQA yes/no
# ==========================================================
def train_bert_classifier(max_samples, out_root):
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)

    ds = load_dataset("pubmed_qa", "pqa_artificial", split="train")
    ds = ds.filter(lambda e: e["final_decision"] in ("yes", "no"))
    ds = cap(ds, max_samples)
    ds = ds.map(lambda e: {"labels": 1 if e["final_decision"] == "yes" else 0})
    ds = ds.map(
        lambda b: tokenizer(b["question"], b["long_answer"], truncation=True, max_length=MAX_LENGTH),
        batched=True, remove_columns=[c for c in ds.column_names if c != "labels"],
    )
    out_dir = os.path.join(out_root, "bert_pubmedqa")
    Trainer(model=model, args=training_args(out_dir, 2e-5), train_dataset=ds).train()
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[bert] saved {out_dir}")


# ==========================================================
# RoBERTa question answering (the model that was selected)
# ==========================================================
def make_disease_symptom_qa(symptoms_csv, max_samples):
    """Representative reconstruction: turn each (Disease, symptoms) row into a
    SQuAD-style example. The exact original format was not preserved."""
    df = pd.read_csv(symptoms_csv).dropna()
    rows = []
    for _, r in df.iterrows():
        prefix = f"{r['Disease']} symptoms: "
        rows.append({
            "question": f"What are the symptoms of {r['Disease']}?",
            "context": prefix + r["symptoms"],
            "answers": {"text": [r["symptoms"]], "answer_start": [len(prefix)]},
        })
    return cap(Dataset.from_list(rows), max_samples)


def qa_features(tokenizer, max_length=384, doc_stride=128):
    """Standard SQuAD preprocessing: map answer character offsets to token
    start/end positions (the step missing from the original loop)."""
    def fn(examples):
        tokenized = tokenizer(
            [q.lstrip() for q in examples["question"]], examples["context"],
            truncation="only_second", max_length=max_length, stride=doc_stride,
            return_overflowing_tokens=True, return_offsets_mapping=True,
            padding="max_length",
        )
        sample_map = tokenized.pop("overflow_to_sample_mapping")
        offsets_all = tokenized.pop("offset_mapping")
        tokenized["start_positions"], tokenized["end_positions"] = [], []
        for i, offsets in enumerate(offsets_all):
            ids = tokenized["input_ids"][i]
            cls = ids.index(tokenizer.cls_token_id)
            seq = tokenized.sequence_ids(i)
            ans = examples["answers"][sample_map[i]]
            start_c = ans["answer_start"][0]
            end_c = start_c + len(ans["text"][0])
            ts = 0
            while seq[ts] != 1:
                ts += 1
            te = len(ids) - 1
            while seq[te] != 1:
                te -= 1
            if not (offsets[ts][0] <= start_c and offsets[te][1] >= end_c):
                tokenized["start_positions"].append(cls)
                tokenized["end_positions"].append(cls)
            else:
                while ts < len(offsets) and offsets[ts][0] <= start_c:
                    ts += 1
                tokenized["start_positions"].append(ts - 1)
                while offsets[te][1] >= end_c:
                    te -= 1
                tokenized["end_positions"].append(te + 1)
        return tokenized
    return fn


def train_roberta_qa(symptoms_csv, max_samples, out_root):
    base = "deepset/roberta-base-squad2"
    tokenizer = RobertaTokenizerFast.from_pretrained(base)
    model = RobertaForQuestionAnswering.from_pretrained(base).to(device)

    raw = make_disease_symptom_qa(symptoms_csv, max_samples)
    feats = raw.map(qa_features(tokenizer), batched=True, remove_columns=raw.column_names)

    out_dir = os.path.join(out_root, "fine_tuned_roberta")   # input to cluster stage 2
    Trainer(model=model, args=training_args(out_dir, 2e-5),
            train_dataset=feats, data_collator=default_data_collator).train()
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[roberta] saved {out_dir}")


def main():
    p = argparse.ArgumentParser(description="Local fine-tuning of candidate models")
    p.add_argument("--model", choices=["distilgpt2", "biogpt", "bert", "roberta", "all"], default="all")
    p.add_argument("--drugbank-csv", required=True)
    p.add_argument("--symptoms-csv", required=True)
    p.add_argument("--max-samples", type=int, default=2000,
                   help="Cap per dataset; local runs used limited data points.")
    p.add_argument("--out", default="./local_models")
    a = p.parse_args()

    need_text = a.model in ("distilgpt2", "biogpt", "all")
    stages = load_text_stages(a.drugbank_csv, a.symptoms_csv, a.max_samples) if need_text else None

    if a.model in ("distilgpt2", "all"):
        train_causal_lm("distilgpt2", "distilgpt2", stages, a.out)
    if a.model in ("biogpt", "all"):
        train_causal_lm("microsoft/biogpt", "biogpt", stages, a.out)
    if a.model in ("bert", "all"):
        train_bert_classifier(a.max_samples, a.out)
    if a.model in ("roberta", "all"):
        train_roberta_qa(a.symptoms_csv, a.max_samples, a.out)


if __name__ == "__main__":
    main()
