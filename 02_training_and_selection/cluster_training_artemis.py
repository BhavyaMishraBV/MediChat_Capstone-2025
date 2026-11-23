"""
Cluster training on Artemis (my assigned node on the Kakos Center for Scientific
Computing cluster, Manhattan University).

Purpose: speed up the later fine-tuning stages of the selected model
(RoBERTa, started from deepset/roberta-base-squad2) after local training
selected it. The stages ran one after another, not in a single job.

    Stage 1  disease-symptom fine-tune of deepset/roberta-base-squad2
             -> saved as ./fine_tuned_roberta  (see local_training.py)
    Stage 2  PubMedQA yes/no classification            (this file, stage2)
             -> saved as ./roberta_pubmedqa_stage2
    Stage 3  short QA fine-tune on a custom medical QA-pair dataset
             ("artificial_med_qa")                     (this file, stage3)
             -> saved as ./model_3_27_Bhavya

Provenance of this file (please read):
  * Stage 2 comes from the original script, recovered from an email. One line
    that was corrupted in the copy was repaired, and the output folder was
    renamed so stage 3 can own the final model name.
  * Stage 3 is a RECONSTRUCTION. The original stage-3 script was not preserved.
    The custom QA dataset is not published (licensing of the source datasets),
    so the expected file format is documented below. Hyperparameters in
    STAGE3_CONFIG are representative values, not the original run's.
  * Written to run on a GPU node; not re-run end to end after consolidation.

Usage:
    python cluster_training_artemis.py --stage 2
    python cluster_training_artemis.py --stage 3 --qa-data data/artificial_med_qa.jsonl
    python cluster_training_artemis.py --stage all --qa-data data/artificial_med_qa.jsonl

Expected QA data format (JSON lines, SQuAD-style), one example per line:
    {"question": "...", "context": "...",
     "answers": {"text": ["..."], "answer_start": [123]}}
"""

import argparse

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    RobertaForQuestionAnswering,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    RobertaTokenizerFast,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

STAGE1_MODEL_PATH = "./fine_tuned_roberta"          # output of stage 1 (local_training.py)
STAGE2_OUTPUT_PATH = "./roberta_pubmedqa_stage2"    # output of stage 2
FINAL_MODEL_PATH = "./model_3_27_Bhavya"            # output of stage 3 (the deployed model)

# Representative values for the reconstructed stage 3 (original values not preserved).
STAGE3_CONFIG = {
    "epochs": 3,
    "learning_rate": 2e-5,
    "batch_size": 8,
    "max_length": 384,
    "doc_stride": 128,
    "validation_fraction": 0.1,
    "seed": 42,
}


# ==========================================================
# Stage 2: PubMedQA yes/no classification
# ==========================================================
def stage2_pubmedqa_classification():
    dataset = load_dataset("pubmed_qa", "pqa_artificial")
    print(dataset)

    tokenizer = RobertaTokenizer.from_pretrained(STAGE1_MODEL_PATH)

    # Load the saved RoBERTa weights and initialize a new binary classification head
    model = RobertaForSequenceClassification.from_pretrained(
        STAGE1_MODEL_PATH,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )
    model.to(device)
    print("Previously saved RoBERTa model loaded successfully!")

    def filter_binary_labels(example):
        return example["final_decision"] in ["yes", "no"]

    dataset = dataset.filter(filter_binary_labels)

    def assign_labels(example):
        # Yes = 1, No = 0
        example["labels"] = 1 if example["final_decision"] == "yes" else 0
        return example

    dataset = dataset.map(assign_labels)

    # Split dataset into training and testing sets
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")

    def preprocess_function(examples):
        return tokenizer(
            examples["question"],
            examples["long_answer"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )

    train_dataset = train_dataset.map(preprocess_function, batched=True)
    test_dataset = test_dataset.map(preprocess_function, batched=True)

    columns = ["input_ids", "attention_mask", "labels"]
    train_dataset.set_format(type="torch", columns=columns)
    test_dataset.set_format(type="torch", columns=columns)

    def compute_metrics(pred):
        predictions = pred.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predicted_labels = np.argmax(predictions, axis=1)
        accuracy = np.mean(predicted_labels == pred.label_ids)
        return {"accuracy": accuracy}

    training_args = TrainingArguments(
        output_dir="./roberta_pubmed_results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir="./roberta_pubmed_logs",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    print("\nStarting RoBERTa fine-tuning on PubMedQA...\n")
    trainer.train()

    trainer.save_model(STAGE2_OUTPUT_PATH)
    tokenizer.save_pretrained(STAGE2_OUTPUT_PATH)
    print(f"\nStage 2 model saved at: {STAGE2_OUTPUT_PATH}")

    results = trainer.evaluate()
    print("\nStage 2 evaluation results:")
    with open("./roberta_pubmed_evaluation.txt", "w") as file:
        for key, value in results.items():
            print(f"{key}: {value}")
            file.write(f"{key}: {value}\n")


# ==========================================================
# Stage 3 (reconstructed): custom medical QA fine-tune
# ==========================================================
def stage3_custom_qa(qa_data_path):
    cfg = STAGE3_CONFIG

    # Fast tokenizer is required for character-offset mapping.
    tokenizer = RobertaTokenizerFast.from_pretrained(STAGE2_OUTPUT_PATH)

    # Encoder weights come from stage 2. The classification head is dropped and a
    # fresh span-prediction (question answering) head is initialized, so this stage
    # trains that head together with the encoder on the custom QA pairs.
    model = RobertaForQuestionAnswering.from_pretrained(STAGE2_OUTPUT_PATH)
    model.to(device)

    raw = load_dataset("json", data_files=qa_data_path, split="train")
    splits = raw.train_test_split(test_size=cfg["validation_fraction"], seed=cfg["seed"])
    train_raw, val_raw = splits["train"], splits["test"]
    print(f"QA training examples: {len(train_raw)} | validation examples: {len(val_raw)}")

    def prepare_features(examples):
        questions = [q.lstrip() for q in examples["question"]]
        tokenized = tokenizer(
            questions,
            examples["context"],
            truncation="only_second",
            max_length=cfg["max_length"],
            stride=cfg["doc_stride"],
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized.pop("offset_mapping")
        tokenized["start_positions"] = []
        tokenized["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            sequence_ids = tokenized.sequence_ids(i)
            answers = examples["answers"][sample_mapping[i]]

            if len(answers["answer_start"]) == 0:
                tokenized["start_positions"].append(cls_index)
                tokenized["end_positions"].append(cls_index)
                continue

            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Locate the context span inside this tokenized window
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # If the answer is not fully inside this window, label it as "no answer"
            if not (offsets[token_start_index][0] <= start_char
                    and offsets[token_end_index][1] >= end_char):
                tokenized["start_positions"].append(cls_index)
                tokenized["end_positions"].append(cls_index)
            else:
                while (token_start_index < len(offsets)
                       and offsets[token_start_index][0] <= start_char):
                    token_start_index += 1
                tokenized["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized["end_positions"].append(token_end_index + 1)

        return tokenized

    train_features = train_raw.map(
        prepare_features, batched=True, remove_columns=train_raw.column_names
    )
    val_features = val_raw.map(
        prepare_features, batched=True, remove_columns=val_raw.column_names
    )

    training_args = TrainingArguments(
        output_dir="./roberta_medqa_results",
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir="./roberta_medqa_logs",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=cfg["seed"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_features,
        eval_dataset=val_features,
        data_collator=default_data_collator,
    )

    print("\nStarting RoBERTa QA fine-tuning on the custom medical QA pairs...\n")
    trainer.train()

    trainer.save_model(FINAL_MODEL_PATH)
    tokenizer.save_pretrained(FINAL_MODEL_PATH)
    print(f"\nFinal model saved at: {FINAL_MODEL_PATH}")

    results = trainer.evaluate()
    print("\nStage 3 evaluation results:")
    for key, value in results.items():
        print(f"{key}: {value}")


def main():
    parser = argparse.ArgumentParser(description="Artemis cluster training stages")
    parser.add_argument("--stage", choices=["2", "3", "all"], default="all")
    parser.add_argument("--qa-data", default="data/artificial_med_qa.jsonl",
                        help="Custom QA pairs in JSON lines format (not distributed).")
    args = parser.parse_args()

    if args.stage in ("2", "all"):
        stage2_pubmedqa_classification()
    if args.stage in ("3", "all"):
        stage3_custom_qa(args.qa_data)


if __name__ == "__main__":
    main()
