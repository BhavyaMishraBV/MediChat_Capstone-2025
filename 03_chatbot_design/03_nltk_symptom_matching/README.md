# 03: NLTK Symptom Matching

**File:** `chat_5_2.py`

## Goal
Explore a rule-based layer between the local model's prediction and the final answer: a way to check or re-route predictions, and a way to collect richer input from the user.

## Design
- **Preprocessing:** NLTK tokenizing and stopword removal on the user's input.
- **Symptom matching:** symptom lists are pulled from each guide entry, and conditions are scored by overlap with the user's keywords.
- **Intake questionnaire:** a short set of questions (pain, duration, list of symptoms) to build a fuller query.
- **Polishing:** GPT-3.5 Turbo for optional refined output.
- **Interface:** a command-line loop, with chat history saved to Supabase on exit.

## Findings
The matching acted as a stop mechanism on the model's predictions, but it re-routed outputs through fuzzy matching, and the intake questionnaire was a feature that was never introduced into the product.

## Decision
Not carried into the final version. The idea of a barrier against wrongly inferred answers carried forward in a different form: the guide became a knowledge base of pre-recorded answers in [stage 04](../04_final_mistral_api/). This folder stays because it shows an approach I evaluated and set aside.
