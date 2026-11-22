# Chatbot Design: From API Stub to Hybrid System

This section documents how the MediChat chatbot was built, stage by stage. The frontend and backend of MediChat were developed at the same time, so the chatbot had to be delivered in working pieces the frontend team could integrate while I was still training and improving the model. Each folder is one stage, with a short README on what I tried, what I found, and what I decided.

The full project (frontend, backend, deployment) lives in the team repository on GitLab. This repo contains only the chatbot work I own.

## Timeline

| Stage | Files | What it is |
|---|---|---|
| Prototype (3/26, not included) | Gradio app | First end-to-end test: the fine-tuned model answers over a fixed context, difflib fuzzy-matches the answer to a small condition guide, and chats are logged to Supabase. Replaced within two weeks. |
| [01 API baseline](01_api_baseline/) | `chatBot_4_10_2025.py` | The same logic behind FastAPI endpoints so the frontend team had something to build against. |
| [02 Hybrid polish experiments](02_hybrid_polish_experiments/) | `chatBot_4_22_2025.py`, `chatBot_05_1_2025.py` | Testing the hybrid idea: local model plus an LLM that polishes its output. Zephyr-7B first, then GPT-3.5 Turbo. |
| [03 NLTK symptom matching](03_nltk_symptom_matching/) | `chat_5_2.py` | An exploration of rule-based matching and an intake questionnaire as a barrier on the model's predictions. |
| [04 Final](04_final_mistral_api/) | `chatbot_2.py` | The version the team shipped: fine-tuned model, a knowledge-base guide, and Mistral-7B-Instruct. |

## Key decisions

1. **Ship a simple API early.** The frontend could not wait for a finished model, so the first FastAPI version was deliberately simple and kept the request and response shape stable while the model behind it improved.
2. **Use an LLM to polish, not to diagnose.** The fine-tuned local model produces the prediction; a hosted LLM trims, fills in and polishes the output. I tested this hybrid idea with Zephyr-7B, then GPT-3.5 Turbo, and settled on Mistral-7B-Instruct for a stable, deployable setup.
3. **Turn the guide into a knowledge base.** In early stages the condition guide routed the user's input. In the final design it holds pre-recorded answers that the model's prediction is matched against, acting as a barrier against wrongly inferred answers. It could hold expert or doctor-written answers if a client wanted that.
4. **Document the experiments that did not ship.** Zephyr and the NLTK approach were not adopted. They stay here because they show what was tried and why the final design looks the way it does.

## Scope and limitations

This was an engineering project built in one semester to deliver an end-to-end product, not a clinical system. It is not a medical device and its output should not be used for medical decisions.

- The guide covers 10 muscle conditions.
- Requests are routed by simple keyword checks.
- Output from the model and the hosted LLM is not clinically validated.
- CORS is open for development and would be restricted in production.
- The code needs the fine-tuned model directory `model_3_27_Bhavya` (see the training section).

## About the code

API keys, database URLs and personal file paths have been replaced with environment variables or placeholders. Otherwise the files are kept as they were written at the time, so they show the real progression, including rough edges.

I am continuing this line of work as part of my own research. This repo is about showing model training and deployment in chatbot form.
