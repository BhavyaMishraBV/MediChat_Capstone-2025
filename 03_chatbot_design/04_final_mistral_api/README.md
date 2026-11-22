# 04: Final Version

**File:** `chatbot_2.py`

## Goal
A stable, deployable hybrid for the team's frontend: the fine-tuned local model predicts, a knowledge base guards against wrong answers, and an LLM trims, fills in and polishes the output.

## Design
1. A FastAPI `/chat` endpoint receives the user's message and the last five conversation turns as context.
2. If the message mentions pain or muscles, the fine-tuned model predicts a condition.
3. The prediction is matched against the **local guide**, which now works as a **knowledge base**: a set of pre-recorded answers acting as a stop barrier against wrongly inferred answers. It could hold expert or doctor-written answers if a client wants them. If nothing matches, the user gets a safe fallback with self-care tips.
4. **Mistral-7B-Instruct** (Hugging Face Inference API) adds further insight. Other messages go to Mistral directly.
5. `/end` saves the conversation to Supabase.

## What changed from earlier stages
- The guide moved from **router** (checking the user's input first) to **knowledge base** (checking the model's prediction).
- The polishing model settled on Mistral after Zephyr and GPT-3.5 ([stage 02](../02_hybrid_polish_experiments/)).
- The API shape from [stage 01](../01_api_baseline/) stayed intact, so the frontend integration never broke.

## Limitations
- The knowledge base covers 10 muscle conditions.
- Routing relies on simple keywords.
- The hosted LLM's insight is not clinically validated.
- Not a medical device.
