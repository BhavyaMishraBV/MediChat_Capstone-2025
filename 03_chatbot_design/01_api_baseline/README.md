# 01: API Baseline

**File:** `chatBot_4_10_2025.py`

## Goal
Give the frontend team a working backend to build against while I was still fine-tuning and updating the model. Frontend and backend were being built in parallel, so a simple, stable interface mattered more than polish.

## Design
- A FastAPI app with `/chat` and `/end` endpoints and CORS enabled.
- A keyword check on "pain" or "muscle" decides whether to run the model.
- The fine-tuned question-answering model answers over a fixed context, and difflib fuzzy-matches the answer to a small condition guide.
- At the end of a chat, the history is saved to Supabase.

This was the same logic as my first prototype, a Gradio app from 3/26, moved behind an API.

## What worked
The frontend team could integrate immediately, and the request and response format stayed the same while the model underneath was replaced.

## Limitations
- Simple by design: one fixed context, no conversation memory, 10 conditions.
- The model's answer is only as good as the string match to the guide.

## Decision
Keep the interface stable and improve what sits behind it. Next I tested whether an LLM could make the output more useful ([stage 02](../02_hybrid_polish_experiments/)).
