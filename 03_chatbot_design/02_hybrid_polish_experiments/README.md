# 02: Hybrid Polish Experiments

**Files:** `chatBot_4_22_2025.py` (Zephyr-7B) and `chatBot_05_1_2025.py` (GPT-3.5 Turbo)

## Goal
Test the hybrid idea: let the local fine-tuned model produce the answer and use a hosted LLM to improve how it reads.

## Design
Both files share the same architecture, so the only real variable is the polishing model.

- The user's input is checked against the local guide first.
- On a match, the local model is asked for a one-line treatment answer, and the LLM corrects and clarifies that sentence.
- With no match, the local model gives a short diagnosis.
- Each response ends by asking whether the user wants more insight, handled by a `/chat/continue` endpoint.

| | `chatBot_4_22_2025.py` | `chatBot_05_1_2025.py` |
|---|---|---|
| Polishing model | Zephyr-7B beta (Hugging Face Inference API) | GPT-3.5 Turbo (OpenAI API) |
| Prompting | Chat-style system prompt for a muscle-focused assistant | System prompt, temperature 0.7 |
| Extras | Model availability check, CLI test mode | Server bound to all interfaces |

## Findings
- **Zephyr-7B** was a lightweight way to test the hybrid theory, but the 7B model needed more resources than it was worth and the results were not promising. Its output had gaps between sentences.
- **GPT-3.5 Turbo** was used as the next polishing model, with the rest of the pipeline unchanged.

## Decision
The hybrid approach was worth keeping. The polishing model was swappable, and the final version moved to Mistral-7B-Instruct, which gave better polishing and a more stable deployment ([stage 04](../04_final_mistral_api/)).
