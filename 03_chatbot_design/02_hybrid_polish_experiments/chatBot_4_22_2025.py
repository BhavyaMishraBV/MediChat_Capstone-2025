import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from transformers import pipeline
from datetime import datetime
from supabase import create_client, Client
import difflib
import requests
import asyncio
import logging

# === Logging Setup ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Model Loading ===
save_path = "model_3_27_Bhavya"
qa_pipeline = pipeline("question-answering", model=save_path, tokenizer=save_path)
logger.info("✅ Local QA model loaded!")

# Credentials removed for publication; set as environment variables
HF_API_KEY = os.environ.get("HF_API_KEY", "YOUR_HUGGINGFACE_API_KEY")
HF_MODEL = "HuggingFaceH4/zephyr-7b-beta"

SUPABASE_URL = os.environ.get("SUPABASE_URL", "YOUR_SUPABASE_PROJECT_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "YOUR_SUPABASE_ANON_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def check_model_availability():
    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        logger.info(f"✅ Hugging Face model '{HF_MODEL}' is available!")
    else:
        logger.error(f"❌ Failed to access Hugging Face model '{HF_MODEL}'. Status code: {response.status_code}")

check_model_availability()

local_guide = {
    "Hi":"Hi, my I am here to help you with any queries you have",
    "Hello":"Hey there, my I am here to help you with any queries you have",
    "muscle strain": "💥 *Cause:* Overstretching or tearing of muscle fibers.\n🧾 *Symptoms:* Sharp pain, swelling, limited range of motion.\n💊 *Treatment:* RICE, ibuprofen, stretching.",
    "muscle spasm": "⚡ *Cause:* Dehydration, overuse, or low electrolytes.\n🧾 *Symptoms:* Sudden, involuntary contraction.\n💊 *Treatment:* Hydration, magnesium, massage.",
    "tendonitis": "🔁 *Cause:* Repetitive movement causing tendon inflammation.\n🧾 *Symptoms:* Dull ache, tenderness, swelling.\n💊 *Treatment:* NSAIDs, rest, PT.",
    "cramp": "🌀 *Cause:* Poor circulation, dehydration.\n🧾 *Symptoms:* Tightness or knotting.\n💊 *Treatment:* Potassium, hydration, massage.",
    "fibromyalgia": "🌫️ *Cause:* Unknown; maybe genetic.\n🧾 *Symptoms:* Widespread pain, fatigue.\n💊 *Treatment:* Pregabalin, exercise, CBT.",
    "rhabdomyolysis": "🚨 *Cause:* Muscle breakdown from exertion or trauma.\n🧾 *Symptoms:* Dark urine, weakness.\n💊 *Treatment:* Emergency care, IV fluids.",
    "sciatica": "🔌 *Cause:* Herniated disk pressing on nerve.\n🧾 *Symptoms:* Pain down leg, numbness.\n💊 *Treatment:* NSAIDs, PT, gabapentin.",
    "myositis": "🔥 *Cause:* Autoimmune or viral inflammation.\n🧾 *Symptoms:* Weakness, fatigue.\n💊 *Treatment:* Corticosteroids, immunosuppressants.",
    "als (amyotrophic lateral sclerosis)": "🧠 *Cause:* Neurodegenerative.\n🧾 *Symptoms:* Weakness, twitching, swallowing issues.\n💊 *Treatment:* Riluzole, edaravone.",
    "muscular dystrophy": "🧬 *Cause:* Genetic mutation.\n🧾 *Symptoms:* Muscle wasting, respiratory issues.\n💊 *Treatment:* Corticosteroids, physiotherapy."
}

# === FastAPI Setup ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    user_input: str
    chat_history: List[List[str]]

# === Utilities ===
def findMatch(prediction: str):
    matches = difflib.get_close_matches(prediction.lower(), local_guide.keys(), n=1, cutoff=0.5)
    return matches[0] if matches else None

def get_diagnosis_with_context(user_input: str, history_context: str) -> str:
    question = f"What muscle condition could cause: {user_input}?"
    extended_context = (
        history_context + "\n" +
        "Common muscle conditions include muscle strain, muscle spasm, tendonitis, cramps, fibromyalgia, "
        "rhabdomyolysis, sciatica, myositis, ALS, and muscular dystrophy. "
        "Each has unique causes and treatments."
    )
    result = qa_pipeline(question=question, context=extended_context)
    diagnosis = result["answer"].lower()
    best_match = findMatch(diagnosis)

    if best_match:
        return f"🩺 **Possible Condition:** *{best_match.capitalize()}*\n\n{local_guide[best_match]}"
    else:
        return (
            "😕 I couldn’t identify a clear match based on your symptoms.\n"
            "Please consider consulting a healthcare provider.\n\n"
            "📌 **Self-care Tips:**\n- Stay hydrated\n- Use proper posture\n- Avoid overexertion\n- See a specialist if symptoms persist"
        )

def call_mistral_model(prompt: str) -> str:
    url = f"https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }

    # Add system message and format as chat-style dialogue
    full_prompt = f"""<|system|>\nYou are a helpful and empathetic AI medical assistant. You specialize in diagnosing muscle-related symptoms and conditions.\n<|user|>\n{prompt}\n<|assistant|>"""

    payload = {"inputs": full_prompt}

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        logger.info("✅ Zephyr model responded successfully.")
        model_output = response.json()
        return model_output[0]["generated_text"].split("<|assistant|>")[-1].strip()
    else:
        logger.error(f"❌ Error {response.status_code}: {response.text}")
        return "⚠️ There was an issue retrieving the model's output."

def save_chat_to_supabase(chat_history: List[List[str]]):
    formatted_history = "\n".join([f"{sender}: {message}" for sender, message in chat_history])
    supabase.table("chat_logs").insert({
        "timestamp": datetime.now().isoformat(),
        "chat_history": formatted_history
    }).execute()

# === Chat Route ===


@app.post("/chat")
async def chat(request: ChatRequest):
    user_input = request.user_input
    chat_history = request.chat_history or []

    # === Step 1: Check local guide first ===
    guide_match = findMatch(user_input)
    guide_response = ""
    formatted_local_model_response = ""

    if guide_match:
        # Local guide content
        guide_info = f"🩺 **Possible Condition:** *{guide_match.capitalize()}*\n\n{local_guide[guide_match]}"
        guide_response += guide_info

        # === Step 2: Also run local model for short medicine/diagnosis ===
        local_question = f"Format this in one line answer: What medicine or treatment is useful for {guide_match}?"
        result = qa_pipeline(question=local_question, context="This context is related to muscle diseases and treatments.")


        # Pass local model's raw response to HF to polish
        local_answer = result["answer"].strip()
        if local_answer:
            hf_polish_prompt = f"Correct the following medical sentence for proper English and clarity. If it is irrelevant to muscles or doesn't make sense, say so: \"{local_answer}\""
            formatted_local_model_response = call_mistral_model(hf_polish_prompt).strip()

            guide_response += f"\n\n💊 *Suggested Treatment:* {formatted_local_model_response}"

        guide_response += "\n\n❓ Would you like additional insight? (Reply with 'yes' or 'no')"

        chat_history.append([user_input, guide_response])
        return {"response": guide_response, "chat_history": chat_history}

    else:
        # === Step 3: Fallback to local model if no guide match ===
        question = f"Format this in one line answer: What muscle condition could cause: {user_input}?"
        #result = qa_pipeline(question=question, context="")
        result = qa_pipeline(question=question,context="This context is related to muscle conditions and symptoms.")

        answer = result["answer"].strip().lower()

        if answer and "not sure" not in answer and "unknown" not in answer and len(answer) > 5:
            response = f"🩺 *Possible diagnosis:* {answer.capitalize()}.\n\n❓ Would you like additional insight? (Reply with 'yes' or 'no')"
        else:
            response = "😕 I couldn’t determine a clear condition from your input.\n\n❓ Would you like additional insight? (Reply with 'yes' or 'no')"

        chat_history.append([user_input, response])
        return {"response": response, "chat_history": chat_history}

@app.post("/chat/continue")
async def continue_chat(request: ChatRequest):
    user_input = request.user_input.strip().lower()
    chat_history = request.chat_history or []
    history_context = "\n".join([f"User: {u}\nBot: {b}" for u, b in chat_history[-5:]])

    if user_input == "yes":
        hf_prompt = f"{history_context}\nUser: {user_input}\nAssistant:"
        response = call_mistral_model(hf_prompt)
    elif user_input == "no":
        response = "👍 Alright. Let me know if you need help with anything else!"
    else:
        response = "❓ Please reply with 'yes' or 'no'."

    chat_history.append([user_input, response])
    return {"response": response, "chat_history": chat_history}

@app.post("/end")
async def end_chat(request: ChatRequest):
    chat_history = request.chat_history or []
    chat_history.append(["", "👋 Bye! Take care of your health!"])
    save_chat_to_supabase(chat_history)
    return {"chat_history": chat_history}

# === CLI Test Mode ===
def test_main():
    logger.info("=== Running CLI Test ===")
    chat_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            break

        test_request = ChatRequest(user_input=user_input, chat_history=chat_history)
        response = asyncio.run(chat(test_request))
        bot_response = response["response"]
        chat_history = response["chat_history"]

        print(f"\nBot: {bot_response}\n")

# === Entry Point ===
if __name__ == "__main__":
    mode = "test"  # Set to "prod" to run FastAPI server
    if mode == "test":
        test_main()
    else:
        import uvicorn
        uvicorn.run("chatBot_4_22_2025:app", host="127.0.0.1", port=8000, reload=True)
