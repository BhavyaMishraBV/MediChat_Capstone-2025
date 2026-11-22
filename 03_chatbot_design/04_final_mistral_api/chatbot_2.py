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

# === Model Loading ===
# Path to the fine-tuned model directory (set MODEL_PATH, or place the model next to this file)
save_path = os.environ.get("MODEL_PATH", "model_3_27_Bhavya")
qa_pipeline = pipeline("question-answering", model=save_path, tokenizer=save_path)
print("Local QA model loaded!")

# Credentials removed for publication; set as environment variables
HF_API_KEY = os.environ.get("HF_API_KEY", "YOUR_HUGGINGFACE_API_KEY")
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"



SUPABASE_URL = os.environ.get("SUPABASE_URL", "YOUR_SUPABASE_PROJECT_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "YOUR_SUPABASE_ANON_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# === Local Muscle Condition Guide ===
local_guide = {
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
    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"inputs": prompt}

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        model_output = response.json()
        return model_output[0]["generated_text"] if isinstance(model_output, list) else str(model_output)
    else:
        return "⚠️ There was an issue with the language model."

def save_chat_to_supabase(chat_history: List[List[str]]):
    formatted_history = "\n".join([f"{sender}: {message}" for sender, message in chat_history])
    supabase.table("chat_logs").insert({
        "timestamp": datetime.now().isoformat(),
        "chat_history": formatted_history
    }).execute()

# === Routes ===
@app.post("/chat")
async def chat(request: ChatRequest):
    user_input = request.user_input
    chat_history = request.chat_history or []

    # Create context from chat history
    history_context = "\n".join([f"User: {u}\nBot: {b}" for u, b in chat_history[-5:]])

    if "pain" in user_input.lower() or "muscle" in user_input.lower():
        response_diagnosis = get_diagnosis_with_context(user_input, history_context)
        mistral_prompt = f"{history_context}\nUser: {user_input}\nAssistant:"
        response_mistral = call_mistral_model(mistral_prompt)
        response = f"{response_diagnosis}\n\n🧠 *Further Insight:* {response_mistral.strip()}"
    else:
        mistral_prompt = f"{history_context}\nUser: {user_input}\nAssistant:"
        response = call_mistral_model(mistral_prompt)

    chat_history.append([user_input, response])
    return {"response": response, "chat_history": chat_history}

@app.post("/end")
async def end_chat(request: ChatRequest):
    chat_history = request.chat_history or []
    chat_history.append(["", "👋 Bye! Take care of your health!"])
    save_chat_to_supabase(chat_history)
    return {"chat_history": chat_history}

# === Dev Server (Only if run directly) ===
import uvicorn
if __name__ == "__main__":
    uvicorn.run("chatbot_2:app", host="127.0.0.1", port=8000, reload=True)
