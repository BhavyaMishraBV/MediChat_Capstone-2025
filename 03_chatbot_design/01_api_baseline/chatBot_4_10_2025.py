import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from transformers import pipeline
from datetime import datetime
from supabase import create_client, Client
import difflib


save_path = "model_3_27_Bhavya"  # path to the fine-tuned model directory
qa_pipeline = pipeline("question-answering", model=save_path, tokenizer=save_path)
print("Model loaded!")

# Credentials removed for publication; set as environment variables
HF_API_KEY = os.environ.get("HF_API_KEY", "YOUR_HUGGINGFACE_API_KEY")
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"  # can switch to another model



SUPABASE_URL = os.environ.get("SUPABASE_URL", "YOUR_SUPABASE_PROJECT_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "YOUR_SUPABASE_ANON_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Local muscle condition guide, I have tried expanding as much as possible
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

# FastAPI app for porting to type script.
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to the frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    user_input: str
    chat_history: List[List[str]]

# Utility: find best match
def findMatch(prediction: str):
    matches = difflib.get_close_matches(prediction.lower(), local_guide.keys(), n=1, cutoff=0.5)
    return matches[0] if matches else None

# Utility: diagnosis logic
def get_diagnosis(user_input: str) -> str:
    question = f"What muscle condition could cause {user_input}?"
    context = (
        "Common muscle conditions include muscle strain, muscle spasm, tendonitis, cramps, fibromyalgia, "
        "rhabdomyolysis, sciatica, myositis, ALS, and muscular dystrophy. Each has unique causes and treatments. "
        "Symptoms like muscle pain, spasms, fatigue, stiffness, weakness, or radiating pain can indicate different conditions. "
        "Treatments vary from NSAIDs, hydration, physical therapy, corticosteroids, to medications like pregabalin or riluzole."
    )
    result = qa_pipeline(question=question, context=context)
    diagnosis = result["answer"].lower()
    best_match = findMatch(diagnosis)

    if best_match:
        response = f"🩺 **Possible Condition:** *{best_match.capitalize()}*\n\n{local_guide[best_match]}"
    else:
        response = (
            "😕 I couldn’t identify a clear match based on your symptoms.\n"
            "Please consider consulting a healthcare provider for an accurate diagnosis."
        )

    response += "\n\n📌 **Self-care Tips:**\n- Stay hydrated\n- Use proper posture\n- Avoid overexertion\n- See a specialist if symptoms persist"
    return response


def save_chat_to_supabase(chat_history: List[List[str]]):
    formatted_history = "\n".join([f"{sender}: {message}" for sender, message in chat_history])
    supabase.table("chat_logs").insert({
        "timestamp": datetime.now().isoformat(),
        "chat_history": formatted_history
    }).execute()


@app.post("/chat")
async def chat(request: ChatRequest):
    user_input = request.user_input
    chat_history = request.chat_history or []

    if "pain" in user_input.lower() or "muscle" in user_input.lower():
        response = get_diagnosis(user_input)
    else:
        response = "I'm here to help with muscle-related symptoms. Please describe your issue."

    chat_history.append([user_input, response])
    return {"response": response, "chat_history": chat_history}


@app.post("/end")
async def end_chat(request: ChatRequest):
    chat_history = request.chat_history or []
    chat_history.append(["", "👋 Bye! Take care of your health!"])
    save_chat_to_supabase(chat_history)
    return {"chat_history": chat_history}

import uvicorn

if __name__ == "__main__":
    uvicorn.run("chatBot_4_10_2025:app", host="127.0.0.1", port=8000, reload=True)
