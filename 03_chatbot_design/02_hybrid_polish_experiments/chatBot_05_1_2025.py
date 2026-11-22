import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from transformers import pipeline
from datetime import datetime
from supabase import create_client, Client
import difflib
import openai
import asyncio
import logging

# === Logging Setup ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Model Loading ===
save_path = "model_3_27_Bhavya"
qa_pipeline = pipeline("question-answering", model=save_path, tokenizer=save_path)
logger.info("✅ Local QA model loaded!")

# === Supabase Setup (credentials removed for publication; set as environment variables) ===
SUPABASE_URL = os.environ.get("SUPABASE_URL", "YOUR_SUPABASE_PROJECT_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "YOUR_SUPABASE_ANON_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# === OpenAI Setup (credentials removed for publication) ===
openai.api_key = os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

# === Local Guide ===
local_guide = {
    "Hi": "Hi, I am here to help you with any queries you have.",
    "Hello": "Hey there, I am here to help you with any queries you have.",
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

# === Utility Functions ===

def findMatch(prediction: str):
    matches = difflib.get_close_matches(prediction.lower(), local_guide.keys(), n=1, cutoff=0.5)
    return matches[0] if matches else None

def call_openai_api(prompt: str, model: str = "gpt-3.5-turbo") -> str:
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        logger.error(f"❌ OpenAI API error: {e}")
        return "⚠️ OpenAI model failed to generate a response."

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

    guide_match = findMatch(user_input)
    guide_response = ""
    formatted_local_model_response = ""

    if guide_match:
        guide_info = f"🩺 **Possible Condition:** *{guide_match.capitalize()}*\n\n{local_guide[guide_match]}"
        guide_response += guide_info

        local_question = f"Format this in one line answer: What medicine or treatment is useful for {guide_match}?"
        result = qa_pipeline(question=local_question, context="This context is related to muscle diseases and treatments.")
        local_answer = result["answer"].strip()

        if local_answer:
            polish_prompt = f"Correct the following medical sentence for proper English and clarity: \"{local_answer}\""
            formatted_local_model_response = call_openai_api(polish_prompt)

            guide_response += f"\n\n💊 *Suggested Treatment:* {formatted_local_model_response}"

        guide_response += "\n\n❓ Would you like additional insight? (Reply with 'yes' or 'no')"
        chat_history.append([user_input, guide_response])
        return {"response": guide_response, "chat_history": chat_history}

    else:
        question = f"Format this in one line answer: What muscle condition could cause: {user_input}?"
        result = qa_pipeline(question=question, context="This context is related to muscle conditions and symptoms.")
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
        prompt = f"{history_context}\nUser: {user_input}\nAssistant:"
        response = call_openai_api(prompt)
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
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("chatBot_05_1_2025:app", host="0.0.0.0", port=8000, reload=True)
