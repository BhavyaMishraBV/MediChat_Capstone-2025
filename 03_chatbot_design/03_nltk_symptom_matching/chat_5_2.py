import os
import logging
import openai
from openai import OpenAI
import torch
import difflib
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from datetime import datetime
from supabase import create_client, Client
from collections import deque


chat_history = []

stop_words = set(stopwords.words('english'))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Supabase Setup (credentials removed for publication; set as environment variables) ===
SUPABASE_URL = os.environ.get("SUPABASE_URL", "YOUR_SUPABASE_PROJECT_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "YOUR_SUPABASE_ANON_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# === Model Loading ===
save_path = "model_3_27_Bhavya"
qa_pipeline = pipeline("question-answering", model=save_path, tokenizer=save_path)
logger.info("✅ Local QA model loaded!")

# === OpenAI Setup (credentials removed for publication) ===
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)
# === Local Guide ===
local_guide = {
    "Hi": "Helllooooo, I am here to help you with any queries you have.",
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

def summarize_input(text, max_words=50):
    words = text.split()
    return " ".join(words[:max_words])


def preprocess(text):
    """Lowercase, tokenize, and remove stopwords."""
    tokens = word_tokenize(text.lower())
    filtered = [word for word in tokens if word.isalnum() and word not in stop_words]
    return filtered

def extract_symptoms(text):
    """Extract symptoms from the local_guide entry."""
    match = re.search(r"🧾 \*Symptoms:\* (.+?)\n", text)
    if match:
        symptoms = re.split(r",|;", match.group(1).lower())
        return [s.strip() for s in symptoms if s.strip()]
    return []

def find_Match(user_input, guide):
    user_keywords = preprocess(user_input)
    if not user_keywords:
        return None
    con =None
    score=0.0

    for condition, details in guide.items():
        condition_symptoms = extract_symptoms(details)
        if condition.lower()==user_input.lower():return condition
        match_count=0
        for keyword in user_keywords:
            for conditions in condition_symptoms:
                if keyword.lower() in conditions.lower():
                    match_count += 1

        match_ratio = match_count / len(user_keywords)
        if match_ratio>=0.5:
            return condition
        if score<match_ratio:
            con=condition

    return con


# Get QA response from local model

def get_qa_response(user_input):
    try:
        context = " ".join(local_guide.values())
        response = qa_pipeline(question=user_input, context=context)
        answer = response.get("answer", "").strip()
        return answer if answer and answer.lower() != "unknown" else None
    except Exception as e:
        logger.error(f"❌ QA model error: {e}")
        return None

# Polish text using OpenAI
def polish_with_openai(raw_text, chat_history=[]):
    try:
        # Format history: only first 50 words per message
        context_snippets = []
        for h in chat_history[-5:]:  # last 5 exchanges
            user_msg = " ".join(h['user'].split()[:50])
            bot_msg = " ".join(h['bot'].split()[:50])
            context_snippets.append(f"User: {user_msg}\nBot: {bot_msg}")
        history_text = "\n".join(context_snippets)

        full_prompt = history_text + f"\n\nCurrent Input: {raw_text}\nInstruction: Improve and clarify the medical suggestion."

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a medical assistant."},
                {"role": "user", "content": full_prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"❌ OpenAI API error: {e}")
        return None


def findMatch(prediction: str):
    matches = difflib.get_close_matches(prediction.lower(), local_guide.keys(), n=1, cutoff=0.5)
    return matches[0] if matches else None


# Function to extract treatment from the string
def extract_treatment(info_string):
    treatment_keyword = "💊 *Treatment:*"
    if treatment_keyword in info_string:
        treatment_part = info_string.split(treatment_keyword)[1].strip()
        return treatment_part
    return "No treatment info found. We can look further into it."

def save_chat_to_supabase(chat_history: list):
    try:
        # Format chat_history as a string (with user input and bot response)
        formatted_history = "\n".join([f"user: {item['user']}, bot: {item['bot']}" for item in chat_history])

        # Insert the formatted history into Supabase
        response = (supabase.table("chat_logs").insert({
            "timestamp": datetime.now().isoformat(),
            "chat_history": formatted_history,
            "user_id": 0000,
            "title": "Bhavya_test"
        }).execute())

        # Log the response from Supabase (optional, for debugging)
        logger.info(f"✅ Chat history saved to Supabase. Response: {response}")
    except Exception as e:
        logger.error(f"❌ Failed to save chat history: {e}")

# Example of the chatbot logic
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit" or user_input.lower() == "bye" or user_input.lower() == "goodbye" or user_input.lower() == "quit":
        break
    u=user_input
    if "pain" in u or "strain" in u or "soreness" in u or "problem" in u or 'trouble' in u:
        print("Tell me the following:\n")
        ans1 = input("1. Do you have pain? Answer: ")
        ans2 = input("2. How long has this been going around? Answer: ")
        ans3 = input("3. List all the symptoms separated by commas: Answer: ")
        ans4 = input("4. Any other remark? Answer: ")
        a = f"""Symptoms screening:
    1. Do you have pain? Answer: {ans1}
    2. How long has this been going around? Answer: {ans2}
    3. Any other remark? Answer: {ans3}"""

        if ans1.lower()=="yes": u=u + ", " + ans3 + ", pain"
        else: u= u + ", " + ans3

    best_match = find_Match(u,local_guide)

    if best_match:
        info_string = local_guide[best_match]
        treatment = extract_treatment(info_string)
        if user_input.lower()=="hi" or user_input.lower()=="hello":
            print(f"Bot: {info_string}")
        else:
            response = (
                f"🩺 **Possible Condition:** *{best_match.capitalize()}*\n\n"
                f"{info_string}\n"
                f"💊 **Treatment:** {treatment}\n"
            )
            print(f"Bot: {response}")
            chat_history.append({"user": user_input + "a", "bot":best_match})
            more = input("❓ Would you like a polished explanation? (yes/no): ").strip().lower()
            if more == 'yes':
                polished_info = polish_with_openai(info_string + a + 'instruction: respond in a line and find treatment also:')
                polished_treat = polish_with_openai(treatment)
                if polished_info:
                    print(f"✨ *Refined Info:* {polished_info}")
        continue

    else:
        qa_output = get_qa_response("find an cause in one line, and respond:"+ user_input)
        if qa_output:
            print(f"\n🩺 *Possible diagnosis:* {qa_output}")
            more = input("❓ Would you like a polished suggestion? (yes/no): ").strip().lower()
            if more == 'yes':
                print("Tell me the following first:\n")
                ans1 = input("1. Do you have pain? Answer: ")
                ans2 = input("2. How long has this been going around? Answer: ")
                ans3 = input("3. List all the symptoms separated by commas: Answer: ")
                ans4 = input("4. Any other remark? Answer: ")
                u=''
                a = f"""Symptoms screening:
            1. Do you have pain? Answer: {ans1}
            2. How long has this been going around? Answer: {ans2}
            3. Any other remark? Answer: {ans3}"""
                chat_history.append({"user": user_input + "a", "bot":best_match})
                polished = polish_with_openai(u + " "+ a)
                if polished:
                    print(f"✨ *Refined Suggestion:* {polished}")

            continue

        # Step 3: If no QA result, go straight to OpenAI
        print("🔍 Searching for suggestions using extended logic...")
        ai_output = polish_with_openai(user_input)
        if ai_output:
            print(f"🧠 *OpenAI Insight:* {ai_output}\n")
            chat_history.append({"user": user_input, "bot": ai_output})
        else:
            print("⚠️ Sorry, I couldn’t generate a response.\n")
save_chat_to_supabase(chat_history)
