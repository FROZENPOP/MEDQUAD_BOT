import os
import pickle

import faiss
import streamlit as st
from dotenv import load_dotenv
from groq import APIError, Groq
from sentence_transformers import SentenceTransformer

load_dotenv()
# Groq retires models regularly, so this can change without a code edit.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# === STREAMLIT UI ===
st.set_page_config(page_title="Medical QA Bot", page_icon="💬")
st.title("Medical Question Answering Bot")
st.caption(
    "For general information only. This is not medical advice. "
    "In an emergency, contact your local emergency services."
)

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("GROQ_API_KEY is not set. Copy `.env.example` to `.env` and add your key.")
    st.stop()


# Streamlit reruns this script on every interaction, so load these only once.
@st.cache_resource(show_spinner="Loading the medical index and embedding model...")
def load_resources():
    index = faiss.read_index("medquad_index.faiss")
    with open("medquad_answers.pkl", "rb") as f:
        metadata = pickle.load(f)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return index, metadata["questions"], metadata["answers"], embedder


index, questions, answers, embedder = load_resources()
client = Groq(api_key=api_key)

if "messages" not in st.session_state:
    st.session_state.messages = []  # conversation history

# Sidebar with toggles and settings
with st.sidebar:
    st.header("Settings")
    temperature = st.slider("Response Temperature", 0.0, 1.0, 0.3, 0.05)
    max_tokens = st.slider("Max response length (tokens)", 50, 2000, 512, 50)
    response_style = st.radio("Response Style", ["Detailed", "General"], index=0)
    top_k = st.slider("Number of relevant entries to retrieve (Top K)", min_value=1, max_value=10, value=3)
    show_source = st.checkbox("Show source context with answers", value=False)

    if st.button("Reset Conversation"):
        st.session_state.messages = []


def show_message(message):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if show_source and "source" in message:
            st.markdown("**Source context:**")
            for q, a in message["source"]:
                st.markdown(f"- **Q:** {q}")
                st.markdown(f"  \n**A:** {a}")


# Show conversation history
for message in st.session_state.messages:
    show_message(message)

# User input
user_input = st.chat_input("Ask a medical question...")

if user_input:
    user_message = {"role": "user", "content": user_input}
    st.session_state.messages.append(user_message)
    show_message(user_message)

    # FAISS search for top-k context
    query_vec = embedder.encode([user_input]).astype("float32")
    _, neighbors = index.search(query_vec, k=top_k)
    contexts = [(questions[idx], answers[idx]) for idx in neighbors[0]]

    # Prepare context string for prompt
    context_text = "Retrieved from medical data:\n\n" + "\n\n".join(f"Q: {q}\nA: {a}" for q, a in contexts)

    # Modify instructions based on response style
    if response_style == "Detailed":
        style_instruction = "Provide a thorough and comprehensive response using all relevant medical context."
    else:
        style_instruction = "Respond concisely, summarizing key points from the context without excessive detail."

    base_prompt = f"""
You are a medical assistant AI that prioritizes factual accuracy and only answers medical questions using provided context from a trusted dataset. The context comes first in the user's message, followed by their question.

If the question is a general greeting or casual message (like "hello", "thanks", or "how are you?"), respond politely and conversationally.

If a medical question is asked but is NOT covered by the provided context answer with a general explanation and say before answering:
"I do not have the medical data to answer that, but I can offer this general explanation. However, I heavily recommend speaking with a medical professional for accurate information."

If the message describes a possible medical emergency, tell the user to contact emergency services immediately.

{style_instruction}
"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": base_prompt},
                {"role": "user", "content": f"{context_text}\n\n{user_input}\n"},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except APIError as e:
        st.error(f"Groq request failed: {e}")
        st.stop()

    # Reasoning models spend tokens before answering, so a low limit can leave no text.
    reply = response.choices[0].message.content or "The model returned no text. Try raising the max response length."

    assistant_message = {"role": "assistant", "content": reply, "source": contexts}
    st.session_state.messages.append(assistant_message)
    show_message(assistant_message)


# Download conversation transcript helper
def get_transcript():
    lines = []
    for msg in st.session_state.messages:
        lines.append(f"{msg['role'].capitalize()}: {msg['content']}\n")
        if show_source and "source" in msg:
            lines.append("Source context:\n")
            for q, a in msg["source"]:
                lines.append(f"Q: {q}\nA: {a}\n")
            lines.append("\n")
    return "".join(lines)


if st.session_state.messages:
    st.download_button(
        label="Download conversation transcript",
        data=get_transcript(),
        file_name="medquad_conversation.txt",
        mime="text/plain",
    )
