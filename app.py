import os
import time
import streamlit as st
from rag_chain import build_rag_chain, REFUSAL_MESSAGE

st.set_page_config(page_title="Zyro Dynamics HR Assistant", page_icon="🏢")

st.write("DEBUG - Key loaded:", bool(os.environ.get("GROQ_API_KEY")))   # <-- add this line, remove later

st.title("🏢 Zyro Dynamics HR Assistant")

st.set_page_config(page_title="Zyro Dynamics HR Assistant", page_icon="🏢")

st.title("🏢 Zyro Dynamics HR Assistant")
st.caption("Ask questions about Zyro Dynamics HR policies. Answers are grounded in official policy documents.")

SUGGESTED_QUESTIONS = [
    "What are the IT and data security guidelines?",
    "What is the work-from-home policy?",
    "How is performance reviewed at Zyro Dynamics?",
    "What is the travel reimbursement process?",
]


@st.cache_resource(show_spinner="Loading HR knowledge base...")
def load_chain():
    return build_rag_chain()


t0 = time.time()
chain, retriever = load_chain()
print(f"[TIMING] load_chain took {time.time() - t0:.2f}s")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for src in msg["sources"]:
                    st.markdown(f"- {src}")

if not st.session_state.messages:
    st.markdown("**Try asking:**")
    cols = st.columns(3)
    for i, q in enumerate(SUGGESTED_QUESTIONS):
        if cols[i % 3].button(q, key=f"suggested_{i}"):
            st.session_state.pending_question = q

user_question = st.chat_input("Ask an HR question...")

if "pending_question" in st.session_state:
    user_question = st.session_state.pop("pending_question")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            docs = retriever.invoke(user_question)
            answer = chain.invoke(user_question).strip()

            is_refusal = answer.strip().rstrip('"').lstrip('"') == REFUSAL_MESSAGE.strip().rstrip('"').lstrip('"')

            sources = []
            if not is_refusal:
                seen = set()
                for doc in docs:
                    source = doc.metadata.get("source", "unknown")
                    if source not in seen:
                        seen.add(source)
                        sources.append(source)

            st.markdown(answer)
            if sources:
                with st.expander("Sources"):
                    for src in sources:
                        st.markdown(f"- {src}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
    st.rerun()

with st.sidebar:
    st.header("About")
    st.write(
        "This assistant answers questions using Zyro Dynamics' 11 HR policy "
        "documents: Company Profile, Employee Handbook, Leave Policy, "
        "Work From Home Policy, Code of Conduct, Performance Review Policy, "
        "Compensation & Benefits Policy, IT & Data Security Policy, "
        "POSH Policy, Onboarding & Separation Policy, and Travel & Expense Policy."
    )
    st.write("Questions outside this scope will be politely declined.")