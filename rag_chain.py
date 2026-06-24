from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from config import GROQ_API_KEY, LLM_MODEL
from vector_store import get_retriever

REFUSAL_MESSAGE = (
    "I can only answer HR-related questions from Zyro Dynamics policy documents. "
    "This question falls outside the scope of the available HR policies, so I'm "
    "unable to answer it."
)

RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are an HR helpdesk assistant for Zyro Dynamics. Answer the employee's
question using ONLY the context below, which comes from official HR policy
documents. Do not use outside knowledge.

If the context does not contain enough information to answer the question,
respond exactly with:
"{refusal}"

Context:
{context}

Question:
{question}

Answer clearly and concisely, citing the relevant policy where useful.

IMPORTANT FORMATTING RULE: When stating a leave entitlement or benefit, lead
with the policy name and the number of days/amount, kept short and scannable.
Do not include eligibility conditions, qualifiers, or any parenthetical notes
about eligibility in your answer — just state the entitlement plainly.

Example:
Source text: "Paternity Leave: 10 days (for male employees and same-sex partners)"
Correct answer: "Paternity Leave: 10 days"

If the employee explicitly asks "who is eligible" or "what are the
conditions," give the full eligibility detail from the source document.
"""
)


def format_docs(docs):
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n".join(formatted)


def build_rag_chain(force_rebuild: bool = False):
    retriever = get_retriever(force_rebuild=force_rebuild)
    llm = ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY, temperature=0)

    chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(
            lambda x: {
                "context": x["context"],
                "question": x["question"],
                "refusal": REFUSAL_MESSAGE,
            }
        )
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def answer_question(chain, question: str) -> str:
    try:
        return chain.invoke(question).strip()
    except Exception as exc:
        return f"Error generating answer: {exc}"