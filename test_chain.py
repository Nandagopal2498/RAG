from rag_chain import build_rag_chain

SAMPLE_QUESTIONS = [
    "How many earned leave days do employees get per year?",
    "What is the WFH policy at Zyro Dynamics?",
    "What's the population of Japan?",
]

if __name__ == "__main__":
    print("Building RAG chain (this may take a minute on first run)...\n")
    chain, retriever = build_rag_chain()

    for q in SAMPLE_QUESTIONS:
        print(f"Q: {q}")
        answer = chain.invoke(q).strip()
        print(f"A: {answer}\n")
        print("-" * 60)