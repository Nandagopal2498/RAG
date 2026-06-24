import csv
import base64
import sys
import time
from pathlib import Path
from rag_chain import build_rag_chain

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "submission.csv"

QUESTIONS = {
    "Q01": "How many days of Earned Leave (EL) are employees entitled to per year?",
    "Q02": "What is the process for applying for maternity leave?",
    "Q03": "Can employees work from home, and what is the approval process?",
    "Q04": "What behavior is considered a violation of the Code of Conduct?",
    "Q05": "How does the Annual Performance Review (APR) process work?",
    "Q06": "What components make up an employee's CTC?",
    "Q07": "What are the rules for using company devices and handling data?",
    "Q08": "How can an employee file a complaint under the POSH policy?",
    "Q09": "What is the probation period for new employees?",
    "Q10": "How are travel expenses reimbursed?",
    "Q11": "What is the capital of France?",
    "Q12": "Can you write me a Python script to sort a list?",
    "Q13": "What is the weather like today?",
    "Q14": "Who won the last football World Cup?",
    "Q15": "Give me a recipe for chocolate cake.",
}


def encode(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("utf-8")


def generate_submission(streamlit_link: str, langsmith_link: str):
    if not streamlit_link or not langsmith_link:
        print("ERROR: Both streamlit_link and langsmith_link are required.")
        sys.exit(1)

    chain, _ = build_rag_chain()

    fieldnames = [
        "question_id",
        "question_enc",
        "answer_enc",
        "streamlit_link",
        "langsmith_link",
    ]

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for qid, question in QUESTIONS.items():
            print(f"Answering {qid}...")
            answer = None
            for attempt in range(3):
                try:
                    answer = chain.invoke(question).strip()
                    break
                except Exception as exc:
                    print(f"  attempt {attempt + 1} failed: {exc}")
                    time.sleep(2 ** attempt)

            if answer is None:
                answer = "ERROR: failed to generate answer after retries"
                print(f"  {qid} FAILED, writing error placeholder")

            writer.writerow(
                {
                    "question_id": qid,
                    "question_enc": encode(question),
                    "answer_enc": encode(answer),
                    "streamlit_link": streamlit_link,
                    "langsmith_link": langsmith_link,
                }
            )
            f.flush()

    print(f"\nSubmission written to {OUTPUT_PATH}")
    print(f"Rows: {len(QUESTIONS)} (expected 15)")


if __name__ == "__main__":
    streamlit_url = input("Paste your Streamlit app URL: ").strip()
    langsmith_url = input("Paste your LangSmith trace URL: ").strip()
    generate_submission(streamlit_url, langsmith_url)