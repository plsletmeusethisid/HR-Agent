import anthropic
from knowledge_base import search, get_stats
from config import ANTHROPIC_API_KEY

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

MODEL = "claude-sonnet-4-20250514"

SYSTEM_PROMPT = """
You are a helpful HR Agent for Shinwootns, a small cloud security company based in South Korea.

You assist employees with human resources related questions based strictly on company HR documents provided to you.

Your areas of expertise include:
- Annual leave and vacation policies
- Onboarding procedures for new employees
- HR policies and employee handbook
- Performance management processes
- Working hours and remote work policies
- Employee benefits and allowances

Rules:
- Only answer based on the document context provided to you
- Always cite which document your answer came from using [Source: filename]
- If the answer is not in the provided documents, say:
  "I couldn't find this information in the company HR documents.
   Please contact the HR department directly."
- Never make up or infer information not explicitly in the documents
- Be warm, professional, and supportive in tone
- If asked about sensitive information (salary details, personal employee data, disciplinary records), decline politely and direct the employee to HR directly
- For urgent HR matters, always recommend speaking with a manager or HR representative directly
"""

def build_context(chunks: list) -> str:
    if not chunks:
        return "No relevant documents found in the knowledge base."

    context = "Relevant information retrieved from HR documents:\n\n"
    for i, chunk in enumerate(chunks, 1):
        context += f"[{i}] Source: {chunk['source']}\n"
        context += f"{chunk['content']}\n\n"
    return context.strip()

def ask(question: str, conversation_history: list = None) -> tuple:
    if conversation_history is None:
        conversation_history = []

    print("  🔍 Searching HR documents...")
    chunks  = search(question, n_results=5)
    context = build_context(chunks)

    if chunks:
        sources = list(set(c["source"] for c in chunks))
        print(f"  📄 Found in: {', '.join(sources)}")
    else:
        print("  ⚠️  No relevant documents found")

    user_message = f"""Question: {question}

---
{context}
---

Please answer the question based only on the document context above.
Cite your sources."""

    messages = conversation_history + [
        {"role": "user", "content": user_message}
    ]

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages
    )

    answer = response.content[0].text

    updated_history = conversation_history + [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": answer}
    ]

    return answer, updated_history

def chat_loop():
    stats = get_stats()

    print("\n" + "=" * 60)
    print("👥  HR AGENT — Shinwootns")
    print("=" * 60)

    if stats["total_chunks"] == 0:
        print("⚠️  Knowledge base is empty!")
        print("   Run: python sync_and_learn.py first\n")
        return

    print(f"📚 Knowledge base: {stats['total_chunks']} chunks loaded")
    print(f"🤖 Model: {MODEL}")
    print("Ask me anything about HR policies and procedures.")
    print("Type 'exit' to quit\n")

    history = []

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if not question:
            continue

        if question.lower() in ("exit", "quit"):
            print("👋 Goodbye!")
            break

        print()
        answer, history = ask(question, history)
        print(f"\n👥 HR Agent: {answer}\n")
        print("-" * 60)

if __name__ == "__main__":
    chat_loop()
