from rag_pipeline import RAGPipeline


def main():
    print("=" * 55)
    print("       ATTENTION PAPER - RAG ASSISTANT")
    print("=" * 55)
    print("Ask questions about the PDF.")
    print("Type 'exit' to quit.\n")

    rag = RAGPipeline()

    while True:
        question = input("\nYour question: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            answer, sources = rag.ask(question)

            print("\nAnswer:")
            print(answer)

            print("\nRetrieved sources:")
            if sources:
                for source in sources:
                    print(
                        f"- {source['file']} | "
                        f"Page {source['page']}"
                    )
            else:
                print("- No sources found.")

        except Exception as error:
            print(f"\nSomething went wrong: {error}")


if __name__ == "__main__":
    main()
