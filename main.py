from llm_wrapper import BedrockClient, create_faiss_store

if __name__ == "__main__":
    client = BedrockClient()

    # Example 1: Llama3 text
    q = """Let's think step by step.
    Question: What is the capital of France?"""
    answer = client.generate_text(q)
    print("===== Llama3 Output =====")
    print(answer)

    # Example 2: Stable Diffusion image
    img_prompt = "4k HD image of a volcano with explosive lava flows and thunderstorm clouds"
    img_path = client.generate_image(img_prompt)
    print(f"===== Image saved at {img_path} =====")

    # Example 3: RAG (Titan embeddings + FAISS)
    docs = [
        "Paris is the capital of France.",
        "Berlin is the capital of Germany.",
        "Rome is the capital of Italy."
    ]
    vectorstore = create_faiss_store(docs)
    query = "What is the capital of France?"
    retrieved_docs = vectorstore.similarity_search(query, k=2)
    context = "\n".join([d.page_content for d in retrieved_docs])

    full_prompt = f"Context:\n{context}\n\nQuestion: {query}"
    rag_answer = client.generate_text(full_prompt)

    print("\n===== RAG Answer =====")
    print(rag_answer)
