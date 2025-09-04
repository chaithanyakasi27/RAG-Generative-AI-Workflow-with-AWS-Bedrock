import streamlit as st
from llm_wrapper import BedrockClient
from utils import load_text_files, load_pdf_files, chunk_text, build_or_load_faiss

# Initialize the Bedrock client
client = BedrockClient()
st.set_page_config(page_title="Bedrock AI", page_icon="🤖", layout="wide")

# App title and description
st.title("🤖 AWS Bedrock AI Playground")
# Subtitle
st.caption("Llama3 • Titan Embeddings • FAISS • Stable Diffusion")
# Tabs
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📚 RAG Q&A", "🎨 Image Gen"])

# --- Chat Tab ---
with tab1:
    # Chat interface
    st.header("Chat with Llama3")
    # Message history
    if "messages" not in st.session_state:
        # Initialize message history
        st.session_state["messages"] = []
    
    # Display message history
    for msg in st.session_state["messages"]:
        # Display each message in the chat
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    if prompt := st.chat_input("Type your question..."):
        # Append user message to history
        st.session_state["messages"].append({"role": "user", "content": prompt})
        # Display user message

        with st.chat_message("user"): st.markdown(prompt)
        # Generate assistant response
        with st.chat_message("assistant"):
            # Display thinking spinner
            with st.spinner("Thinking..."):
                # Send user message to Bedrock
                reply = client.generate_text(prompt)
                st.markdown(reply)
        # Append assistant message to history
        st.session_state["messages"].append({"role": "assistant", "content": reply})

    # Clear message history
    if st.button("🧹 Clear History"):
        st.session_state["messages"] = []

# --- RAG Q&A Tab ---
with tab2:
    # RAG Q&A interface
    st.header("Retrieval-Augmented QA")
    # File uploaders
    uploaded_txt = st.file_uploader("Upload text files", type=["txt"], accept_multiple_files=True)
    uploaded_pdfs = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

    # Load documents
    docs = []
    if uploaded_txt: docs.extend(load_text_files(uploaded_txt))
    if uploaded_pdfs: docs.extend(load_pdf_files(uploaded_pdfs))

    if docs:
        # Create document chunks and vectorstore
        chunks = [c for d in docs for c in chunk_text(d)]
        # Build or load FAISS vectorstore
        vectorstore = build_or_load_faiss(chunks)
        st.success(f"{len(chunks)} chunks embedded")

        # User query
        query = st.text_input("Ask your question:")
        if query:
            with st.spinner("Searching..."):
                # Perform similarity search
                results = vectorstore.similarity_search(query, k=3)
                # Extract context from results
                context = "\n".join([d.page_content for d in results])
                # Construct full prompt
                full_prompt = f"Context:\n{context}\n\nQuestion: {query}"
                # Generate answer
                answer = client.generate_text(full_prompt)
                st.markdown(f"**Answer:** {answer}")

                # Show sources
                for i, doc in enumerate(results, 1):
                    with st.expander(f" Source {i}"):
                        st.write(doc.page_content)

# --- Image Tab ---
with tab3:
    # Image generation interface
    st.header("Image Generation")
    # User input
    prompt = st.text_area("Enter your image prompt:", "A futuristic cityscape at night")
    # Image size
    size = st.radio("Select size:", [512, 768, 1024], index=2)
    # Image format
    format = st.selectbox("Select format:", ["PNG", "JPEG"], index=0)
    if st.button("Generate Image"):
        # Validate user input
        if not prompt:
            st.warning("Please enter an image prompt.")
        with st.spinner("Generating..."):
            # Generate image
            img_path = client.generate_image(prompt, out_dir="output_images")
            st.image(img_path, caption=prompt, use_container_width=True)
            # Show download button
            with open(img_path, "rb") as f:
                st.download_button(" Download", f, file_name="generated.png")
