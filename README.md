# 🖼️ RAG + Generative AI Workflow with AWS Bedrock

This project implements an **end-to-end Retrieval-Augmented Generation (RAG) pipeline** combined with **Generative AI for text and image generation**, all orchestrated inside a **Docker container**. It leverages **AWS Bedrock models** (Titan, LLaMA3, Stable Diffusion) for embedding, reasoning, and image generation.

---

## 🚀 Features

- **Document Ingestion & Processing**
  - Upload PDF or text documents
  - Automatic text chunking and preprocessing
  - Embeddings generated using **Titan Embeddings** via Bedrock
  - Indexed into **FAISS** (local) or alternative vector stores

- **RAG Question Answering (QA)**
  - Query → embedded with Titan  
  - FAISS retrieves relevant chunks  
  - Retrieved context + query sent to **LLaMA3 (Bedrock)**  
  - LLaMA3 generates context-aware answers

- **Generative AI**
  - **Text QA** → Powered by LLaMA3  
  - **Image Generation** → Prompts passed to **Stable Diffusion XL (Bedrock)**  
  - Outputs saved locally under `output_images/`

- **Deployment & UI**
  - Fully containerized with Docker  
  - **Streamlit app** for user interface  
  - CLI mode for advanced operations  

---

## 🏗️ Architecture (Based on Diagram)

1. **User Interaction**
   - Access via Streamlit App (`app.py`) or CLI
   - Choose: RAG QA, Chat (LLaMA3), or Image Generation (Stable Diffusion)

2. **Document Upload**
   - PDF/Text uploaded → stored in Docs Storage

3. **Processing Layer**
   - Documents loaded & chunked
   - Sent to Bedrock **Titan Embedding Model**
   - Stored in FAISS index

4. **Query Flow**
   - User asks question → converted into embeddings
   - FAISS retrieves relevant chunks
   - Retrieved context + query sent to **LLaMA3**
   - LLaMA3 returns **Final Answer**

5. **Image Flow**
   - User enters prompt → sent to **Stable Diffusion (Bedrock)**
   - Model returns base64-encoded image
   - Saved into `output_images/` and displayed in UI

6. **Docker Container**
   - Encapsulates entire pipeline (Streamlit, FAISS, Bedrock API calls)

---

## ⚙️ Tech Stack

- **Frontend/UI**: Streamlit  
- **Vector Store**: FAISS  
- **AWS Bedrock Models**:  
  - Titan Embeddings → for semantic vector generation  
  - LLaMA3 → for text Q&A  
  - Stable Diffusion XL → for image generation  
- **Orchestration**: Docker, Python (boto3, LangChain)

---

## 🐳 Run Locally

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd <your-repo>
```

### 2. Configure AWS
Make sure your AWS credentials are set:
```bash
aws configure
```
You need:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`

### 3. Build Docker Image
```bash
docker build -t rag-bedrock-app .
```

### 4. Run Container
```bash
docker run -p 8501:8501 rag-bedrock-app
```

### 5. Access App
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📂 Project Structure
```
.
├── utils/                  # Document loaders, chunkers, embeddings
├── vector_store/           # FAISS indexes
├── output_images/          # Generated images
├── stablediffusion.py      # Bedrock Stable Diffusion script
├── app.py                  # Streamlit entry point
├── Dockerfile              # Docker container config
└── README.md               # Project documentation
```

---

## 📸 Example Use Cases
- Upload product manuals → Ask support questions (RAG QA).  
- Upload research papers → Summarize and query insights.  
- Generate creative visuals using **Stable Diffusion** with natural prompts.  

---

## ✅ Next Steps
- Add Bedrock Claude integration for summarization  
- Extend vector store options beyond FAISS  
- Enable multi-user access with authentication  
