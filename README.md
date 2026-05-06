# Enterprise Cloud-Native RAG Engine

A fully decoupled Retrieval-Augmented Generation (RAG) pipeline built for high-performance, private document querying. This project demonstrates a production-ready approach to AI data retrieval, featuring isolated ETL processes, cloud-based vector storage, and rate-limit-optimized architecture.

## 🚀 Architecture Highlights
Unlike standard monolithic RAG tutorials, this pipeline is engineered for deployment stability:
* **Decoupled Architecture:** Separates the heavy-lifting document ingestion (ETL) phase from the lightweight, real-time chat retrieval phase to strictly protect API rate limits and optimize latency.
* **Infrastructure-as-Code (IaC):** Utilizes direct API provisioning scripts to forcefully allocate 3072-dimensional vector indexing in the cloud, bypassing graphical interface limitations.
* **Transient Error Handling:** Designed to gracefully manage `429 RESOURCE_EXHAUSTED` and `503 UNAVAILABLE` cloud traffic spikes common in scalable AI applications.
* **Zero-Trust Security:** API credentials and environment variables are strictly isolated from source control.

## 🛠️ Tech Stack
* **Framework:** LangChain (Community & Core)
* **LLM / Generation:** Google Gemini 2.5 Flash API
* **Embeddings:** Google `gemini-embedding-001` (3072 Dimensions)
* **Vector Database:** Pinecone (Cloud-Native Serverless)
* **Environment:** Python 3.14+

## ⚙️ Core Workflow
1. **Ingestion & Chunking:** Reads complex unstructured text (tested on esoteric astrological data and corporate handbooks), applying `RecursiveCharacterTextSplitter` to optimize LLM context windows.
2. **Mathematical Vectorization:** Translates text chunks into 3072-dimension high-fidelity floating-point arrays.
3. **Cloud Sync:** Pushes embeddings directly to a managed Pinecone serverless index.
4. **Retrieval & Grounding:** The chat engine securely queries the vector DB via cosine similarity and restricts the LLM to answer *strictly* using the retrieved context to prevent hallucination.

## 📈 Future Roadmap
* Implementing exponential backoff logic (Tenacity) for automated retry handling during server-side API throttling.
* Containerization (Docker) of the retrieval engine for rapid client-side deployment.
