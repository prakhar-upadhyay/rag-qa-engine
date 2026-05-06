import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone

# 1. Secure Authentication
load_dotenv()
if not os.environ.get("GEMINI_API_KEY") or not os.environ.get("PINECONE_API_KEY"):
    raise ValueError("Missing API keys in .env file!")

print("1. Environment secured. Connecting to cloud infrastructure...")

# Initialize Cloud Connection
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "fde-rag-index"

# 2. Connect to Existing Database (NO UPLOADING!)
print("2. Syncing with existing Pinecone vectors (Skipping embedding phase)...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# The Hotwire Method to prevent the TypeError
active_index = pc.Index(index_name)
vector_store = PineconeVectorStore(
    index=active_index, 
    embedding=embeddings,
    text_key="text" 
)
retriever = vector_store.as_retriever()

# 3. The Brain (The Generation Model)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say that you don't know. "
    "Use three sentences maximum and keep the answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 4. Assemble the Pipeline
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

print("3. Cloud RAG Pipeline Assembled! \n")
print("=" * 50)
print("Welcome to the Cloud-Native Q&A Engine.")
print("Type 'exit' or 'quit' to close.")
print("=" * 50)

# 5. Interactive Loop
while True:
    user_question = input("\nAsk a question: ")
    if user_question.lower() in ['exit', 'quit']:
        break
    if not user_question.strip():
        continue
        
    print("Querying Pinecone Cloud & Generating Answer...")
    response = rag_chain.invoke({"input": user_question})
    
    print("\n[AI Answer]:")
    print(response["answer"])