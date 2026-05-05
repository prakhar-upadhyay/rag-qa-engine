import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load the secret API key from the .env file
load_dotenv()
if not os.environ.get("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY missing from .env file!")

print("Environment secured. Loading data...")

# 2. Load the private document
loader = TextLoader("handbook.txt")
document = loader.load()

# 3. Chunk the document
# We split the text into chunks of 200 characters, with a 20-character overlap 
# so we don't accidentally cut a sentence or concept in half.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, 
    chunk_overlap=20
)
chunks = text_splitter.split_documents(document)

# 4. Verification
print(f"Success! The document was split into {len(chunks)} chunks.")
print("-" * 40)
print("Preview of Chunk #1:")
print(chunks[0].page_content)