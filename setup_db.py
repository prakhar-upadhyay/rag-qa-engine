import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# 1. Secure Authentication
load_dotenv()
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

index_name = "fde-rag-index"

# 2. Purge the corrupted architecture
existing_indexes = pc.list_indexes().names()
if index_name in existing_indexes:
    print(f"Deleting the old misconfigured '{index_name}' index...")
    pc.delete_index(index_name)
    print("Waiting 10 seconds for the cloud servers to purge the data...")
    time.sleep(10)

# 3. Force-provision the new 3072-dimension index
print(f"Provisioning a new '{index_name}' index with exactly 3072 dimensions...")
pc.create_index(
    name=index_name,
    dimension=3072,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

print("\nSuccess! The 3072-dimension cloud infrastructure is fully provisioned.")
print("You can now safely run your pipeline.")