# Install: pip install chromadb openai sentence-transformers
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Your "documents" (imagine these are lecture notes)
docs = [
    "Deadlock occurs when processes wait for each other indefinitely",
    "Prevention strategies include resource ordering and banker's algorithm",
    "Detection uses wait-for graphs to find cycles",
]

# 2. Create embeddings (convert text to vectors)
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(docs)

# 3. Store in ChromaDB
client = chromadb.Client()
collection = client.create_collection("lecture_notes")
collection.add(
    documents=docs,
    embeddings=embeddings.tolist(),
    ids=["doc1", "doc2", "doc3"]
)

# 4. Query — this is what happens when grading
query = "How do we prevent deadlocks?"
query_embedding = model.encode([query])
results = collection.query(query_embeddings=query_embedding.tolist(), n_results=2)
print(results['documents'])
# Returns the 2 most relevant chunks about deadlock prevention