from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter


# 1️⃣ Load PDF documents
pdf_path = "app/data/Onboarding Guide 2025.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# 2️⃣ Split into smaller chunks
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)
texts = [doc.page_content for doc in docs]


# 3️⃣ Create local embeddings (NO API NEEDED)
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts)

# 4️⃣ Store embeddings in FAISS vectorstore
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Map indices to original texts for retrieval
id_to_text = {i: text for i, text in enumerate(texts)}


# 5️⃣ Function to ask questions (naive retrieval, no LLM)
def ask_question(question):
    question_embedding = model.encode([question])
    distances, indices = index.search(np.array(question_embedding), k=3)
    
    valid_indices = [idx for idx in indices[0] if idx != -1]
    if not valid_indices:
        return "Sorry, I couldn't find anything relevant in the document."
    
    results = [id_to_text[idx] for idx in valid_indices]
    return "\n\n".join(results)



# 🔍 Test the pipeline:
if __name__ == "__main__":
    question = "Who do I contact for IT support?"
    answer = ask_question(question)
    print("\n🤖 Docsy (Local) says:\n", answer)
