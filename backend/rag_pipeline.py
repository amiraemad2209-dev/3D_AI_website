from dotenv import load_dotenv
import os
load_dotenv()
# Set LangSmith environment variables GLOBAL first

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "RAG_Pipeline"

os.environ["LANGCHAIN_API_KEY"] = os.getenv("GROQ_API_KEY")

import fitz  # PyMuPDF
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
#from google.colab import userdata
import re
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
# Ragas imports
from datasets import Dataset
from ragas import evaluate
from ragas.metrics.collections import faithfulness, answer_relevancy, context_recall, context_precision
from langsmith import traceable                  

# ===========================================================================
# 1. LOADER
# ===========================================================================
def load_and_chunk(file_path: str) -> list[Document]:
    print("[INFO] Extracting text from PDF...")
    pdf = fitz.open(file_path)
    full_text = ""
    for page in pdf:
        full_text += page.get_text() + "\n"
    pdf.close()

    raw_doc = Document(page_content=full_text, metadata={"source": file_path})

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n", "،", ".", "؟", "!", " ", ""],
    )

    print("[INFO] Splitting text into chunks...")
    chunks: list[Document] = splitter.split_documents([raw_doc])

    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx
        chunk.metadata["source"] = file_path

    return chunks           


# ===========================================================================
# 2. EMBEDDER
# ===========================================================================
_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
_model: SentenceTransformer | None = None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print("[INFO] Loading SentenceTransformer model...")
        _model = SentenceTransformer(_MODEL_NAME)
    return _model
model=_get_model()

def embed_texts(texts: list[str]) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=100,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.array(embeddings)

def embed_query(query: str) -> np.ndarray:
    return embed_texts([query])[0] # Handle GROQ_API_KEY using userdata.get with a fallback to manual input

"""
groq_api_key = userdata.get('GROQ_API_KEY')

if groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key
    print("[INFO] GROQ_API_KEY loaded from Colab secrets.")
elif not os.environ.get("GROQ_API_KEY"): # Fallback if not set by userdata or elsewhere
    print("[WARNING] GROQ_API_KEY not found in Colab secrets or environment.")
    print("Please set it in Colab secrets (under the key 'GROQ_API_KEY') or provide it now:")
    key = input("Enter GROQ_API_KEY: ").strip()
    if key:
        os.environ["GROQ_API_KEY"] = key
        print("API Key set from manual input.\n")
    else:
        print("[ERROR] GROQ_API_KEY is still not set. Some functionalities may fail.")   
        
"""
# ===========================================================================
# 3. VECTOR STORE (ChromaDB)
# ===========================================================================
from langchain_core.embeddings import Embeddings
from typing import List, Tuple
import chromadb
from chromadb import PersistentClient
from chromadb.api.models.Collection import Collection


# ===========================================================================
# 3. VECTOR STORE (ChromaDB)
# ===========================================================================
_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None
_COLLECTION_NAME = "rag_demo"

def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.EphemeralClient()
    return _client

def build_index(chunks: list[Document]) -> None:
    global _collection
    client = _get_client()

    try:
        client.delete_collection(_COLLECTION_NAME)
    except Exception:
        pass

    _collection = client.create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [chunk.page_content for chunk in chunks]
    print("[INFO] Computing embeddings for vector store...")
    embeddings: np.ndarray = embed_texts(texts)

    _collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        ids=[
    f'{chunk.metadata["source"]}_{chunk.metadata["chunk_id"]}'
    for chunk in chunks
         ],
        metadatas=[chunk.metadata for chunk in chunks],
    )
    print(f"[INFO] Vector index built with {len(chunks)} chunks.")

def vector_search(query_embedding: np.ndarray, top_k: int = 20) -> list[dict]:
    if _collection is None:
        raise RuntimeError("Call build_index() before vector_search().")

    results = _collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(top_k, _collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    hits: list[dict] = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "text": doc,
            "meta": meta,
            "score": 1.0 - dist,
        })
    return hits



# ===========================================================================
# 4. BM25 STORE
# ===========================================================================
_bm25_index: BM25Okapi | None = None
_chunks: list[Document] = []
_TOKEN_PATTERN = re.compile(r"[\w\u0600-\u06FF]+")

def tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())

def build_bm25_index(chunks: list[Document]) -> None:
    global _bm25_index, _chunks
    _chunks = chunks
    tokenized_corpus = [tokenize(chunk.page_content) for chunk in chunks]
    _bm25_index = BM25Okapi(tokenized_corpus)
    print(f"[INFO] BM25 index built with {len(chunks)} chunks.")

def bm25_search(query: str, top_k: int = 20) -> list[dict]:
    if _bm25_index is None:
        raise RuntimeError("Call build_bm25_index() before bm25_search().")

    query_tokens = tokenize(query)
    scores: np.ndarray = _bm25_index.get_scores(query_tokens)

    ranked = sorted(
        [(score, idx) for idx, score in enumerate(scores) if score > 0],
        reverse=True,
    )

    hits: list[dict] = []
    for score, idx in ranked[:top_k]:
        chunk = _chunks[idx]
        hits.append({
            "text": chunk.page_content,
            "meta": chunk.metadata,
            "score": float(score),
        })
    return hits


# ===========================================================================
# 5. QUERY TRANSFORM
# ===========================================================================
_transform_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7,
)

_transform_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are a query expansion assistant. "
        "Your task is to generate exactly 2 alternative versions of the user's search query. "
        "Rules:\n"
        "1. Match the language of the original query exactly "
        "(if the query is in Arabic, respond in Arabic; if English, respond in English).\n"
        "2. Return ONLY the 2 alternatives, one per line.\n"
        "3. Do NOT number the lines.\n"
        "4. Do NOT add any explanation, prefix, or commentary.\n"
        "5. Each alternative must rephrase the intent without adding new facts."
    ),
    HumanMessagePromptTemplate.from_template("{query}"),
])

_transform_chain = _transform_prompt | _transform_llm | StrOutputParser()

def expand_queries(query: str) -> list[str]:
    try:
        response = _transform_chain.invoke({"query": query})
        raw: str = response.strip()
        variations = [line.strip() for line in raw.splitlines() if line.strip()]
        variations = variations[:3]
        return [query] + variations
    except Exception as e:
        print(f"[WARNING] Query expansion failed: {e}")
        return [query]


# ===========================================================================
# 6. RETRIEVER
# ===========================================================================
RRF_K = 60

def reciprocal_rank_fusion(dense_results: list[dict], sparse_results: list[dict]) -> list[dict]:
    scores: dict[str, float] = {}
    meta_map: dict[str, dict] = {}
    text_map: dict[str, str] = {}

    for result_list in (dense_results, sparse_results):
        for rank, hit in enumerate(result_list):
            key = hit["text"][:100]
            rrf_contribution = 1.0 / (RRF_K + rank + 1)
            scores[key] = scores.get(key, 0.0) + rrf_contribution
            if key not in meta_map:
                meta_map[key] = hit["meta"]
                text_map[key] = hit["text"]

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [
        {
            "text": text_map[key],
            "meta": meta_map[key],
            "rrf_score": score,
        }
        for key, score in fused
    ]

def hybrid_retrieve(query: str, top_k: int = 20) -> list[dict]:
    query_emb = embed_query(query)
    dense  = vector_search(query_emb, top_k=5)
    sparse = bm25_search(query, top_k=top_k)
    return reciprocal_rank_fusion(dense, sparse)


# ===========================================================================
# 7. GENERATOR
# ===========================================================================
_gen_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

_gen_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "أنت مدرس NLP خبير.\n\n"
        "قواعد:\n"
        "- اعتمد فقط على السياق\n"
        "- لا تخترع معلومات\n"
        "- لو الإجابة غير موجودة قل: \"غير مذكور في النص\"\n"
        "- اشرح ببساطة\n"
        "- نظّم الإجابة (نقاط أو خطوات)"
    ),
    HumanMessagePromptTemplate.from_template(
        "السياق:\n{context}\n\nالسؤال:\n{question}\n\nالإجابة:"
    ),
])

_gen_chain = _gen_prompt | _gen_llm | StrOutputParser()

def generate_answer(query: str, context_chunks: list[dict]) -> dict:
    chunks = context_chunks[:5]
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(f"[Chunk {i}]\n{chunk['text']}")
    context_text = "\n---\n".join(context_parts)

    response = _gen_chain.invoke({"context": context_text, "question": query})
    answer: str = response.strip()
    sources: list[dict] = [
        {**chunk["meta"], "text": chunk["text"]}
        for chunk in chunks
    ]

    return {"answer": answer, "sources": sources}   

# ===================================================================
# ASK RAG FUNCTION
# ===================================================================

def ask_rag(question: str):

    print(f"[QUESTION] {question}")

    expanded = expand_queries(question)

    seen_keys = set()
    all_candidates = []

    for q in expanded:

        hits = hybrid_retrieve(q, top_k=3)

        for hit in hits:

            key = hit["text"][:100]

            if key not in seen_keys:
                seen_keys.add(key)
                all_candidates.append(hit)

    top_hits = all_candidates[:3]

    result = generate_answer(question, top_hits)

    return result["answer"]


# ===========================================================================
# 8. PIPELINE ENTRY POINT
# ===========================================================================
@traceable(project_name="RAG_Pipeline")
def main():
    # Confirming tracing configuration
    project = os.environ.get("LANGCHAIN_PROJECT", "Not Set")
    print("======================================================")
    print(f"             Google Colab RAG Pipeline (Project: {project})")
    print("======================================================")

    pdf_folder = "data"

    if not os.path.exists(pdf_folder):
       print(f"[ERROR] Folder not found: {pdf_folder}")
       return

    all_chunks = []

    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
           file_path = os.path.join(pdf_folder, file)
           print(f"[INFO] Processing {file_path}")
           all_chunks.extend(load_and_chunk(file_path))

    build_index(all_chunks)
    build_bm25_index(all_chunks)
    print("✅ Indexing Complete.")

    # Interactive Query Phase
    while True:
        print("\n" + "="*54)
        question = input("❓ Ask a question (or type 'quit' to exit): ").strip()
        if question.lower() in ["quit", "q", "exit"]:
            print("Exiting pipeline. Goodbye!")
            break

        if not question:
            continue

        print("\n--- Phase 2: Retrieval & Generation ---")

        print("[INFO] Expanding query...")
        expanded: list[str] = expand_queries(question)
        print("Expanded queries:")
        for i, q in enumerate(expanded, 1):
            print(f"  {i}. {q}")

        print("[INFO] Running Hybrid Retrieval (Dense + BM25)...")
        seen_keys: set[str] = set()
        all_candidates: list[dict] = []

        for q in expanded:
            hits = hybrid_retrieve(q, top_k=3)
            for hit in hits:
                key = hit["text"][:100]
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_candidates.append(hit)

        print(f"[INFO] Retrieved {len(all_candidates)} unique chunks across variations.")

        # Answer Generation

         
        
        print("[INFO] Generating final answer using ChatGroq...")
        top_3_hits = all_candidates[:3]
        result = generate_answer(question, top_3_hits)

        print("\n======================================================")
        print("💡 Answer:")
        print("======================================================")
        print(result["answer"])
        print("\n📚 Sources Used (Top Chunks):")
        print("-" * 54)
        retrieved_contexts_for_ragas = []
        for i, src in enumerate(result["sources"], start=1):
            chunk_id = src.get('chunk_id', 'N/A')
            chunk_text = src.get('text', '').strip()
            print(f"\n  [{i}] Chunk ID: {chunk_id}")
            print(f"      {chunk_text[:300]}{'...' if len(chunk_text) > 300 else ''}")
            retrieved_contexts_for_ragas.append(chunk_text)
        print("\n" + "=" * 54 + "\n")

        # RAGAS Evaluation Prompt
        eval_choice = input("✨ Do you want to evaluate this interaction with RAGAS? (yes/no): ").strip().lower()
        if eval_choice == 'yes':
            ground_truth = input("✍️ Please provide the ground truth answer for this question: ").strip()

            # Prepare data for RAGAS
            data = {
                "question": [question],
                "answer": [result["answer"]],
                "contexts": [retrieved_contexts_for_ragas],
                "ground_truths": [[ground_truth]]
            }
            dataset = Dataset.from_dict(data)

            # Define Ragas metrics.
            metrics = [
                faithfulness,
                answer_relevancy,
                context_recall,
                context_precision,
            ]

            print("\n[INFO] Running RAGAS evaluation for this interaction...")
            try:
                ragas_result = evaluate(dataset, metrics)
                print("\n======================================================")
                print("         RAGAS Evaluation Results (Current Interaction)q")
                print("======================================================")
                print(ragas_result)
                print("\n" + "=" * 54 + "\n")
            except Exception as e:
                print(f"[ERROR] RAGAS evaluation failed: {e}")



pdf_folder = "data"
all_chunks = []

for file in os.listdir(pdf_folder):
    if file.endswith(".pdf"):
        file_path = os.path.join(pdf_folder, file)
        print(f"[INFO] Processing {file_path}")
        all_chunks.extend(load_and_chunk(file_path))

build_index(all_chunks)
build_bm25_index(all_chunks)

print("[INFO] RAG Ready 🚀")

if __name__ == "__main__":
    #main()       # for terminal loop
    pass          # for using flask            