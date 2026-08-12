from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentRAG:

    def __init__(self):

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.vector_db = Chroma(
            collection_name="documind_documents",
            embedding_function=self.embeddings,
            persist_directory="data/chroma_db",
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
        )

    def add_document(self, text, source="document"):

        chunks = self.text_splitter.split_text(
            text
        )

        if not chunks:
            return 0

        self.vector_db.add_texts(
            texts=chunks,
            metadatas=[
                {"source": source}
                for _ in chunks
            ],
        )

        return len(chunks)

    def search(self, query, k=4):

        results = self.vector_db.similarity_search(
            query,
            k=k,
        )

        return results