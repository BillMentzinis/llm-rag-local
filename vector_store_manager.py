"""
Vector Store Manager for RAG Pipeline.
Manages ChromaDB vector database and embedding generation.
"""

import os
import hashlib
from typing import List, Dict, Optional
from datetime import datetime
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from config import RAG_CONFIG, PATHS


class VectorStoreManager:
    """
    Manages vector storage, embedding generation, and retrieval using ChromaDB.
    """

    def __init__(self, persist_directory: str = None, embedding_model_name: str = None,
                 collection_name: str = "documents"):
        """
        Initialize VectorStoreManager.

        Args:
            persist_directory: Directory for ChromaDB persistence (default from config)
            embedding_model_name: Name of sentence-transformers model (default from config)
            collection_name: Name of the ChromaDB collection
        """
        self.persist_directory = persist_directory or PATHS["chroma_db"]
        self.embedding_model_name = embedding_model_name or RAG_CONFIG["embedding_model"]
        self.collection_name = collection_name

        # Create persist directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)

        # Initialize embedding model
        print(f"Loading embedding model: {self.embedding_model_name}...")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        print(f"Embedding model loaded. Dimension: {self.embedding_model.get_sentence_embedding_dimension()}")

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Document chunks for RAG"}
        )

        print(f"ChromaDB collection '{self.collection_name}' ready. Current count: {self.collection.count()}")

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def _generate_chunk_id(self, filename: str, chunk_index: int) -> str:
        """
        Generate unique ID for a chunk.

        Args:
            filename: Source filename
            chunk_index: Index of chunk

        Returns:
            Unique chunk ID
        """
        # Create hash of filename for uniqueness
        file_hash = hashlib.md5(filename.encode()).hexdigest()[:8]
        return f"{file_hash}_{chunk_index}"

    def add_documents(self, chunks: List[Dict], batch_size: int = 100) -> int:
        """
        Add document chunks to vector store.

        Args:
            chunks: List of chunk dictionaries with 'text' and metadata
            batch_size: Number of chunks to process at once

        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0

        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            chunk_id = self._generate_chunk_id(
                chunk.get("filename", "unknown"),
                chunk.get("chunk_index", 0)
            )

            ids.append(chunk_id)
            documents.append(chunk["text"])

            # Prepare metadata (ChromaDB doesn't accept nested dicts)
            metadata = {
                "filename": chunk.get("filename", "unknown"),
                "file_type": chunk.get("file_type", "unknown"),
                "chunk_index": chunk.get("chunk_index", 0),
                "total_chunks": chunk.get("total_chunks", 1),
                "upload_timestamp": chunk.get("upload_timestamp", datetime.now().isoformat()),
                "char_count": chunk.get("char_count", len(chunk["text"])),
            }
            metadatas.append(metadata)

        # Generate embeddings
        print(f"Generating embeddings for {len(documents)} chunks...")
        embeddings = self.generate_embeddings(documents)

        # Add to collection in batches
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_docs = documents[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]

            self.collection.add(
                ids=batch_ids,
                documents=batch_docs,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas
            )

        print(f"Added {len(ids)} chunks to vector store.")
        return len(ids)

    def search(self, query: str, top_k: int = None, min_similarity: float = None,
               filter_metadata: Dict = None) -> List[Dict]:
        """
        Search for relevant chunks using semantic similarity.

        Args:
            query: Search query
            top_k: Number of results to return (default from config)
            min_similarity: Minimum similarity threshold (default from config)
            filter_metadata: Optional metadata filter (e.g., {"filename": "doc.pdf"})

        Returns:
            List of result dictionaries with text, metadata, and similarity score
        """
        if self.collection.count() == 0:
            return []

        top_k = top_k or RAG_CONFIG["top_k"]
        min_similarity = min_similarity if min_similarity is not None else RAG_CONFIG["min_similarity"]

        # Generate query embedding
        query_embedding = self.generate_embeddings([query])[0]

        # Search collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata,  # Optional metadata filtering
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        formatted_results = []
        if results["documents"] and results["documents"][0]:
            for i, (doc, metadata, distance) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )):
                # Convert distance to similarity (ChromaDB uses L2 distance)
                # Similarity is inversely related to distance
                # We'll normalize it to 0-1 range where 1 is most similar
                similarity = 1 / (1 + distance)

                # Filter by minimum similarity
                if similarity < min_similarity:
                    continue

                formatted_results.append({
                    "text": doc,
                    "metadata": metadata,
                    "similarity": similarity,
                    "rank": i + 1
                })

        return formatted_results

    def delete_document(self, filename: str) -> int:
        """
        Delete all chunks from a specific document.

        Args:
            filename: Name of file to delete

        Returns:
            Number of chunks deleted
        """
        # Get all IDs for this filename
        results = self.collection.get(
            where={"filename": filename},
            include=[]
        )

        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            count = len(results["ids"])
            print(f"Deleted {count} chunks from '{filename}'")
            return count

        return 0

    def list_documents(self) -> List[Dict]:
        """
        List all unique documents in the vector store.

        Returns:
            List of document info dictionaries
        """
        # Get all metadata
        all_data = self.collection.get(include=["metadatas"])

        if not all_data["metadatas"]:
            return []

        # Group by filename
        documents = {}
        for metadata in all_data["metadatas"]:
            filename = metadata.get("filename", "unknown")

            if filename not in documents:
                documents[filename] = {
                    "filename": filename,
                    "file_type": metadata.get("file_type", "unknown"),
                    "upload_timestamp": metadata.get("upload_timestamp", ""),
                    "chunk_count": 0
                }

            documents[filename]["chunk_count"] += 1

        return list(documents.values())

    def get_document_info(self, filename: str) -> Optional[Dict]:
        """
        Get information about a specific document.

        Args:
            filename: Name of the document

        Returns:
            Document info dictionary or None if not found
        """
        results = self.collection.get(
            where={"filename": filename},
            include=["metadatas"]
        )

        if not results["metadatas"]:
            return None

        metadata = results["metadatas"][0]
        return {
            "filename": filename,
            "file_type": metadata.get("file_type", "unknown"),
            "upload_timestamp": metadata.get("upload_timestamp", ""),
            "chunk_count": len(results["metadatas"])
        }

    def clear_all(self) -> int:
        """
        Delete all documents from the vector store.

        Returns:
            Number of chunks deleted
        """
        count = self.collection.count()
        if count > 0:
            # Delete the collection and recreate it
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Document chunks for RAG"}
            )
            print(f"Cleared all {count} chunks from vector store")

        return count

    def get_stats(self) -> Dict:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with statistics
        """
        documents = self.list_documents()

        return {
            "total_chunks": self.collection.count(),
            "total_documents": len(documents),
            "embedding_model": self.embedding_model_name,
            "embedding_dimension": self.embedding_model.get_sentence_embedding_dimension(),
            "persist_directory": self.persist_directory,
            "documents": documents
        }

    def is_empty(self) -> bool:
        """
        Check if vector store is empty.

        Returns:
            True if no documents are stored
        """
        return self.collection.count() == 0


if __name__ == "__main__":
    # Simple test
    manager = VectorStoreManager()
    print("\nVectorStoreManager initialized successfully!")
    print("\nStats:", manager.get_stats())
