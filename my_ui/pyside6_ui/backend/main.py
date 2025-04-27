import os
import faiss
import numpy as np
import pickle
from typing import List, Dict, Any, Optional, Tuple
from backend.gemini_client import GeminiClient

# Import logger
from utils.logger import setup_logger

# Set up logger
logger = setup_logger('backend.main')

class FAISSVectorStore:
    def __init__(self, dimension: int = 768, index_file: str = "faiss_index.idx", 
                 metadata_file: str = "metadata.pkl"):
        """
        Initialize the FAISS vector store.
        
        Args:
            dimension: Dimension of the embedding vectors
            index_file: Path to save/load the FAISS index
            metadata_file: Path to save/load the metadata
        """
        self.dimension = dimension
        self.index_file = index_file
        self.metadata_file = metadata_file
        self.metadata = []
        
        logger.info(f"Initializing FAISSVectorStore with dimension={dimension}, index_file={index_file}, metadata_file={metadata_file}")
        
        # Create or load the index
        if os.path.exists(index_file):
            logger.info(f"Loading existing index from {index_file}")
            self.index = faiss.read_index(index_file)
            if os.path.exists(metadata_file):
                logger.info(f"Loading existing metadata from {metadata_file}")
                with open(metadata_file, 'rb') as f:
                    self.metadata = pickle.load(f)
                logger.info(f"Loaded {len(self.metadata)} metadata entries")
        else:
            logger.info(f"Creating new FAISS index with dimension {dimension}")
            self.index = faiss.IndexFlatL2(dimension)
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[int]:
        """
        Add texts and their embeddings to the vector store.
        
        Args:
            texts: List of text documents to add
            metadatas: Optional list of metadata dictionaries for each text
            
        Returns:
            List of IDs for the added texts
        """
        logger.info(f"Adding {len(texts)} texts to vector store")
        
        # Initialize Gemini client for embeddings
        client = GeminiClient()
        
        # Generate embeddings for all texts
        logger.info("Generating embeddings for texts")
        embeddings = client.get_batch_embeddings(texts)
        
        # Convert embeddings to numpy array
        embeddings_np = np.array(embeddings).astype('float32')
        
        # Get current number of vectors in the index
        start_id = self.index.ntotal
        logger.info(f"Current index size: {start_id}, adding {len(texts)} new vectors")
        
        # Add embeddings to the index
        self.index.add(embeddings_np)
        
        # Add metadata
        if metadatas:
            for i, metadata in enumerate(metadatas):
                self.metadata.append({
                    "id": start_id + i,
                    "text": texts[i],
                    "metadata": metadata
                })
        else:
            for i, text in enumerate(texts):
                self.metadata.append({
                    "id": start_id + i,
                    "text": text,
                    "metadata": {}
                })
        
        # Save index and metadata
        logger.info("Saving updated index and metadata")
        self._save()
        
        # Return IDs of added documents
        return list(range(start_id, start_id + len(texts)))
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents to the query.
        
        Args:
            query: The query text
            k: Number of results to return
            
        Returns:
            List of dictionaries containing text and metadata of similar documents
        """
        logger.info(f"Performing similarity search for query: '{query}' with k={k}")
        
        # Initialize Gemini client for embeddings
        client = GeminiClient()
        
        # Generate embedding for the query
        logger.info("Generating embedding for query")
        query_embedding = client.get_embedding(query)
        
        # Convert to numpy array
        query_embedding_np = np.array([query_embedding]).astype('float32')
        
        # Search the index
        logger.info(f"Searching index with {self.index.ntotal} vectors")
        distances, indices = self.index.search(query_embedding_np, k)
        
        # Get the corresponding texts and metadata
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # -1 means no result found
                for item in self.metadata:
                    if item["id"] == idx:
                        results.append({
                            "text": item["text"],
                            "metadata": item["metadata"],
                            "distance": float(distances[0][i])
                        })
                        break
        
        logger.info(f"Found {len(results)} results")
        return results
    
    def _save(self):
        """Save the index and metadata to disk."""
        logger.info(f"Saving index to {self.index_file}")
        faiss.write_index(self.index, self.index_file)
        
        logger.info(f"Saving metadata to {self.metadata_file}")
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        logger.info(f"Saved {self.index.ntotal} vectors and {len(self.metadata)} metadata entries")
    
    def delete(self, ids: List[int]) -> None:
        """
        Delete documents from the index by their IDs.
        
        Args:
            ids: List of document IDs to delete
        """
        logger.info(f"Deleting {len(ids)} documents from index")
        
        # FAISS doesn't support direct deletion in all index types
        # For simplicity, we'll rebuild the index without the deleted documents
        
        # Get all vectors and their IDs
        all_vectors = []
        all_metadata = []
        
        for item in self.metadata:
            if item["id"] not in ids:
                all_metadata.append(item)
        
        logger.info(f"Rebuilding index with {len(all_metadata)} remaining documents")
        
        # Create a new index
        new_index = faiss.IndexFlatL2(self.dimension)
        
        # Add all vectors except the deleted ones
        if all_metadata:
            # Get texts to re-embed
            texts = [item["text"] for item in all_metadata]
            
            # Initialize Gemini client for embeddings
            client = GeminiClient()
            
            # Generate embeddings for all texts
            logger.info("Regenerating embeddings for remaining documents")
            embeddings = client.get_batch_embeddings(texts)
            
            # Convert embeddings to numpy array
            embeddings_np = np.array(embeddings).astype('float32')
            
            # Add embeddings to the new index
            new_index.add(embeddings_np)
            
            # Update IDs in metadata
            for i, item in enumerate(all_metadata):
                item["id"] = i
        
        # Replace the old index and metadata
        self.index = new_index
        self.metadata = all_metadata
        
        # Save the updated index and metadata
        logger.info("Saving updated index after deletion")
        self._save()


class RAGSystem:
    def __init__(self, vector_store: FAISSVectorStore = None):
        """
        Initialize the RAG system.
        
        Args:
            vector_store: The vector store to use for document retrieval
        """
        logger.info("Initializing RAGSystem")
        if vector_store is None:
            logger.info("Creating new FAISSVectorStore")
            self.vector_store = FAISSVectorStore()
        else:
            logger.info("Using provided FAISSVectorStore")
            self.vector_store = vector_store
        
        self.gemini_client = GeminiClient()
    
    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[int]:
        """
        Add documents to the RAG system.
        
        Args:
            documents: List of documents to add
            metadatas: Optional list of metadata for each document
                       Each metadata should contain 'filepath' and 'filename'
            
        Returns:
            List of document IDs
        """
        logger.info(f"Adding {len(documents)} documents to RAG system")
        return self.vector_store.add_texts(documents, metadatas)
    
    def query(self, query: str, k: int = 1) -> str:
        """
        Query the RAG system.
        
        Args:
            query: The query text
            k: Number of documents to retrieve
            
        Returns:
            The generated response
        """
        logger.info(f"Querying RAG system with: '{query}', k={k}")
        
        # Retrieve relevant documents
        results = self.vector_store.similarity_search(query, k)
        
        # Extract the text from the results
        contexts = [result["text"] for result in results]
        logger.info(f"Retrieved {len(contexts)} contexts for RAG")
        
        # Generate a response using the retrieved contexts
        logger.info("Generating response with Gemini")
        response = self.gemini_client.rag_response(query, contexts)
        
        return response


# Example usage
if __name__ == "__main__":
    # Initialize the RAG system
    rag = RAGSystem()
    
    # Add some example documents
    documents = [
        "FAISS (Facebook AI Similarity Search) is a library for efficient similarity search.",
        "Vector databases are optimized for storing and querying high-dimensional vectors.",
        "Embeddings are numerical representations of text that capture semantic meaning.",
        "RAG (Retrieval-Augmented Generation) combines retrieval with text generation.",
        "Google's Gemini is a multimodal AI model that can process text, images, and more."
    ]
    
    metadatas = [
        {"source": "FAISS documentation", "category": "technology"},
        {"source": "Vector DB guide", "category": "database"},
        {"source": "NLP handbook", "category": "machine learning"},
        {"source": "LLM techniques", "category": "AI"},
        {"source": "Google AI blog", "category": "AI models"}
    ]
    
    # Add documents to the RAG system
    doc_ids = rag.add_documents(documents, metadatas)
    print(f"Added {len(doc_ids)} documents to the RAG system")
    
    # Query the RAG system
    query = "How does RAG work with vector databases?"
    response = rag.query(query)
    
    print(f"Query: {query}")
    print(f"Response: {response}")