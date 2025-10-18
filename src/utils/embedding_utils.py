"""
Embedding Utilities
Helper functions for working with embeddings
"""

import numpy as np
from typing import List, Tuple, Optional
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Manages text embeddings for semantic search and similarity
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu"
    ):
        """
        Initialize embedding manager
        
        Args:
            model_name: HuggingFace model name
            device: 'cpu' or 'cuda'
        """
        self.model_name = model_name
        self.device = device
        
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for texts
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            numpy array of embeddings (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def encode_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for single text
        
        Args:
            text: Text string
            
        Returns:
            numpy array of embedding (embedding_dim,)
        """
        return self.model.encode([text], convert_to_numpy=True)[0]
    
    def similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """
        Calculate cosine similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        emb1 = self.encode_single(text1)
        emb2 = self.encode_single(text2)
        
        similarity = cosine_similarity(
            emb1.reshape(1, -1),
            emb2.reshape(1, -1)
        )[0][0]
        
        return float(similarity)
    
    def find_most_similar(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 5
    ) -> List[Tuple[int, str, float]]:
        """
        Find most similar texts to query
        
        Args:
            query: Query text
            candidates: List of candidate texts
            top_k: Number of results to return
            
        Returns:
            List of (index, text, similarity_score) tuples
        """
        if not candidates:
            return []
        
        # Encode query and candidates
        query_emb = self.encode_single(query)
        candidate_embs = self.encode(candidates)
        
        # Calculate similarities
        similarities = cosine_similarity(
            query_emb.reshape(1, -1),
            candidate_embs
        )[0]
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [
            (int(idx), candidates[idx], float(similarities[idx]))
            for idx in top_indices
        ]
        
        return results
    
    def cluster_texts(
        self,
        texts: List[str],
        n_clusters: int = 5
    ) -> Tuple[np.ndarray, List[List[int]]]:
        """
        Cluster texts using K-means
        
        Args:
            texts: List of texts
            n_clusters: Number of clusters
            
        Returns:
            Tuple of (cluster_labels, cluster_groups)
        """
        from sklearn.cluster import KMeans
        
        if len(texts) < n_clusters:
            logger.warning(f"Not enough texts ({len(texts)}) for {n_clusters} clusters")
            n_clusters = min(n_clusters, len(texts))
        
        # Generate embeddings
        embeddings = self.encode(texts)
        
        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        
        # Group by cluster
        clusters = [[] for _ in range(n_clusters)]
        for idx, label in enumerate(labels):
            clusters[label].append(idx)
        
        return labels, clusters
    
    def semantic_search(
        self,
        query: str,
        corpus: List[str],
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[Tuple[int, str, float]]:
        """
        Semantic search in corpus
        
        Args:
            query: Search query
            corpus: List of documents
            top_k: Number of results
            threshold: Minimum similarity threshold
            
        Returns:
            List of (index, text, score) tuples
        """
        results = self.find_most_similar(query, corpus, top_k)
        
        # Filter by threshold
        if threshold > 0:
            results = [
                (idx, text, score)
                for idx, text, score in results
                if score >= threshold
            ]
        
        return results


def calculate_similarity_matrix(
    texts: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> np.ndarray:
    """
    Calculate pairwise similarity matrix for texts
    
    Args:
        texts: List of texts
        model_name: Embedding model name
        
    Returns:
        Similarity matrix (n_texts, n_texts)
    """
    embedder = EmbeddingManager(model_name=model_name)
    embeddings = embedder.encode(texts)
    
    similarity_matrix = cosine_similarity(embeddings)
    
    return similarity_matrix


def deduplicate_texts(
    texts: List[str],
    threshold: float = 0.95,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> List[int]:
    """
    Find duplicate texts based on semantic similarity
    
    Args:
        texts: List of texts
        threshold: Similarity threshold for duplicates
        model_name: Embedding model name
        
    Returns:
        Indices of unique texts
    """
    if len(texts) <= 1:
        return list(range(len(texts)))
    
    # Calculate similarity matrix
    sim_matrix = calculate_similarity_matrix(texts, model_name)
    
    # Find duplicates
    unique_indices = []
    seen = set()
    
    for i in range(len(texts)):
        if i in seen:
            continue
        
        unique_indices.append(i)
        
        # Mark similar texts as seen
        for j in range(i + 1, len(texts)):
            if sim_matrix[i, j] >= threshold:
                seen.add(j)
    
    return unique_indices


def find_diverse_subset(
    texts: List[str],
    n: int,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> List[int]:
    """
    Find diverse subset of texts (maximize diversity)
    
    Args:
        texts: List of texts
        n: Number of texts to select
        model_name: Embedding model name
        
    Returns:
        Indices of diverse texts
    """
    if n >= len(texts):
        return list(range(len(texts)))
    
    embedder = EmbeddingManager(model_name=model_name)
    embeddings = embedder.encode(texts)
    
    # Greedy selection for diversity
    selected = [0]  # Start with first text
    
    for _ in range(n - 1):
        max_min_dist = -1
        best_idx = -1
        
        # Find text with maximum minimum distance to selected
        for i in range(len(texts)):
            if i in selected:
                continue
            
            # Calculate minimum distance to selected texts
            distances = [
                1 - cosine_similarity(
                    embeddings[i].reshape(1, -1),
                    embeddings[j].reshape(1, -1)
                )[0][0]
                for j in selected
            ]
            
            min_dist = min(distances)
            
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_idx = i
        
        selected.append(best_idx)
    
    return selected


# ═══════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Initialize
    embedder = EmbeddingManager()
    
    # Example texts
    texts = [
        "The customer has a good credit history.",
        "Customer has excellent payment record.",
        "This applicant has high income.",
        "The loan amount is very large.",
        "Customer wants to buy a vehicle."
    ]
    
    # Find similar texts
    query = "Customer has strong credit score"
    results = embedder.find_most_similar(query, texts, top_k=3)
    
    print("Query:", query)
    print("\nMost similar texts:")
    for idx, text, score in results:
        print(f"  [{idx}] {text} (similarity: {score:.3f})")
    
    # Calculate similarity
    sim = embedder.similarity(texts[0], texts[1])
    print(f"\nSimilarity between texts 0 and 1: {sim:.3f}")
    
    # Find duplicates
    all_texts = texts + [texts[0]]  # Add duplicate
    unique_indices = deduplicate_texts(all_texts, threshold=0.90)
    print(f"\nUnique text indices: {unique_indices}")