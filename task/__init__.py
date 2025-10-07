"""Helper package exposing the Groq-based structured RAG pipeline."""

from .structured_rag import DEFAULT_DATASET, GroqChatClient, StructuredRAGPipeline

__all__ = ["DEFAULT_DATASET", "GroqChatClient", "StructuredRAGPipeline"]
