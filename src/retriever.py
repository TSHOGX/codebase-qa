import os
import logging
from typing import List, Dict, Any, Optional, Union
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
from src.indexer import CodeIndexer, MockEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeRetriever:
    """Retriever for code blocks based on natural language queries."""

    def __init__(
        self,
        index_path: str = "code_index",
        llm: Optional[Any] = None,
        embedding_model: Optional[Any] = None,
        model_name: str = "qwen2.5",
        embedding_model_name: str = "nomic-embed-text",
    ):
        """Initialize the code retriever.

        Args:
            index_path: Path to the index
            llm: Language model to use for query enhancement
            embedding_model: Model to use for embeddings
            model_name: Name of Ollama model for LLM
            embedding_model_name: Name of Ollama model for embeddings
        """
        # Make index_path absolute if it's not already
        if not os.path.isabs(index_path):
            self.index_path = os.path.abspath(index_path)
        else:
            self.index_path = index_path

        logger.info(f"Using index path: {self.index_path}")

        # Default to Ollama LLM if none provided
        if llm is None:
            self.llm = Ollama(model=model_name)
            logger.info(f"Using Ollama LLM with model: {model_name}")
        else:
            self.llm = llm

        # Default to Ollama embeddings if none provided
        if embedding_model is None:
            self.embedding_model = OllamaEmbeddings(model=embedding_model_name)
            logger.info(f"Using Ollama embeddings with model: {embedding_model_name}")
        else:
            self.embedding_model = embedding_model

        # Load the indexer
        self.indexer = CodeIndexer(
            embedding_model=self.embedding_model, index_path=self.index_path
        )

        # Check if index exists
        if (
            not os.path.exists(f"{self.index_path}.faiss")
            and not os.path.exists(f"{self.index_path}/index.faiss")
            and not (
                os.path.isdir(self.index_path)
                and os.path.exists(f"{self.index_path}/index.faiss")
            )
        ):
            logger.warning(
                f"No index found at {self.index_path}. Please ingest code files first."
            )

    def _enhance_query(self, query: str) -> str:
        """Use LLM to enhance the user query for better retrieval.

        Args:
            query: Original user query

        Returns:
            Enhanced query for better semantic search
        """
        # When using mock embeddings, skip query enhancement
        if isinstance(self.embedding_model, MockEmbeddings):
            logger.info("Using mock embeddings, skipping query enhancement")
            return query

        prompt_template = """You are an expert programmer. Your job is to help improve code search queries.
Given the original query: "{query}"
Rewrite this as a more specific and detailed query for finding relevant code snippets.
Focus on technical details, code patterns, or functionality that might be present in the code.
Do not include any explanations or rationale, just output the improved query.

Enhanced Query:"""

        prompt = PromptTemplate(template=prompt_template, input_variables=["query"])

        try:
            response = self.llm.invoke(prompt.format(query=query))
            enhanced_query = response.content.strip()

            logger.info(f"Original query: '{query}'")
            logger.info(f"Enhanced query: '{enhanced_query}'")

            return enhanced_query
        except Exception as e:
            logger.error(f"Error enhancing query: {str(e)}")
            return query

    def hybrid_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Perform hybrid search using both keyword and semantic search.

        Args:
            query: User query
            top_k: Number of results to return

        Returns:
            List of matching code blocks with metadata
        """
        if (
            not hasattr(self.indexer, "vector_store")
            or self.indexer.vector_store is None
        ):
            logger.error("No vector store available. Please ingest code files first.")
            return []

        try:
            # Enhanced semantic search
            enhanced_query = self._enhance_query(query)
            semantic_results = self.indexer.vector_store.similarity_search_with_score(
                enhanced_query, k=top_k * 2  # Get more results for re-ranking
            )

            # Convert to list of dicts with metadata
            results = []
            for doc, score in semantic_results:
                doc_id = doc.metadata["id"]
                block_data = self.indexer.get_block_by_id(doc_id)

                if block_data:
                    results.append(
                        {
                            "file_path": doc.metadata["file_path"],
                            "start_line": doc.metadata["start_line"],
                            "end_line": doc.metadata["end_line"],
                            "language": doc.metadata["language"],
                            "block_type": doc.metadata["block_type"],
                            "code_block": block_data["content"],
                            "comments": block_data.get("comments", ""),
                            "score": float(score),
                        }
                    )

            # Sort by score (ascending because FAISS returns distances)
            results.sort(key=lambda x: x["score"])

            # Take top K results
            return results[:top_k]

        except Exception as e:
            logger.error(f"Error during hybrid search: {str(e)}")
            return []

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Primary retrieval method combining different search approaches.

        Args:
            query: User query
            top_k: Number of results to return

        Returns:
            List of matching code blocks with metadata
        """
        return self.hybrid_search(query, top_k)

    def qa_retrieval(self, query: str) -> str:
        """Perform QA retrieval with LLM-based answer.

        Args:
            query: User query

        Returns:
            Generated answer based on relevant code
        """
        if (
            not hasattr(self.indexer, "vector_store")
            or self.indexer.vector_store is None
        ):
            return "No code has been indexed yet. Please ingest code files first."

        # When using mock embeddings, return a simple response
        if isinstance(self.embedding_model, MockEmbeddings):
            results = self.retrieve(query, top_k=2)
            if not results:
                return "No matching code blocks found."

            response = f"Found {len(results)} code blocks that might help:\n\n"
            for i, result in enumerate(results):
                response += f"Block {i+1} from {result['file_path']}:\n"
                response += result["code_block"] + "\n\n"
            return response

        template = """You are an expert programmer helping to answer questions about code.
Use the following code snippets as context to answer the question.

Context:
{context}

Question: {question}

Answer with specific code examples when relevant. If the answer cannot be determined from the 
context, say so clearly and offer to help in another way."""

        prompt = PromptTemplate(
            template=template, input_variables=["context", "question"]
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.indexer.vector_store.as_retriever(),
            chain_type_kwargs={"prompt": prompt},
        )

        try:
            response = qa_chain.invoke({"query": query})
            return response["result"]
        except Exception as e:
            logger.error(f"Error during QA retrieval: {str(e)}")
            return f"An error occurred: {str(e)}"
