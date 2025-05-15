import os
import json
import pickle
import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.code_processor import CodeBlock

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockEmbeddings:
    """Mock embedding provider for testing without API keys."""

    def __init__(self, size: int = 1536):
        self.size = size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate random embeddings for testing."""
        return [self._get_embedding() for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        """Generate random embedding for a query."""
        return self._get_embedding()

    def _get_embedding(self) -> List[float]:
        """Generate a single random embedding."""
        return list(np.random.rand(self.size).astype(float))

    def __call__(self, text: str) -> List[float]:
        """Make the embedding model callable."""
        return self._get_embedding()


class CodeIndexer:
    """Indexer for code blocks using vector embeddings."""

    def __init__(
        self,
        embedding_model: Optional[Any] = None,
        index_path: str = "code_index",
        model_name: str = "nomic-embed-text",
    ):
        """Initialize the code indexer.

        Args:
            embedding_model: Model to use for embeddings (defaults to Ollama)
            index_path: Path to store the index
            model_name: Name of Ollama model to use for embeddings
        """
        # Make index_path absolute if it's not already
        if not os.path.isabs(index_path):
            self.index_path = os.path.abspath(index_path)
        else:
            self.index_path = index_path

        logger.info(f"Using index path: {self.index_path}")

        # Default to Ollama embeddings if none provided
        if embedding_model is None:
            self.embedding_model = OllamaEmbeddings(model=model_name)
            logger.info(f"Using Ollama embeddings with model: {model_name}")
        else:
            self.embedding_model = embedding_model

        # Create the index directory if it doesn't exist
        os.makedirs(
            (
                os.path.dirname(self.index_path)
                if os.path.dirname(self.index_path)
                else "."
            ),
            exist_ok=True,
        )

        # Block metadata storage
        self.metadata_path = f"{self.index_path}_metadata.json"
        self.metadata_store = self._load_metadata()

        # Store the model info in the metadata
        model_info_path = f"{self.index_path}_model_info.json"
        current_model_info = {
            "model_name": model_name,
            "embedding_type": type(self.embedding_model).__name__,
        }

        # Check if model has changed since last run
        force_recreate = False
        if os.path.exists(model_info_path):
            try:
                with open(model_info_path, "r") as f:
                    previous_model_info = json.load(f)

                if previous_model_info != current_model_info:
                    logger.warning(
                        f"Embedding model changed from {previous_model_info['model_name']} to {current_model_info['model_name']}. Will recreate index."
                    )
                    force_recreate = True
            except Exception as e:
                logger.warning(
                    f"Could not read model info file: {str(e)}. Will recreate index."
                )
                force_recreate = True

        # Save current model info
        try:
            with open(model_info_path, "w") as f:
                json.dump(current_model_info, f)
        except Exception as e:
            logger.warning(f"Could not save model info: {str(e)}")

        # Initialize or load vector store
        if force_recreate:
            # Delete existing index if model changed
            self._clear_index_files()
            self.vector_store = None
            logger.info("Creating new vector index after model change")
        elif (
            os.path.exists(f"{self.index_path}.faiss")
            or os.path.exists(f"{self.index_path}/index.faiss")
            or os.path.isdir(self.index_path)
            and os.path.exists(f"{self.index_path}/index.faiss")
        ):
            logger.info(f"Loading existing index from {self.index_path}")
            try:
                self.vector_store = self._load_index()
            except Exception as e:
                logger.error(f"Failed to load index: {str(e)}. Creating new index.")
                self._clear_index_files()
                self.vector_store = None
        else:
            logger.info("Creating new vector index")
            self.vector_store = None

    def _clear_index_files(self):
        """Remove existing index files."""
        try:
            if os.path.exists(f"{self.index_path}.faiss"):
                os.remove(f"{self.index_path}.faiss")
            if os.path.exists(f"{self.index_path}.pkl"):
                os.remove(f"{self.index_path}.pkl")
            if os.path.isdir(self.index_path):
                faiss_path = os.path.join(self.index_path, "index.faiss")
                pkl_path = os.path.join(self.index_path, "index.pkl")
                if os.path.exists(faiss_path):
                    os.remove(faiss_path)
                if os.path.exists(pkl_path):
                    os.remove(pkl_path)
            logger.info("Cleared existing index files")
        except Exception as e:
            logger.error(f"Error clearing index files: {str(e)}")

    def _load_index(self) -> Optional[FAISS]:
        """Load existing index from disk."""
        try:
            if os.path.exists(f"{self.index_path}.faiss"):
                # Old style: file.faiss
                return FAISS.load_local(
                    self.index_path,
                    self.embedding_model,
                    allow_dangerous_deserialization=True,
                )
            elif os.path.isdir(self.index_path) and os.path.exists(
                f"{self.index_path}/index.faiss"
            ):
                # New style: directory/index.faiss
                return FAISS.load_local(
                    self.index_path,
                    self.embedding_model,
                    allow_dangerous_deserialization=True,
                )
            else:
                logger.error(f"Index files not found at {self.index_path}")
                return None
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            return None

    def _save_index(self) -> None:
        """Save index to disk."""
        if self.vector_store:
            self.vector_store.save_local(self.index_path)
            logger.info(f"Index saved to {self.index_path}")

    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata store from disk."""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata: {str(e)}")
                return {}
        return {}

    def _save_metadata(self) -> None:
        """Save metadata store to disk."""
        try:
            with open(self.metadata_path, "w") as f:
                json.dump(self.metadata_store, f, indent=2)
            logger.info(f"Metadata saved to {self.metadata_path}")
        except Exception as e:
            logger.error(f"Error saving metadata: {str(e)}")

    def add_documents(self, code_blocks: List[CodeBlock]) -> None:
        """Add code blocks to the index.

        Args:
            code_blocks: List of CodeBlock objects to add to the index
        """
        if not code_blocks:
            logger.warning("No code blocks to add to index")
            return

        logger.info(f"Adding {len(code_blocks)} code blocks to index")

        # Convert code blocks to LangChain documents
        documents = []
        for block in code_blocks:
            block_dict = block.to_dict()

            # Combine code and comments for better context
            content = f"{block.code}\n\n"
            if block.comments:
                content += f"Comments:\n{block.comments}"

            doc = Document(
                page_content=content,
                metadata={
                    "file_path": block.file_path,
                    "start_line": block.start_line,
                    "end_line": block.end_line,
                    "language": block.language,
                    "block_type": block.block_type,
                    "id": f"{block.file_path}:{block.start_line}-{block.end_line}",
                },
            )
            documents.append(doc)

            # Store full metadata separately (including the actual code)
            doc_id = f"{block.file_path}:{block.start_line}-{block.end_line}"
            self.metadata_store[doc_id] = block_dict

        # Create or update the vector store
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embedding_model)
        else:
            self.vector_store.add_documents(documents)

        # Save the index and metadata
        self._save_index()
        self._save_metadata()

        logger.info(f"Successfully added {len(code_blocks)} code blocks to index")

    def get_block_by_id(self, block_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a code block by its ID.

        Args:
            block_id: Block ID in the format 'file_path:start_line-end_line'

        Returns:
            Code block metadata and content, or None if not found
        """
        return self.metadata_store.get(block_id)

    def delete_blocks(self, file_path: str) -> None:
        """Delete all blocks from a specific file.

        This requires reindexing as FAISS doesn't support document deletion.

        Args:
            file_path: Path to the file whose blocks should be deleted
        """
        if not self.metadata_store:
            return

        # Get all block IDs not from this file
        remaining_ids = [
            block_id
            for block_id in self.metadata_store
            if not block_id.startswith(f"{file_path}:")
        ]

        # If no blocks were deleted, nothing to do
        if len(remaining_ids) == len(self.metadata_store):
            logger.info(f"No blocks found for file {file_path}")
            return

        # Get the remaining blocks
        remaining_blocks = [self.metadata_store[block_id] for block_id in remaining_ids]

        # Clear existing index
        self.vector_store = None
        self.metadata_store = {}

        # Re-add the remaining blocks if any
        if remaining_blocks:
            # Convert dicts back to CodeBlock objects
            code_blocks = [
                CodeBlock(
                    code=block["content"],
                    file_path=block["file_path"],
                    start_line=block["start_line"],
                    end_line=block["end_line"],
                    language=block["language"],
                    block_type=block["block_type"],
                    comments=block["comments"],
                )
                for block in remaining_blocks
            ]

            self.add_documents(code_blocks)
        else:
            # No blocks left, just save empty metadata
            self._save_metadata()

            # Remove index files if they exist
            if os.path.exists(f"{self.index_path}.faiss"):
                os.remove(f"{self.index_path}.faiss")
            if os.path.exists(f"{self.index_path}.pkl"):
                os.remove(f"{self.index_path}.pkl")

        logger.info(f"Removed blocks for file {file_path}")

    def clear(self) -> None:
        """Clear the entire index."""
        self.vector_store = None
        self.metadata_store = {}

        # Save empty metadata
        self._save_metadata()

        # Remove index files
        self._clear_index_files()

        logger.info("Index cleared")
