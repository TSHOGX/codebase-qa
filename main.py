import argparse
from src.code_processor import CodeProcessor
from src.indexer import CodeIndexer, MockEmbeddings
from src.retriever import CodeRetriever
import os
from dotenv import load_dotenv

load_dotenv()

# Get configuration from environment variables with defaults
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5")
INDEX_PATH = os.getenv("INDEX_PATH", "code_index")


def main():
    parser = argparse.ArgumentParser(description="Code RAG System")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest code files")
    ingest_parser.add_argument("path", help="Path to code file or directory")
    ingest_parser.add_argument(
        "--mock", action="store_true", help="Use mock embeddings for testing"
    )
    ingest_parser.add_argument(
        "--embed-model",
        type=str,
        default=OLLAMA_EMBED_MODEL,
        help="Ollama embedding model to use",
    )
    ingest_parser.add_argument(
        "--index-path", type=str, default=INDEX_PATH, help="Path to store the index"
    )

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the code database")
    query_parser.add_argument("query", help="Natural language query")
    query_parser.add_argument(
        "--top_k", type=int, default=3, help="Number of results to return"
    )
    query_parser.add_argument(
        "--mock", action="store_true", help="Use mock embeddings for testing"
    )
    query_parser.add_argument(
        "--embed-model",
        type=str,
        default=OLLAMA_EMBED_MODEL,
        help="Ollama embedding model to use",
    )
    query_parser.add_argument(
        "--llm-model",
        type=str,
        default=OLLAMA_LLM_MODEL,
        help="Ollama LLM model to use",
    )
    query_parser.add_argument(
        "--index-path", type=str, default=INDEX_PATH, help="Path to the index"
    )

    # QA command
    qa_parser = subparsers.add_parser("qa", help="Ask questions about the code")
    qa_parser.add_argument("question", help="Natural language question about the code")
    qa_parser.add_argument(
        "--mock", action="store_true", help="Use mock embeddings for testing"
    )
    qa_parser.add_argument(
        "--embed-model",
        type=str,
        default=OLLAMA_EMBED_MODEL,
        help="Ollama embedding model to use",
    )
    qa_parser.add_argument(
        "--llm-model",
        type=str,
        default=OLLAMA_LLM_MODEL,
        help="Ollama LLM model to use",
    )
    qa_parser.add_argument(
        "--index-path", type=str, default=INDEX_PATH, help="Path to the index"
    )

    args = parser.parse_args()

    # Set Ollama base URL for all Ollama models
    os.environ["OLLAMA_API_BASE"] = OLLAMA_BASE_URL

    if args.command == "ingest":
        processor = CodeProcessor()

        # Use mock embeddings if specified
        embedding_model = MockEmbeddings() if args.mock else None
        indexer = CodeIndexer(
            embedding_model=embedding_model,
            model_name=args.embed_model,
            index_path=args.index_path,
        )

        if os.path.isfile(args.path):
            code_blocks = processor.process_file(args.path)
            indexer.add_documents(code_blocks)
        elif os.path.isdir(args.path):
            code_blocks = processor.process_directory(args.path)
            indexer.add_documents(code_blocks)
        else:
            print(f"Path not found: {args.path}")
            return

        print(f"Successfully ingested code from {args.path}")

    elif args.command == "query":
        # Use mock embeddings if specified
        embedding_model = MockEmbeddings() if args.mock else None
        retriever = CodeRetriever(
            embedding_model=embedding_model,
            model_name=args.llm_model,
            embedding_model_name=args.embed_model,
            index_path=args.index_path,
        )
        results = retriever.retrieve(args.query, top_k=args.top_k)

        print(f"\nResults for query: '{args.query}'\n")
        for i, result in enumerate(results):
            print(f"Result {i+1}:")
            print(f"File: {result['file_path']}")
            print(f"Score: {result['score']:.4f}")
            print(f"Code:\n{result['code_block']}")
            print("-" * 80)

    elif args.command == "qa":
        # Use mock embeddings if specified
        embedding_model = MockEmbeddings() if args.mock else None
        retriever = CodeRetriever(
            embedding_model=embedding_model,
            model_name=args.llm_model,
            embedding_model_name=args.embed_model,
            index_path=args.index_path,
        )
        answer = retriever.qa_retrieval(args.question)

        print(f"\nQuestion: {args.question}\n")
        print(f"Answer:\n{answer}")


if __name__ == "__main__":
    main()
