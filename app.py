import shutil
import streamlit as st
import os
import json
import time
import bcrypt
import uuid
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from dotenv import load_dotenv
from src.retriever import CodeRetriever
from src.code_processor import CodeProcessor
from src.indexer import CodeIndexer

# Load environment variables
load_dotenv()

# Get configuration from environment variables with defaults
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5")
BASE_INDEX_PATH = os.getenv("INDEX_PATH", "code_index")

# Set Ollama base URL for all Ollama models
os.environ["OLLAMA_API_BASE"] = OLLAMA_BASE_URL

# Define paths for user data and conversations
USER_DATA_DIR = Path("user_data")
USER_DATA_DIR.mkdir(exist_ok=True)
USER_FILE = USER_DATA_DIR / "users.json"
CONVERSATION_DIR = USER_DATA_DIR / "conversations"
CONVERSATION_DIR.mkdir(exist_ok=True)
USER_INDEX_DIR = USER_DATA_DIR / "indices"
USER_INDEX_DIR.mkdir(exist_ok=True)

# Define app title and configuration
st.set_page_config(
    page_title="Code RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #0F52BA;
        margin-bottom: 1rem;
    }
    
    .chat-container {
        border-radius: 10px;
        border: 1px solid #e6e6e6;
        padding: 1rem;
        margin-bottom: 1rem;
        overflow-y: auto;
        max-height: 60vh;
    }
    
    .user-message {
        background-color: #e6f7ff;
        padding: 0.8rem;
        border-radius: 8px;
        margin-bottom: 0.8rem;
        border-left: 4px solid #1890ff;
    }
    
    .assistant-message {
        background-color: #f6f6f6;
        padding: 0.8rem;
        border-radius: 8px;
        margin-bottom: 0.8rem;
        border-left: 4px solid #52c41a;
    }
    
    .code-block {
        background-color: #f8f8f8;
        padding: 0.5rem;
        border-radius: 5px;
        font-family: monospace;
        overflow-x: auto;
        border: 1px solid #ddd;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ====== User Management Functions ======


def load_users() -> Dict:
    """Load user data from the JSON file."""
    if USER_FILE.exists():
        with open(USER_FILE, "r") as f:
            return json.load(f)
    return {}


def save_users(users: Dict):
    """Save user data to the JSON file."""
    with open(USER_FILE, "w") as f:
        json.dump(users, f, indent=4)


def hash_password(password: str) -> str:
    """Hash password using bcrypt."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(stored_hash: str, provided_password: str) -> bool:
    """Verify password against stored hash."""
    return bcrypt.checkpw(provided_password.encode(), stored_hash.encode())


def register_user(username: str, password: str) -> bool:
    """Register a new user."""
    users = load_users()
    if username in users:
        return False

    users[username] = {
        "password_hash": hash_password(password),
        "created_at": datetime.now().isoformat(),
    }
    save_users(users)

    # Create user-specific directories for conversations and index
    user_conv_dir = CONVERSATION_DIR / username
    user_conv_dir.mkdir(exist_ok=True)

    user_index_dir = USER_INDEX_DIR / username
    user_index_dir.mkdir(exist_ok=True)

    return True


def authenticate_user(username: str, password: str) -> bool:
    """Authenticate a user."""
    users = load_users()
    if username not in users:
        return False

    return verify_password(users[username]["password_hash"], password)


def get_user_index_path(username: str) -> str:
    """Get the index path for a specific user."""
    return str(USER_INDEX_DIR / username / "code_index")


# ====== Conversation Management Functions ======


def get_user_conversations(username: str) -> List[Dict[str, Union[str, datetime]]]:
    """Get list of conversation IDs for a user."""
    user_dir = CONVERSATION_DIR / username
    if not user_dir.exists():
        return []

    conversations = []
    for file_path in user_dir.glob("*.json"):
        conv_id = file_path.stem
        with open(file_path, "r") as f:
            data = json.load(f)
            if "title" in data and "created_at" in data:
                conversations.append(
                    {
                        "id": conv_id,
                        "title": data.get("title", "Untitled"),
                        "created_at": data.get("created_at", ""),
                    }
                )

    # Sort by creation date, newest first
    return sorted(conversations, key=lambda x: x.get("created_at", ""), reverse=True)


def create_conversation(username: str, title: str) -> str:
    """Create a new conversation for the user."""
    user_dir = CONVERSATION_DIR / username
    user_dir.mkdir(exist_ok=True)

    conv_id = str(uuid.uuid4())
    conversation = {
        "id": conv_id,
        "title": title,
        "created_at": datetime.now().isoformat(),
        "messages": [],
    }

    with open(user_dir / f"{conv_id}.json", "w") as f:
        json.dump(conversation, f, indent=4)

    return conv_id


def load_conversation(username: str, conv_id: str) -> Optional[Dict[str, Any]]:
    """Load a conversation."""
    file_path = CONVERSATION_DIR / username / f"{conv_id}.json"
    if not file_path.exists():
        return None

    with open(file_path, "r") as f:
        return json.load(f)


def save_conversation(username: str, conv_id: str, conversation: Dict):
    """Save a conversation."""
    user_dir = CONVERSATION_DIR / username
    user_dir.mkdir(exist_ok=True)

    with open(user_dir / f"{conv_id}.json", "w") as f:
        json.dump(conversation, f, indent=4)


def add_message(username: str, conv_id: str, role: str, content: str):
    """Add a message to a conversation."""
    conversation = load_conversation(username, conv_id)
    if not conversation:
        return

    conversation["messages"].append(
        {"role": role, "content": content, "timestamp": datetime.now().isoformat()}
    )

    save_conversation(username, conv_id, conversation)


def rename_conversation(username: str, conv_id: str, new_title: str) -> bool:
    """Rename a conversation."""
    conversation = load_conversation(username, conv_id)
    if not conversation:
        return False

    conversation["title"] = new_title
    save_conversation(username, conv_id, conversation)
    return True


def delete_conversation(username: str, conv_id: str) -> bool:
    """Delete a conversation."""
    file_path = CONVERSATION_DIR / username / f"{conv_id}.json"
    if not file_path.exists():
        return False

    file_path.unlink()
    return True


# ====== RAG System Integration ======


@st.cache_resource(hash_funcs={str: lambda x: x})
def get_retriever(username: str):
    """Get or initialize the code retriever for a specific user."""
    user_index_path = get_user_index_path(username)

    try:
        retriever = CodeRetriever(
            index_path=user_index_path,
            model_name=OLLAMA_LLM_MODEL,
            embedding_model_name=OLLAMA_EMBED_MODEL,
        )
        return retriever
    except Exception as e:
        st.error(f"Error initializing retriever: {str(e)}")
        return None


def process_query(username: str, query: str, top_k: int = 3) -> Tuple[str, List[Dict]]:
    """Process a user query and return the answer and relevant code blocks."""
    retriever = get_retriever(username)
    if not retriever:
        return "Error: Could not initialize the code retriever.", []

    try:
        # Get the answer
        answer = retriever.qa_retrieval(query)

        # Get relevant code blocks
        results = retriever.retrieve(query, top_k=top_k)

        return answer, results
    except Exception as e:
        return f"Error processing query: {str(e)}", []


def ingest_code(username: str, path: str) -> str:
    """Ingest code files or directories into the user's index."""
    user_index_path = get_user_index_path(username)

    try:
        processor = CodeProcessor()
        indexer = CodeIndexer(
            model_name=OLLAMA_EMBED_MODEL,
            index_path=user_index_path,
        )

        if os.path.isfile(path):
            code_blocks = processor.process_file(path)
            indexer.add_documents(code_blocks)
        elif os.path.isdir(path):
            code_blocks = processor.process_directory(path)
            indexer.add_documents(code_blocks)
        else:
            return f"Error: Path not found: {path}"

        return f"Successfully ingested code from {path} into your personal index"
    except Exception as e:
        return f"Error during ingestion: {str(e)}"


def get_index_status(username: str) -> Dict[str, Any]:
    """Get status of user's code index."""
    user_index_path = get_user_index_path(username)
    index_file = Path(f"{user_index_path}.faiss")
    index_dir = Path(user_index_path)

    if index_file.exists() or (
        index_dir.exists() and (index_dir / "index.faiss").exists()
    ):
        # Try to get document count if the method exists
        try:
            retriever = get_retriever(username)
            doc_count = None
            if retriever and hasattr(retriever.indexer, "get_document_count"):
                doc_count = retriever.indexer.get_document_count()
                return {
                    "exists": True,
                    "document_count": doc_count,
                    "message": f"Your code index contains {doc_count} code blocks.",
                }
        except Exception:
            pass

        return {
            "exists": True,
            "document_count": None,
            "message": "Your code index exists.",
        }

    return {
        "exists": False,
        "document_count": 0,
        "message": "No code index found. Please ingest some code files.",
    }


def reset_user_index(username: str) -> str:
    """Reset (delete) a user's code index."""
    user_index_path = get_user_index_path(username)

    try:
        # Delete index directory if it exists
        index_dir = Path(user_index_path)
        if index_dir.exists():
            shutil.rmtree(index_dir)

        # Clear cache to reinitialize retriever
        st.cache_resource.clear()

        return "Your code index has been reset successfully."
    except Exception as e:
        return f"Error resetting index: {str(e)}"


# ====== UI Components ======


def render_login_page():
    """Render the login page."""
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown(
            '<h1 class="main-header">Code RAG System</h1>', unsafe_allow_html=True
        )

        tab1, tab2 = st.tabs(["Login", "Register"])

        with tab1:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login")

                if submitted:
                    if authenticate_user(username, password):
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.success("Login successful!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Invalid username or password")

        with tab2:
            with st.form("register_form"):
                new_username = st.text_input("New Username")
                new_password = st.text_input("New Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                submitted = st.form_submit_button("Register")

                if submitted:
                    if not new_username or not new_password:
                        st.error("Username and password are required")
                    elif new_password != confirm_password:
                        st.error("Passwords do not match")
                    elif register_user(new_username, new_password):
                        st.success("Registration successful! You can now log in.")
                    else:
                        st.error("Username already exists")


def render_sidebar():
    """Render the sidebar with conversations and options."""
    st.sidebar.markdown(f"## Welcome, {st.session_state.username}!")

    if st.sidebar.button("Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.sidebar.markdown("---")

    # Code index status
    index_status = get_index_status(st.session_state.username)
    st.sidebar.markdown("## Your Code Index")
    st.sidebar.info(index_status["message"])

    # Reset index button
    if index_status["exists"] and st.sidebar.button("Reset Index"):
        result = reset_user_index(st.session_state.username)
        st.sidebar.info(result)
        time.sleep(2)
        st.rerun()

    # Code ingestion section
    st.sidebar.markdown("## Ingest Code")

    with st.sidebar.form("ingest_form"):
        path = st.text_input("Path to code file or directory")
        submitted = st.form_submit_button("Ingest Code")

        if submitted and path:
            with st.sidebar:
                with st.spinner("Ingesting code..."):
                    result = ingest_code(st.session_state.username, path)
                    st.info(result)
                    st.rerun()

    st.sidebar.markdown("---")

    # New conversation button
    if st.sidebar.button("New Conversation"):
        title = f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        conv_id = create_conversation(st.session_state.username, title)
        st.session_state.current_conversation = conv_id
        st.rerun()

    # List existing conversations
    st.sidebar.markdown("## Your Conversations")
    conversations = get_user_conversations(st.session_state.username)

    for conv in conversations:
        col1, col2, col3 = st.sidebar.columns([4, 1, 1])
        with col1:
            if st.button(
                f"📝 {conv.get('title', 'Untitled')}", key=f"conv_{conv.get('id', '')}"
            ):
                st.session_state.current_conversation = conv.get("id", "")
                st.rerun()
        with col2:
            if st.button("✏️", key=f"edit_{conv.get('id', '')}"):
                st.session_state.editing_conv = conv.get("id", "")
                st.session_state.editing_title = conv.get("title", "Untitled")
        with col3:
            if st.button("🗑️", key=f"delete_{conv.get('id', '')}"):
                st.session_state.deleting_conv = conv.get("id", "")

    # Handle conversation editing
    if "editing_conv" in st.session_state:
        with st.sidebar.form("rename_form"):
            new_title = st.text_input("New title", value=st.session_state.editing_title)
            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("Save"):
                    rename_conversation(
                        st.session_state.username,
                        st.session_state.editing_conv,
                        new_title,
                    )
                    del st.session_state.editing_conv
                    del st.session_state.editing_title
                    st.rerun()
            with col2:
                if st.form_submit_button("Cancel"):
                    del st.session_state.editing_conv
                    del st.session_state.editing_title
                    st.rerun()

    # Handle conversation deletion
    if "deleting_conv" in st.session_state:
        with st.sidebar.form("delete_form"):
            st.warning("Are you sure you want to delete this conversation?")
            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("Confirm"):
                    delete_conversation(
                        st.session_state.username, st.session_state.deleting_conv
                    )
                    if (
                        st.session_state.get("current_conversation")
                        == st.session_state.deleting_conv
                    ):
                        # If we're deleting the active conversation, create a new one
                        title = (
                            f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                        )
                        st.session_state.current_conversation = create_conversation(
                            st.session_state.username, title
                        )
                    del st.session_state.deleting_conv
                    st.rerun()
            with col2:
                if st.form_submit_button("Cancel"):
                    del st.session_state.deleting_conv
                    st.rerun()


def render_chat_interface():
    """Render the chat interface for the current conversation."""
    # Get current conversation
    if "current_conversation" not in st.session_state:
        # Create a new conversation if none exists
        title = f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        conv_id = create_conversation(st.session_state.username, title)
        st.session_state.current_conversation = conv_id

    conversation = load_conversation(
        st.session_state.username, st.session_state.current_conversation
    )
    if not conversation:
        st.error("Conversation not found")
        return

    # Display conversation title
    st.markdown(
        f"<h1 class='main-header'>{conversation.get('title', 'Untitled')}</h1>",
        unsafe_allow_html=True,
    )

    # # Display messages
    # st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    for message in conversation.get("messages", []):
        if message.get("role") == "user":
            st.markdown(
                f'<div class="user-message"><b>You:</b> {message.get("content", "")}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="assistant-message"><b>Assistant:</b> {message.get("content", "")}</div>',
                unsafe_allow_html=True,
            )

    # st.markdown("</div>", unsafe_allow_html=True)

    # Check index status before allowing questions
    index_status = get_index_status(st.session_state.username)
    if not index_status["exists"]:
        st.warning(
            "You need to ingest some code files before asking questions. Use the 'Ingest Code' section in the sidebar."
        )

    # Input for new message
    query = st.text_area("Ask a question about your code:", height=100)
    col1, col2 = st.columns([1, 6])

    with col1:
        top_k = st.number_input("Results:", min_value=1, max_value=10, value=3)

    with col2:
        if st.button(
            "Send", use_container_width=True, disabled=not index_status["exists"]
        ):
            if query:
                # Add user message to conversation
                add_message(
                    st.session_state.username,
                    st.session_state.current_conversation,
                    "user",
                    query,
                )

                # Process query
                with st.spinner("Processing your query..."):
                    answer, results = process_query(
                        st.session_state.username, query, top_k
                    )

                # Add assistant message to conversation
                assistant_response = answer
                if results:
                    assistant_response += "\n\n**Relevant Code Blocks:**\n\n"
                    for i, result in enumerate(results):
                        file_path = result.get("file_path", "")
                        code_block = result.get("code_block", "")
                        assistant_response += f"**Block {i+1}** from `{file_path}`:\n```\n{code_block}\n```\n\n"

                add_message(
                    st.session_state.username,
                    st.session_state.current_conversation,
                    "assistant",
                    assistant_response,
                )

                # Rerun to update the UI
                st.rerun()


# ====== Main App Logic ======


def main():
    # Initialize session state
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    # Show login page or main app based on login status
    if not st.session_state.logged_in:
        render_login_page()
    else:
        render_sidebar()
        render_chat_interface()


if __name__ == "__main__":
    main()
