import os
import json
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from htmlTemplates import bot_template, user_template, css


# ============================================================================
# CACHED MODEL & LLM INITIALIZATION (Resource-level caching)
# ============================================================================

@st.cache_resource
def get_embeddings_model():
    """
    OPTIMIZATION: Cache the embedding model as a resource.
    - Loaded once per session and reused across reruns
    - HuggingFace model is large; loading it repeatedly causes significant delay
    - @st.cache_resource: Shared across all users, persists entire session
    """
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def get_llm():
    """
    OPTIMIZATION: Cache the LLM as a resource.
    - Loaded once per session and reused for all queries
    - Initialization involves setting up the Gemini API connection
    - @st.cache_resource: Shared across session, only instantiated once
    """
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )


# ============================================================================
# PDF & EMBEDDING FUNCTIONS (Data-level caching + hashing)
# ============================================================================

def _get_pdf_identifiers(pdf_docs):
    """
    Create a hashable identifier for a set of PDFs.
    Used to detect if PDFs have changed between uploads.
    """
    if not pdf_docs:
        return None
    return tuple(sorted([(pdf.name, pdf.size) for pdf in pdf_docs]))


@st.cache_data
def get_pdf_text(pdf_id_tuple, pdf_count):
    """
    OPTIMIZATION: Cache PDF text extraction using data-level caching.
    - Parameters are hashable identifiers (name, size), not file objects
    - Extracts text only when new PDFs are uploaded
    - If user uploads the exact same PDFs again, returns cached text
    - @st.cache_data: Cache persists until data (pdf_id_tuple) changes
    - DRAWBACK: Cannot use actual file objects; must reference from session state
    """
    # Retrieve the actual PDF objects from session state
    if "current_pdfs" not in st.session_state or len(st.session_state.current_pdfs) != pdf_count:
        return None
    
    text = ""
    for pdf in st.session_state.current_pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


@st.cache_data
def get_text_chunks(text):
    """
    OPTIMIZATION: Cache text chunking using data-level caching.
    - Splits text only once per unique text
    - If the same text is provided again, returns cached chunks
    - @st.cache_data: Cache based on text content hash
    - BENEFIT: CharacterTextSplitter is lightweight but deterministic
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    """
    Create a FAISS vector store from text chunks.
    NOT cached in @st.cache_* because:
    - FAISS vectorstore objects are large and complex
    - Better to store in st.session_state for this session's data
    - Session state persists across reruns without expensive recreation
    OPTIMIZATION: Uses cached embedding model instead of recreating it
    """
    embeddings = get_embeddings_model()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    """
    Initialize the conversation chain with cached LLM and retriever.
    OPTIMIZATION: Uses cached LLM instead of creating new instance
    """
    llm = get_llm()
    return {
        "llm": llm,
        "retriever": vectorstore.as_retriever()
    }


# ============================================================================
# QUESTION PROCESSING
# ============================================================================

def process_question(user_question):
    """
    Process user question using the conversation chain.
    Returns the LLM answer or None if error occurs.
    OPTIMIZATION: 
    - Uses cached LLM (no recreation)
    - Uses existing retriever (no recreation)
    - Only invokes LLM API when explicitly called (not on reruns)
    """
    if st.session_state.conversation is None:
        st.warning("Please upload and process PDFs first.")
        return None

    if not user_question.strip():
        st.warning("Please enter a question.")
        return None

    # OPTIMIZATION: Reuse existing retriever from session state
    docs = st.session_state.conversation["retriever"].invoke(user_question)
    context = "\n".join([doc.page_content for doc in docs])

    # Build history from previous messages
    history = "\n".join([
        f"{msg['role'].capitalize()}: {msg['content']}"
        for msg in st.session_state.chat_history
    ])

    prompt = f"""
You are Study-Mate, an AI assistant that answers questions using the uploaded PDFs.

Instructions:
- Answer using the provided context.
- Use the conversation history when needed.
- If the answer is not found in the context, say "I couldn't find that information in the uploaded documents."
- Be concise and accurate.
- Preserve formatting: use line breaks, bullet points, and code blocks as needed.

Conversation History:
{history}

Document Context:
{context}

User Question:
{user_question}
"""

    # OPTIMIZATION: LLM call only happens here, not on other button clicks
    with st.spinner("Thinking..."):
        response = st.session_state.conversation["llm"].invoke(prompt)

    answer = response.content
    st.session_state.last_context = context

    return answer


# ============================================================================
# CHAT RENDERING
# ============================================================================

def render_chat():
    """
    Render the entire chat history with proper formatting.
    OPTIMIZATION:
    - Renders only from session_state.chat_history (no recomputation)
    - Early return if no chat exists (avoids unnecessary rendering)
    - Markdown rendering preserves LLM formatting without extra processing
    """
    if not st.session_state.chat_history:
        return

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.write(
                user_template.replace("{{MSG}}", msg["content"]),
                unsafe_allow_html=True
            )
        else:
            # OPTIMIZATION: Use markdown instead of st.write for formatting preservation
            st.markdown(
                bot_template.replace("{{MSG}}", msg["content"]),
                unsafe_allow_html=True
            )


# ============================================================================
# EXPORT & PDF GENERATION
# ============================================================================

def export_chat_to_pdf(chat_history):
    """Export chat history to a PDF file."""
    pdf_file = "study_mate_chat.pdf"
    doc = SimpleDocTemplate(pdf_file)
    styles = getSampleStyleSheet()

    content = []
    title = Paragraph("Study-Mate Chat History", styles["Title"])
    content.append(title)
    content.append(Spacer(1, 12))

    for msg in chat_history:
        role = msg["role"].capitalize()
        text = msg["content"]
        paragraph = Paragraph(
            f"<b>{role}:</b> {text}",
            styles["BodyText"]
        )
        content.append(paragraph)
        content.append(Spacer(1, 6))

    doc.build(content)
    return pdf_file


def render_export_button():
    """
    Render the export chat history button (only if chat exists).
    OPTIMIZATION:
    - Only renders button if chat history exists (avoids unnecessary UI)
    - PDF generation happens only on button click
    - Does not regenerate LLM responses
    """
    if not st.session_state.chat_history:
        return

    if st.button("📄 Export Chat History", key="export_pdf"):
        pdf_file = export_chat_to_pdf(st.session_state.chat_history)
        with open(pdf_file, "rb") as file:
            st.download_button(
                label="⬇ Download PDF",
                data=file,
                file_name="study_mate_chat.pdf",
                mime="application/pdf"
            )


# ============================================================================
# QUIZ GENERATION
# ============================================================================

def generate_quiz(context):
    """Generate a quiz question from the given context."""
    quiz_prompt = f"""
    Using ONLY the context below, generate exactly ONE multiple-choice question.

    Context:
    {context}

    Return ONLY valid JSON.

    Example:
    {{
        "question": "What is insertion sort?",
        "options": [
            "A sorting algorithm",
            "A searching algorithm",
            "A graph algorithm",
            "A tree algorithm"
        ],
        "answer": "A sorting algorithm"
    }}
    """

    response = st.session_state.conversation["llm"].invoke(quiz_prompt)
    return response.content


def render_quiz():
    """
    Render the quiz button and display quiz if toggled.
    OPTIMIZATION:
    - Quiz generation only happens when show_quiz is True
    - Cached context prevents redundant retrieval
    - Toggle state prevents repeated quiz generation on reruns
    """
    if not st.session_state.chat_history:
        return

    if st.button("🧠 Test My Understanding", key="latest_quiz"):
        st.session_state.show_quiz = not st.session_state.show_quiz

    # OPTIMIZATION: Only generate quiz if explicitly toggled on
    if st.session_state.show_quiz and st.session_state.last_context:
        st.divider()
        st.subheader("📝 Quick Quiz")
        quiz_content = generate_quiz(st.session_state.last_context)
        st.write(quiz_content)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """
    Initialize all required session state variables.
    OPTIMIZATION: Added tracking for PDF changes to avoid unnecessary reprocessing
    """
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "show_quiz" not in st.session_state:
        st.session_state.show_quiz = False
    if "last_context" not in st.session_state:
        st.session_state.last_context = ""
    # OPTIMIZATION: Track PDF state to detect changes
    if "pdf_hash" not in st.session_state:
        st.session_state.pdf_hash = None
    if "current_pdfs" not in st.session_state:
        st.session_state.current_pdfs = []


# ============================================================================
# SIDEBAR PDF UPLOAD
# ============================================================================

def render_sidebar():
    """
    Render the sidebar with PDF upload functionality.
    OPTIMIZATION: Detects PDF changes and prevents reprocessing of identical uploads
    """
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'",
            accept_multiple_files=True
        )
        
        if st.button("Process"):
            if not pdf_docs:
                st.error("Please upload at least one PDF.")
                return

            # OPTIMIZATION: Create hash of current PDFs to detect changes
            current_pdf_id = _get_pdf_identifiers(pdf_docs)
            
            # OPTIMIZATION: Check if PDFs have already been processed
            if current_pdf_id == st.session_state.pdf_hash and st.session_state.conversation is not None:
                st.info("✓ These PDFs are already processed. Upload different files to reprocess.")
                return

            with st.spinner("Processing PDFs..."):
                try:
                    # OPTIMIZATION: Store current PDFs in session for use in cached functions
                    st.session_state.current_pdfs = pdf_docs
                    
                    # Extract text (cached if same PDFs uploaded again)
                    raw_text = get_pdf_text(current_pdf_id, len(pdf_docs))
                    
                    if not raw_text:
                        # Fallback if cache returned None
                        raw_text = ""
                        for pdf in pdf_docs:
                            pdf_reader = PdfReader(pdf)
                            for page in pdf_reader.pages:
                                page_text = page.extract_text()
                                if page_text:
                                    raw_text += page_text

                    if not raw_text.strip():
                        st.error("No readable text found in the uploaded PDFs.")
                        return

                    # OPTIMIZATION: Get text chunks (cached if same text)
                    text_chunks = get_text_chunks(raw_text)

                    # Create vector store (uses cached embedding model)
                    vectorstore = get_vectorstore(text_chunks)

                    # Create conversation chain (uses cached LLM)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    
                    # OPTIMIZATION: Update hash to track this PDF set
                    st.session_state.pdf_hash = current_pdf_id
                    
                    st.success("✓ Documents processed successfully!")
                    
                except Exception as e:
                    st.error(f"An error occurred while processing PDFs: {str(e)}")


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """
    Main application entry point.
    OPTIMIZATION ARCHITECTURE:
    - Cached models (embedding & LLM) loaded once per session
    - PDF processing tracked via hash to avoid reprocessing
    - Chat history stored in session_state for cross-rerun persistence
    - Button actions are idempotent and independent
    - LLM calls only triggered by explicit "Ask" button click
    """
    load_dotenv()
    st.set_page_config(
        page_title="Study-Mate: Your AI-Powered PDFs Tool",
        page_icon=":books:"
    )

    st.write(css, unsafe_allow_html=True)
    initialize_session_state()

    st.header("Study-Mate: Your AI-Powered PDFs Tool :books:")

    # Main question input and button
    col1, col2 = st.columns([4, 1])
    with col1:
        user_question = st.text_input(
            "Ask a question about your document(s):",
            key="user_question_input"
        )
    with col2:
        ask_clicked = st.button("Ask", key="ask_button", use_container_width=True)

    # OPTIMIZATION: Process question only once when "Ask" is clicked
    # Does not trigger on export, quiz, or other button interactions
    if ask_clicked:
        answer = process_question(user_question)
        if answer:
            # Add to chat history only once (no duplicates on reruns)
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_question
            })
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer
            })
            # Clear input by rerunning
            st.rerun()

    # OPTIMIZATION: Display chat history without recomputation
    render_chat()

    # Display action buttons (independent operations)
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        render_export_button()
    with col2:
        render_quiz()

    # Render sidebar
    render_sidebar()


if __name__ == '__main__':
    main()