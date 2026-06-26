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
# PDF & EMBEDDING FUNCTIONS
# ============================================================================

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF documents."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


def get_text_chunks(text):
    """Split text into chunks for vector storage."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    """Create a FAISS vector store from text chunks."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    """Initialize the LLM and retriever for the conversation chain."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

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
    """
    if st.session_state.conversation is None:
        st.warning("Please upload and process PDFs first.")
        return None

    if not user_question.strip():
        st.warning("Please enter a question.")
        return None

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

    with st.spinner("Thinking..."):
        response = st.session_state.conversation["llm"].invoke(prompt)

    answer = response.content
    st.session_state.last_context = context

    return answer


# ============================================================================
# CHAT RENDERING
# ============================================================================

def render_chat():
    """Render the entire chat history with proper formatting."""
    if not st.session_state.chat_history:
        return

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.write(
                user_template.replace("{{MSG}}", msg["content"]),
                unsafe_allow_html=True
            )
        else:
            # Use markdown to preserve formatting from LLM responses
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
    """Render the export chat history button (only if chat exists)."""
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
    """Render the quiz button and display quiz if toggled."""
    if not st.session_state.chat_history:
        return

    if st.button("🧠 Test My Understanding", key="latest_quiz"):
        st.session_state.show_quiz = not st.session_state.show_quiz

    if st.session_state.show_quiz and st.session_state.last_context:
        st.divider()
        st.subheader("📝 Quick Quiz")
        quiz_content = generate_quiz(st.session_state.last_context)
        st.write(quiz_content)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """Initialize all required session state variables."""
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "show_quiz" not in st.session_state:
        st.session_state.show_quiz = False
    if "last_context" not in st.session_state:
        st.session_state.last_context = ""


# ============================================================================
# SIDEBAR PDF UPLOAD
# ============================================================================

def render_sidebar():
    """Render the sidebar with PDF upload functionality."""
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

            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)

                if not raw_text.strip():
                    st.error("No readable text found in the uploaded PDFs.")
                    return

                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("Documents processed successfully!")


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
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

    # Process question only when "Ask" button is clicked
    if ask_clicked:
        answer = process_question(user_question)
        if answer:
            # Add to chat history only once
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

    # Display chat history
    render_chat()

    # Display action buttons
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