import os
import json
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
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
    
    # OPTIMIZATION: Reset quiz when new answer is generated
    # This prevents showing old quiz for new questions
    st.session_state.quiz_data = None
    st.session_state.quiz_answers = []
    st.session_state.quiz_submitted = False
    st.session_state.show_simplification = False
    st.session_state.simplified_explanation = ""

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

def _safe_paragraph_text(text):
    escaped = (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
    )
    return escaped.replace("\n", "<br />\n")


def _text_block_is_preformatted(block):
    lines = block.splitlines()
    if any(line.startswith('    ') or line.startswith('\t') for line in lines):
        return True
    code_markers = ['for ', 'while ', 'if ', 'else:', 'elif ', 'return ', 'def ', 'class ', '=>', '->', '://']
    if sum(marker in block for marker in code_markers) >= 2:
        return True
    return False


def _create_flowables_for_text(text, styles):
    flowables = []
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = [block.strip() for block in normalized_text.split("\n\n") if block.strip()]

    for idx, block in enumerate(blocks):
        if _text_block_is_preformatted(block):
            flowables.append(Preformatted(block, styles["Code"]))
        else:
            flowables.append(Paragraph(_safe_paragraph_text(block), styles["BodyText"]))

        if idx < len(blocks) - 1:
            flowables.append(Spacer(1, 8))

    return flowables


def _make_pdf_flowables_for_message(role, text, styles):
    flowables = []
    role_header = Paragraph(f"<b>{role}:</b>", styles["Heading4"])
    flowables.append(role_header)
    flowables.append(Spacer(1, 4))

    if not text:
        flowables.append(Paragraph("<i>(No content)</i>", styles["BodyText"]))
        flowables.append(Spacer(1, 12))
        return flowables

    flowables.extend(_create_flowables_for_text(text, styles))
    flowables.append(Spacer(1, 12))
    return flowables


def export_chat_to_pdf(chat_history):
    """Export chat history to a PDF file."""
    pdf_file = "study_mate_chat.pdf"
    doc = SimpleDocTemplate(
        pdf_file,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    styles = getSampleStyleSheet()

    if "Code" not in styles:
        styles.add(ParagraphStyle(
            name="Code",
            fontName="Courier",
            fontSize=10,
            leading=14,
            leftIndent=12,
            rightIndent=12,
            spaceAfter=6,
            wordWrap="CJK"
        ))
    if "Header" not in styles:
        styles.add(ParagraphStyle(
            name="Header",
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            spaceAfter=6
        ))

    content = []
    header_title = Paragraph("Study-Mate – Chat History", styles["Header"])
    timestamp = Paragraph(
        f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        styles["Normal"]
    )
    content.extend([header_title, timestamp, Spacer(1, 16)])

    for msg in chat_history:
        role = msg["role"].capitalize()
        text = msg["content"]
        content.extend(_make_pdf_flowables_for_message(role, text, styles))

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
# QUIZ GENERATION (5 Questions + Evaluation)
# ============================================================================

def generate_quiz_questions(context):
    """
    Generate exactly 5 conceptual multiple-choice questions from the context.
    OPTIMIZATION:
    - Generated once per assistant response
    - Stored in session state (not regenerated on reruns)
    - Focuses on conceptual understanding, not memorization
    """
    quiz_prompt = f"""
You are an educational expert. Generate exactly 5 multiple-choice questions to test conceptual understanding of the following content. Do NOT generate memorization questions.

Content:
{context}

Requirements:
- Each question should test deeper understanding, application, or critical thinking
- Each question must have exactly 4 options
- Exactly ONE option is correct
- Options should be plausible but clearly distinguishable
- Questions should cover different aspects of the content

Return a JSON array (ONLY valid JSON, no other text). Example format:
[
  {{
    "question": "Which of the following best explains why insertion sort is less efficient than merge sort for large datasets?",
    "options": ["It has a higher space complexity", "It performs more comparisons in the average case", "It requires more iterations per element", "It cannot handle duplicate values"],
    "correct_answer": "It performs more comparisons in the average case",
    "explanation": "Insertion sort has O(n²) average time complexity due to the nested loop structure that compares elements sequentially."
  }}
]

Generate 5 questions now:
"""

    response = st.session_state.conversation["llm"].invoke(quiz_prompt)
    
    # Parse JSON response carefully
    try:
        # Extract JSON from response (may have extra text)
        import re
        json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
        if json_match:
            questions = json.loads(json_match.group())
            return questions
        else:
            # Fallback if JSON not found
            questions = json.loads(response.content)
            return questions
    except json.JSONDecodeError:
        st.error("Failed to parse quiz questions. Please try again.")
        return None


def identify_weak_topics(questions, answers):
    """
    Identify weak topics based on incorrect answers.
    OPTIMIZATION:
    - Analyzes patterns in wrong answers
    - No additional LLM call (uses stored data)
    """
    weak_topics = []
    
    if not questions or not answers:
        return weak_topics
    
    for idx, q in enumerate(questions):
        if idx < len(answers) and answers[idx] != q.get("correct_answer"):
            # Extract topic from question
            topic = q.get("question", f"Question {idx+1}").split("?")[0][:50]
            weak_topics.append({
                "question_num": idx + 1,
                "topic": topic,
                "concept": q.get("explanation", "")[:100],
                "user_answer": answers[idx],
                "correct_answer": q.get("correct_answer"),
                "explanation": q.get("explanation", "")
            })
    
    return weak_topics


def get_quiz_summary_topics(weak_topics):
    """Build a short list of quiz topics based on incorrect answers."""
    if not weak_topics:
        return []
    topics = []
    for topic in weak_topics:
        text = topic.get("topic", "").strip()
        if text:
            topics.append(text)
    return topics


def generate_simplified_explanation(context, weak_topics):
    """
    Generate a simpler explanation when user struggles with quiz.
    OPTIMIZATION:
    - Only called if user explicitly requests simpler explanation
    - Uses existing context (no additional retrieval)
    """
    weak_concepts = "\n".join([t.get("concept", "") for t in weak_topics])
    
    simplification_prompt = f"""
The user struggled with understanding these concepts from the material:
{weak_concepts}

Please provide a MUCH SIMPLER explanation using:
- Simpler vocabulary
- Real-world examples or analogies
- Step-by-step breakdowns
- Concrete examples from the context below

Original Context:
{context}

Generate a clear, beginner-friendly explanation:
"""

    response = st.session_state.conversation["llm"].invoke(simplification_prompt)
    return response.content


# ============================================================================
# QUIZ UI & RENDERING
# ============================================================================

def render_quiz_interface():
    """
    Display the quiz interface with 5 questions and radio buttons.
    OPTIMIZATION:
    - Uses cached quiz_data from session state
    - Does not regenerate quiz on reruns
    """
    if not st.session_state.quiz_data:
        return
    
    st.divider()
    st.subheader("📝 Test Your Understanding (5 Questions)")
    st.markdown("Answer all questions below and click **Submit Quiz** to see your results.")
    
    # Initialize answers if not already in session
    if "quiz_answers" not in st.session_state:
        st.session_state.quiz_answers = [None] * len(st.session_state.quiz_data)
    
    # Render each question with radio buttons
    for idx, question in enumerate(st.session_state.quiz_data):
        st.markdown(f"**Question {idx + 1}:** {question.get('question', '')}")
        
        # Radio buttons for options
        selected_option = st.radio(
            label=f"question_{idx}",
            options=question.get("options", []),
            label_visibility="collapsed",
            key=f"quiz_q{idx}"
        )
        
        # Store answer in session state
        if selected_option:
            st.session_state.quiz_answers[idx] = selected_option
        
        st.markdown("---")
    
    # Submit button
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("✅ Submit Quiz", key="submit_quiz_btn", use_container_width=True):
            # Check if all questions answered
            if None in st.session_state.quiz_answers:
                st.warning("⚠️ Please answer all questions before submitting.")
            else:
                st.session_state.quiz_submitted = True
                st.rerun()


def render_quiz_results():
    """
    Display quiz results, score, and weak topic analysis.
    OPTIMIZATION:
    - Only renders if quiz was submitted
    - Displays results once per submission
    """
    if not st.session_state.quiz_submitted or not st.session_state.quiz_data:
        return
    
    st.divider()
    st.subheader("📊 Quiz Results")
    
    # Calculate score
    score = 0
    for idx, question in enumerate(st.session_state.quiz_data):
        correct = question.get("correct_answer")
        if idx < len(st.session_state.quiz_answers) and st.session_state.quiz_answers[idx] == correct:
            score += 1
    
    total = len(st.session_state.quiz_data)
    
    # Display score with formatting
    score_color = "green" if score >= 3 else "orange" if score >= 2 else "red"
    st.markdown(f"### 🎯 Score: **{score}/{total}**")
    
    # Show incorrect answers with feedback
    incorrect_count = 0
    for idx, question in enumerate(st.session_state.quiz_data):
        correct = question.get("correct_answer")
        if idx < len(st.session_state.quiz_answers) and st.session_state.quiz_answers[idx] != correct:
            incorrect_count += 1
            st.markdown(f"**Question {idx + 1}:** {question.get('question', '')}")
            st.markdown(f"- Your answer: *{st.session_state.quiz_answers[idx]}*")
            st.markdown(f"- Correct answer: *{correct}*")
            if question.get("explanation"):
                st.markdown(f"- Explanation: {question.get('explanation')}")
            st.markdown("---")
    
    weak_topics = identify_weak_topics(st.session_state.quiz_data, st.session_state.quiz_answers)
    summary_topics = get_quiz_summary_topics(weak_topics)

    if summary_topics:
        st.markdown("**Quiz summary — topics to review:**")
        st.markdown(", ".join(summary_topics))

    if score < 3 and incorrect_count > 0:
        st.warning("💡 It looks like you're finding this topic challenging. Would you like a simpler explanation?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes, explain more simply", key="simplify_yes"):
                st.session_state.show_simplification = True
                if not st.session_state.simplified_explanation:
                    st.session_state.simplified_explanation = ""
                st.rerun()
        with col2:
            if st.button("No, thanks", key="simplify_no"):
                st.session_state.show_simplification = False
                st.rerun()
    
    if incorrect_count == 0:
        st.success(f"🎉 Perfect! You scored {score}/{total}. Great understanding!")
        if st.button("Clear quiz and continue", key="clear_quiz"):
            st.session_state.quiz_submitted = False
            st.session_state.quiz_data = None
            st.session_state.quiz_answers = []
            st.rerun()


def render_simplified_explanation_section():
    """
    Display simplified explanation if user requested it.
    OPTIMIZATION:
    - Only renders if user clicked "Yes" for simpler explanation
    - Generated once and cached in session state
    """
    if not st.session_state.show_simplification:
        return
    
    if not st.session_state.conversation:
        return
    
    st.divider()
    st.subheader("📚 Simplified Explanation")
    
    # Generate simplified explanation if not already cached
    if not st.session_state.simplified_explanation:
        weak_topics = identify_weak_topics(st.session_state.quiz_data, st.session_state.quiz_answers)
        with st.spinner("Generating simplified explanation..."):
            simplified = generate_simplified_explanation(st.session_state.last_context, weak_topics)
            st.session_state.simplified_explanation = simplified
    
    # Display explanation
    st.markdown(st.session_state.simplified_explanation)
    
    # Option to continue
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back to quiz results", key="back_to_results"):
            st.session_state.show_simplification = False
            st.rerun()
    with col2:
        if st.button("Return to chat", key="return_to_chat"):
            st.session_state.quiz_submitted = False
            st.session_state.quiz_data = None
            st.session_state.quiz_answers = []
            st.session_state.show_simplification = False
            st.session_state.simplified_explanation = ""
            st.rerun()


def render_quiz_section():
    """
    Main orchestrator for the entire quiz feature.
    OPTIMIZATION:
    - Only renders if chat history exists
    - Generates quiz only once per assistant response
    - Handles all quiz states: generation, answering, evaluation, simplification
    """
    if not st.session_state.chat_history:
        return
    
    # Show "Test My Understanding" button only if we have context from latest response
    if st.session_state.last_context and not st.session_state.quiz_data:
        st.divider()
        if st.button("🧠 Test My Understanding", key="test_understanding_btn", use_container_width=True):
            # Generate quiz only once per response
            with st.spinner("Generating quiz questions..."):
                quiz_data = generate_quiz_questions(st.session_state.last_context)
                if quiz_data:
                    st.session_state.quiz_data = quiz_data
                    st.session_state.quiz_answers = [None] * len(quiz_data)
                    st.session_state.quiz_submitted = False
                    st.rerun()
    
    # Display quiz interface if questions exist
    if st.session_state.quiz_data and not st.session_state.quiz_submitted:
        render_quiz_interface()
    
    # Display results if submitted
    if st.session_state.quiz_submitted:
        render_quiz_results()
    
    # Display simplified explanation if requested
    if st.session_state.show_simplification:
        render_simplified_explanation_section()


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
    
    # OPTIMIZATION: Quiz feature session state
    if "quiz_data" not in st.session_state:
        st.session_state.quiz_data = None
    if "quiz_answers" not in st.session_state:
        st.session_state.quiz_answers = []
    if "quiz_submitted" not in st.session_state:
        st.session_state.quiz_submitted = False
    if "show_simplification" not in st.session_state:
        st.session_state.show_simplification = False
    if "simplified_explanation" not in st.session_state:
        st.session_state.simplified_explanation = ""
    if "latest_response_id" not in st.session_state:
        st.session_state.latest_response_id = None


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
        pass  # Quiz moved to separate section below

    # Render quiz section (5-question feature)
    render_quiz_section()

    # Render sidebar
    render_sidebar()


if __name__ == '__main__':
    main()