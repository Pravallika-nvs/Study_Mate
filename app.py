import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from htmlTemplates import bot_template, user_template, css

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    return {
        "llm": llm,
        "retriever": vectorstore.as_retriever()
    }


def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("Please upload and process PDFs first.")
        return
    
    docs = st.session_state.conversation["retriever"].invoke(user_question)

    context = "\n".join([doc.page_content for doc in docs])

    history = "\n".join([
        f"{msg['role']}: {msg['content']}"
        for msg in st.session_state.chat_history
    ])

    prompt = f"""
You are Study-Mate, an AI assistant that answers questions using the uploaded PDFs.

Instructions:
- Answer using the provided context.
- Use the conversation history when needed.
- If the answer is not found in the context, say "I couldn't find that information in the uploaded documents."
- Be concise and accurate.

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

    # Save conversation
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_question
    })

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer
    })

    # Display conversation
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.write(
                user_template.replace("{{MSG}}", msg["content"]),
                unsafe_allow_html=True
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", msg["content"]),
                unsafe_allow_html=True
            )


def main():
    load_dotenv()
    st.set_page_config(page_title = "Study-Mate: Your AI-Powered PDFs Tool", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Study-Mate: Your AI-Powered PDFs Tool :books:")
    user_question = st.text_input("Ask a question about your document(s):")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get the pdf text
                raw_text = get_pdf_text(pdf_docs)

                if not raw_text.strip():
                    st.error("No readable text found in the uploaded PDFs.")
                    return

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("Documents processed successfully!")


if __name__ == '__main__':
    main()