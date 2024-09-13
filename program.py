from langchain_community.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
import streamlit as st
from genai.extensions.langchain import LangChainChatInterface
LLm = OllamaLLM(model="gemma2:2b")
@st.cache_resource
def load_pdf():
    pdf_name = 'Doc.pdf'
    
    # Load the PDF
    loader = PyPDFLoader(pdf_name)
    documents = loader.load()

    # Set up text splitter and embedding model
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L12-v2')
    
    # Use FAISS (or another vector store)
    vectorstore = FAISS.from_documents(splitted_docs, embedding=embeddings)
    
    return vectorstore

# Load the vector store index
index = load_pdf()

# Create the RetrievalQA chain
chain = RetrievalQA.from_chain_type(
    llm=LLm,
    chain_type="stuff",  # "stuff", "map_reduce", or "refine" depending on your use case
    retriever=index.as_retriever()
)
st.title('NICSI ChatBot')
if 'messages' not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

prompt = st.chat_input('Pass your Prompt here')
if prompt:

    st.chat_message('User').markdown(prompt)

    st.session_state.messages.append({'role':'user','content':prompt})
    
    response = chain.run(prompt)

    st.chat_message('assistant').markdown(response)
    st.session_state.messages.append({'role':'assistant','content':response})