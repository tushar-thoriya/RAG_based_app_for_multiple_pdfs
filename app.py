import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAI
from langchain_community.embeddings.huggingface import HuggingFaceInstructEmbeddings, HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
# from langchain_community import HuggingFaceHub
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
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
    # embeddings = OpenAIEmbeddings()
    # embeddings = OpenAIEmbeddings(embedding_ctx_length=300000)
    embeddings = HuggingFaceInstructEmbeddings(model_name="WhereIsAI/UAE-Large-V1")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl",task="text-generation", model_kwargs={"temperature":0.5, "max_length":512})
    # llm = HuggingFaceEndpoint(repo_id="google/flan-t5-xxl",task="text-generation", model_kwargs={"temperature":0.5, "max_length":512})
    
    from huggingface_hub import hf_hub_download
    # llm = hf_hub_download(repo_id="google-t5/t5-small", filename="config.json")
    # Load model directly
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    # tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
    # llm = AutoModelForSeq2SeqLM.from_pretrained("roborovski/superprompt-v1")

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory
# # from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from transformers import T5Tokenizer, T5ForConditionalGeneration

# def get_conversation_chain(vectorstore):
#     # Load the tokenizer and LLM
#     tokenizer = T5Tokenizer.from_pretrained("t5-small")
#     model = T5ForConditionalGeneration.from_pretrained("t5-small")

#     # Initialize conversation memory
#     memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    
#     # Create the conversation chain
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=model,
#         retriever=vectorstore.as_retriever(),
#         memory=memory
#     )
#     return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    # os.environ['OPENAI_API_KEY'] = apikey
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()
