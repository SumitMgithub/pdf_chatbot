import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from streamlit_chat import message
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader

st.title('LLM Custom Chatbot💬')

with st.sidebar:
    
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model

    ''')
    
    st.write('Made by Sumit Malviya')


def process_pdf(file_path):
    pdf_reader = PdfReader(file_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# Enable the user to upload a file using Streamlit file uploader
uploaded_file = st.sidebar.file_uploader("upload", type="pdf")

if uploaded_file:

    text = process_pdf(uploaded_file)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    # Initialize embeddings, vectorstore, and chain for conversational retrieval
    # vectorstore is used to retrieve relevant information based on the user's input.

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    chain = ConversationalRetrievalChain.from_llm(llm=ChatOpenAI(
        temperature=0.0,
        model_name='gpt-3.5-turbo'),
        retriever=vectorstore.as_retriever())


    # Creating a function that accepts user query, feeds it into ConversationalRetrievalChain instance (i.e. chain), and returns the generated response
    # It also keeps track of the conversation history, which is useful for creating more contextual responses.

    def conversational_chat(query):
        result = chain({"question": query, "chat_history": st.session_state['request']})
        st.session_state['request'].append((query, result["answer"]))
        return result["answer"]


    # Check and initialize the session state variables if not present.
    # This will display the initial messages in the chat.
    # If the state history is empty, the chatbot will prompt a “Hello” message and reply with “Hello! Ask me about file uploaded.”
    # Otherwise, the chatbot resumes from the last conversations.

    if 'request' not in st.session_state:
        st.session_state['request'] = []

    if 'response' not in st.session_state:
        st.session_state['response'] = ["Hello ! Ask me about " + uploaded_file.name + " 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]

    # A Streamlit container is a feature that allows the grouping of multiple elements in a Streamlit app.
    # Defining two containers using st.container().

    # 1. container for the chat history
    response_container = st.container()

    # 2. container for the user's input
    input_container = st.container()

    with input_container:

        # with st.form(key='my_form', clear_on_submit=True):
            # Capture user's input via Streamlit text_input
        user_input = st.text_input("Query:", placeholder="Talk about your pdf data here: ", key='input')

        submit_button = st.form_submit_button(label='Ask')

        if submit_button and user_input:
            # Call the conversational_chat function with user input and retrieve output
            output = conversational_chat(user_input)

            # Update session state with user input and generated output
            st.session_state['past'].append(user_input)
            st.session_state['response'].append(output)

    if st.session_state['response']:

        # Display chat history in the response container
        with response_container:

            for i in range(len(st.session_state['response'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["response"][i], key=str(i), avatar_style="thumbs")
