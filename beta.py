import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from key import cohere_api_key
from langchain_community.chat_models import ChatCohere

st.set_page_config(page_title='Chat with MySQL', page_icon=':speech_balloon:')
st.title('Chat with MySQL')

# Function to connect to the database
def connect(root: str, host: str, name: str) -> SQLDatabase:
    db_url = f"mysql+mysqlconnector://{root}:@{host}/{name}"
    return SQLDatabase.from_uri(db_url)

# Function to get SQL chain
def get_sql_chain(db):
    # Load Model
    class CustomChatCohere(ChatCohere):
        def _get_generation_info(self, response):
            # Custom handling of generation info
            generation_info = {}
            if hasattr(response, 'token_count'):
                generation_info["token_count"] = response.token_count
            # Add other attributes if needed
            return generation_info

    llm = CustomChatCohere(cohere_api_key=cohere_api_key)
    # Prompt Template
    template = """
    Based on the table schema below, write only a SQL query that would answer the user's question and your answer should be in text without putting it in SQL editor: 
    {schema}

    Question: {question}
    {chat_history}
    """
    prompt = ChatPromptTemplate.from_template(template)

    def get_schema(_):
        return db.get_table_info()

    # Chains
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)
    template2 = """
    Based on the table schema below, question, sql query, and sql response, write a natural language response:
    Here are a few things to take note of:
    No.1. If the SQL query is incorrect or in a format that is incorrect, reply by saying "Could not get any 
    information at this time, can you please ask again?"
    No.2. If the user question is not related to what is in the database, respond by saying "Your question
    is unrelated to the information on the database. After this, then go global and answer the question."
    {schema}

    {chat_history}
    Question: {question}
    SQL Query: {query}
    SQL Response: {response}
    """
    prompt2 = ChatPromptTemplate.from_template(template2)

    class CustomChatCohere(ChatCohere):
        def _get_generation_info(self, response):
            # Custom handling of generation info
            generation_info = {}
            if hasattr(response, 'token_count'):
                generation_info["token_count"] = response.token_count
            # Add other attributes if needed
            return generation_info

    llm = CustomChatCohere(cohere_api_key=cohere_api_key)

    def get_schema(_):
        return db.get_table_info()

    def run_query(query):
        try:
            return db.run(query)
        except Exception as e:
            return None

    response_chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=get_schema,
            response=lambda variables: run_query(variables['query'])
        )
        | ChatPromptTemplate.from_template(template2)
        | llm
        | StrOutputParser()
    )

    # First, try to generate a response based on the database
    response = response_chain.invoke({
        'question': user_query,
        'chat_history': chat_history
    })

    # Check if the response indicates the question is unrelated to the database
    if "Your question is unrelated to the information on the database." in response:
        # Generate a global response
        global_prompt_template = """
        The user has asked a question: {question}
        Provide a comprehensive answer.
        """
        global_prompt = ChatPromptTemplate.from_template(global_prompt_template)
        global_response_chain = (
            RunnablePassthrough.assign(question=user_query)
            | global_prompt
            | llm
            | StrOutputParser()
        )
        global_response = global_response_chain.invoke({
            'question': user_query
        })
        return global_response

    return response

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I am an SQL assistant. Ask me anything about your database.")
    ]

# Sidebar for database connection
with st.sidebar:
    st.subheader('Enter your database details')
    root = st.text_input('Root', value='root', key='root')
    host = st.text_input('Host', value='localhost', key='host')
    name = st.text_input('Name', value='sample', key='name')
    if st.button('Connect'):
        with st.spinner('Connecting to Database...'):
            try:
                db = connect(root, host, name)
                st.session_state.db = db
                st.success('Connected Successfully')
            except Exception as e:
                st.error(f"Connection failed: {e}")

# Display chat messages
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

# User query input
user_query = st.chat_input('Type a message...')
if user_query and user_query.strip():
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        if 'db' in st.session_state:
            response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
            st.markdown(response)
            st.session_state.chat_history.append(AIMessage(content=response))
        else:
            response = "Please connect to the database first."
            st.markdown(response)
            st.session_state.chat_history.append(AIMessage(content=response))

