import streamlit as st
from chat_engine import ChatEngine

# Streamlit UI
st.set_page_config(page_title="AI Chat Assistant", page_icon="🤖")

# Simple CSS for a cleaner look
st.markdown("""
<style>
    .stApp {
        max-width: 800px;
        margin: 0 auto;
    }
    .stTextInput > div > div > input {
        background-color: #f0f2f6;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("AI Chat Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Lazy loading of ChatEngine with caching
@st.cache_resource
def get_chat_engine():
    return ChatEngine()

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Show loading spinner while initializing chat engine
        with st.spinner("Initializing AI... This may take a moment."):
            chat_engine = get_chat_engine()
        
        for response_chunk in chat_engine.get_response(prompt):
            full_response += response_chunk
            message_placeholder.write(full_response + "▌")
        message_placeholder.write(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Simple footer
st.markdown("---")
st.caption("Powered by LlamaIndex and Streamlit")