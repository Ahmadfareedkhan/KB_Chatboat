import streamlit as st
from chat_engine import ChatEngine

# Streamlit UI
st.set_page_config(page_title="AI Chat Assistant", page_icon="🤖")

# CSS for RTL support and Arabic/Urdu text
st.markdown("""
<style>
    .stApp {
        max-width: 800px;
        margin: 0 auto;
    }
    .rtl-text {
        direction: rtl;
        text-align: right;
        font-family: 'Noto Naskh Arabic', 'Nastaliq', serif;
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
        # Check if the message contains Arabic or Urdu script
        if any('\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F' for c in message["content"]):
            st.markdown(f'<div class="rtl-text">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.write(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message with RTL support if needed
    with st.chat_message("user"):
        if any('\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F' for c in prompt):
            st.markdown(f'<div class="rtl-text">{prompt}</div>', unsafe_allow_html=True)
        else:
            st.write(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Show loading spinner while initializing chat engine
        with st.spinner("Initializing AI... This may take a moment."):
            chat_engine = get_chat_engine()
        
        # Stream the response
        for response_chunk in chat_engine.get_response(prompt):
            full_response += response_chunk
            # Check if response contains Arabic/Urdu script
            if any('\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F' for c in full_response):
                message_placeholder.markdown(
                    f'<div class="rtl-text">{full_response}{"▌"}</div>',
                    unsafe_allow_html=True
                )
            else:
                message_placeholder.write(full_response + "▌")
        
        # Final response
        if any('\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F' for c in full_response):
            message_placeholder.markdown(
                f'<div class="rtl-text">{full_response}</div>',
                unsafe_allow_html=True
            )
        else:
            message_placeholder.write(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Footer
st.markdown("---")
st.caption("Powered by LlamaIndex and Streamlit")