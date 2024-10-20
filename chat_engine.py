import os
from dotenv import load_dotenv
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.core import ChatPromptTemplate
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.prompts.prompt_type import PromptType

class ChatEngine:
    def __init__(self):
        # Load environment variables
        load_dotenv()

        # Set OpenAI API key
        os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

        # Initialize Pinecone
        api_key = os.getenv("PINECONE_API_KEY")
        pc = Pinecone(api_key=api_key)

        # Connect to existing Pinecone index
        index_name = "langchainvector"
        pinecone_index = pc.Index(index_name)

        # Create vector store
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

        # Create storage context
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )

        # Specify the OpenAI model
        llm = OpenAI(model="gpt-4o", temperature=0.7)

        # Define custom chat prompt template
        custom_prompt = ChatPromptTemplate(
            message_templates=[
                ChatMessage(
                    role="system",
                    content="""You are an AI assistant designed to provide accurate, detailed, and comprehensive information about Prophet Muhammad (Peace Be Upon Him) and Islam. Your responses should be based on authentic Islamic sources and the data provided through the vector database embeddings. Please follow these guidelines:

                    1. Respond in the same language as the user's query. The query language is: {query_language}
                    2. Provide extremely detailed information with multiple references from authentic Islamic sources.
                    3. When addressing topics related to Qadiyani groups, provide a comprehensive explanation of their beliefs, the Islamic perspective on these beliefs, and why they are considered problematic by mainstream Islamic scholars.
                    4. Emphasize the authenticity and finality of Prophet Muhammad's (PBUH) prophethood with detailed arguments and evidence.
                    5. Be respectful and objective in your responses, avoiding any form of disrespect towards other beliefs.
                    6. If asked about topics not related to Islam or Prophet Muhammad (PBUH), politely redirect the conversation to the intended subject matter.
                    7. If you're unsure about any information, state that clearly rather than making assumptions.
                    8. Provide comprehensive responses, elaborating on key points and providing historical, theological, and scholarly context where necessary.
                    9. Utilize the information from the vector database to enrich your responses with relevant facts, dates, names, and details.
                    10. Always include multiple references to Quranic verses, Hadith, and respected Islamic scholars in your responses.

                    Context information from the vector database is below.
                    ---------------------
                    {context_str}
                    ---------------------
                    Given this information, please provide a detailed, comprehensive answer to the user's question, ensuring to respond in {query_language}."""
                ),
                ChatMessage(
                    role="user",
                    content="{query_str}"
                ),
                ChatMessage(
                    role="assistant",
                    content="Based on the information available in our database and authentic Islamic sources, I can provide the following detailed response: {response}"
                )
            ],
            prompt_type=PromptType.CUSTOM
        )

        # Load the index (this doesn't redo the embedding)
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            llm=llm
        )

        # Create chat engine with custom prompt
        self.chat_engine = CondenseQuestionChatEngine.from_defaults(
            query_engine=index.as_query_engine(),
            llm=llm,
            verbose=True,
            chat_prompt=custom_prompt
        )

    def get_response(self, prompt):
        response = self.chat_engine.stream_chat(prompt)
        return response.response_gen