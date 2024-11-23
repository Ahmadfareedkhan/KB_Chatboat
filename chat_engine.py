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
from langdetect import detect

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

        # Load the index
        self.index = VectorStoreIndex.from_vector_store(
            vector_store
        )

    def detect_language(self, text):
        try:
            lang = detect(text)
            if lang == 'ar':
                return {
                    'code': 'ar',
                    'name': 'Arabic',
                    'system_msg': """أنت مساعد متخصص في موضوع ختم النبوة. يجب أن تجيب باللغة العربية فقط.
                    
المبادئ التوجيهية للإجابة:
1. استخدم اللغة العربية الفصحى
2. قدم الإجابات مع الأدلة من القرآن والسنة
3. اذكر المراجع والمصادر الموثوقة
4. كن دقيقاً في النقل والاقتباس

المعلومات المتوفرة من قاعدة البيانات:
{context_str}""",
                    'prefix': 'الجواب: '
                }
            elif lang in ['ur', 'hi']:
                return {
                    'code': 'ur',
                    'name': 'Urdu',
                    'system_msg': """آپ ختم نبوت کے موضوع پر ماہر معاون ہیں۔ آپ کو صرف اردو میں جواب دینا ہے۔

جواب دینے کے اصول:
1. معیاری اردو کا استعمال کریں
2. قرآن و سنت سے دلائل فراہم کریں
3. معتبر حوالہ جات کا ذکر کریں
4. اقتباسات میں دقت کا خیال رکھیں

ڈیٹا بیس سے دستیاب معلومات:
{context_str}""",
                    'prefix': 'جواب: '
                }
            else:
                return {
                    'code': 'en',
                    'name': 'English',
                    'system_msg': """You are an expert assistant on the topic of the Finality of Prophethood. You must respond in English only.

Response guidelines:
1. Use formal English
2. Provide evidence from Quran and Sunnah
3. Cite reliable sources
4. Be precise in quotations

Information available from the database:
{context_str}""",
                    'prefix': 'Response: '
                }
        except:
            return {
                'code': 'en',
                'name': 'English',
                'system_msg': "Default English response...",
                'prefix': 'Response: '
            }

    def get_response(self, prompt):
        # Detect the language of the prompt
        lang_info = self.detect_language(prompt)
        
        # Set model parameters based on language
        self.llm = OpenAI(
            model="gpt-4o",
            temperature=0.7,
            system_prompt=lang_info['system_msg']
        )

        # Create custom chat prompt template
        custom_prompt = ChatPromptTemplate(
            message_templates=[
                ChatMessage(
                    role="system",
                    content=lang_info['system_msg']
                ),
                ChatMessage(
                    role="user",
                    content=prompt
                ),
                ChatMessage(
                    role="assistant",
                    content=f"{lang_info['prefix']}{{response}}"
                )
            ],
            prompt_type=PromptType.CUSTOM
        )

        # Create query engine with language-specific settings
        query_engine = self.index.as_query_engine(
            llm=self.llm,
            similarity_top_k=5,
            response_mode="compact"
        )

        # Create chat engine with the updated prompt
        chat_engine = CondenseQuestionChatEngine.from_defaults(
            query_engine=query_engine,
            llm=self.llm,
            verbose=True,
            chat_prompt=custom_prompt
        )
        
        # Get response
        response = chat_engine.stream_chat(prompt)
        return response.response_gen