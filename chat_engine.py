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
            # First, use langdetect as a starting point
            lang = detect(text)
            
            # Additional heuristics to distinguish Arabic from Urdu
            def is_likely_arabic(text):
                # Arabic-specific characters that are rare in Urdu
                arabic_chars = ['ض', 'ص', 'ث', 'ق', 'ف', 'غ', 'ع', 'ه', 'خ', 'ح', 'ج', 'ش', 'س', 'ي', 'ب', 'ل', 'ا', 'ت', 'ن', 'م', 'ك', 'ط', 'ذ', 'د', 'ز', 'ر', 'و']
                
                # Urdu-specific characters that are rare in Arabic
                urdu_chars = ['ں', 'ے', 'ھ', 'ڈ', 'ٹ', 'ڑ', 'چ', 'پ', 'ژ', 'گ', 'ڇ', 'ٹھ', 'ڈھ', 'ڑھ', 'کھ', 'گھ', 'نھ', 'مھ', 'بھ', 'پھ', 'تھ', 'ٹھ', 'جھ', 'چھ', 'دھ', 'ڈھ', 'رھ', 'ڑھ', 'سھ', 'شھ', 'کھ', 'گھ', 'لھ', 'مھ', 'نھ', 'ھ']
                
                # Common Arabic words
                arabic_words = ['الله', 'النبي', 'القرآن', 'الإسلام', 'المسلمين', 'الرسول', 'عليه', 'وسلم', 'صلى', 'تعالى', 'سبحانه', 'الحمد', 'رب', 'العالمين']
                
                # Common Urdu words
                urdu_words = ['اللہ', 'نبی', 'قرآن', 'اسلام', 'مسلمان', 'رسول', 'علیہ', 'وسلم', 'صلی', 'تعالیٰ', 'سبحانہ', 'الحمد', 'رب', 'العالمین', 'ہے', 'کا', 'کے', 'کی', 'میں', 'سے', 'کو', 'نے', 'کر', 'اور', 'یہ', 'وہ', 'اس', 'کہ', 'جو', 'کون', 'کیا', 'کیوں', 'کہاں', 'کب', 'کیسے']
                
                # Count Arabic vs Urdu specific characters
                arabic_char_count = sum(1 for char in text if char in arabic_chars)
                urdu_char_count = sum(1 for char in text if char in urdu_chars)
                
                # Count Arabic vs Urdu specific words
                arabic_word_count = sum(1 for word in arabic_words if word in text)
                urdu_word_count = sum(1 for word in urdu_words if word in text)
                
                # Calculate scores
                arabic_score = arabic_char_count + (arabic_word_count * 3)
                urdu_score = urdu_char_count + (urdu_word_count * 3)
                
                # If there are clear Urdu-specific characters, it's likely Urdu
                if urdu_char_count > 0:
                    return False
                
                # If there are Arabic-specific words or characters, it's likely Arabic
                if arabic_score > urdu_score:
                    return True
                
                # If the original detection was Arabic, trust it unless there's strong evidence otherwise
                if lang == 'ar':
                    return True
                
                return False
            
            # Check if text contains Arabic or Urdu script
            has_arabic_script = any('\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F' for c in text)
            
            if has_arabic_script:
                if is_likely_arabic(text):
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
                else:
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