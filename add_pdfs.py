import os
from dotenv import load_dotenv
from pinecone import Pinecone
from llama_index.core import VectorStoreIndex, Document
from llama_index.core import StorageContext
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pathlib import Path
import fitz
import time
from typing import List

# Load environment variables
load_dotenv()

def test_pinecone_connection():
    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        # List available indexes
        indexes = pc.list_indexes()
        print("Available Pinecone indexes:", [index.name for index in indexes])
        return pc
    except Exception as e:
        print(f"Error connecting to Pinecone: {e}")
        return None

def extract_pdf_content(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return None

def get_pdf_files_from_directories(directories: List[str]) -> List[Path]:
    """Get all PDF files from multiple directories"""
    all_pdf_files = []
    for directory in directories:
        pdf_directory = Path(directory)
        if pdf_directory.exists():
            pdf_files = list(pdf_directory.glob('*.pdf'))
            all_pdf_files.extend(pdf_files)
            print(f"Found {len(pdf_files)} PDF files in {directory}")
        else:
            print(f"Directory {directory} does not exist, skipping...")
    return all_pdf_files

def get_existing_documents_from_index(pinecone_index):
    """Get list of existing document metadata to avoid duplicates"""
    try:
        stats = pinecone_index.describe_index_stats()
        print(f"Index currently contains {stats.total_vector_count} vectors")
        return stats.total_vector_count
    except Exception as e:
        print(f"Error getting index stats: {e}")
        return 0

def main():
    # Test Pinecone connection first
    print("Testing Pinecone connection...")
    pc = test_pinecone_connection()
    if not pc:
        print("Failed to connect to Pinecone. Please check your internet connection and API key.")
        return

    # Define directories to process
    # You can modify this list to include any directories you want to process
    directories_to_process = [
        # "data",        # Original data directory
        "new_data",    # New data directory
        # "new_data_2",  # Uncomment if you have this directory
        # Add more directories as needed
    ]
    
    # Get all PDF files from specified directories
    pdf_files = get_pdf_files_from_directories(directories_to_process)
    
    if not pdf_files:
        print("No PDF files found in any of the specified directories")
        return

    print(f"\nTotal PDF files found: {len(pdf_files)}")
    for pdf in pdf_files:
        print(f"- {pdf.parent.name}/{pdf.name}")

    # Connect to existing index
    try:
        index_name = "langchainvector"
        pinecone_index = pc.Index(index_name)
        
        # Get existing index stats
        existing_count = get_existing_documents_from_index(pinecone_index)
        
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

    except Exception as e:
        print(f"Error connecting to Pinecone index: {e}")
        return

    # Process PDFs in batches to avoid memory issues
    batch_size = 5  # Process 5 PDFs at a time
    total_processed = 0
    
    for i in range(0, len(pdf_files), batch_size):
        batch = pdf_files[i:i + batch_size]
        print(f"\nProcessing batch {i//batch_size + 1} ({len(batch)} files)...")
        
        documents = []
        for pdf_path in batch:
            print(f"Processing {pdf_path.parent.name}/{pdf_path.name}...")
            content = extract_pdf_content(pdf_path)
            if content:
                doc = Document(
                    text=content,
                    metadata={
                        "filename": pdf_path.name,
                        "source_directory": pdf_path.parent.name,
                        "file_path": str(pdf_path)
                    }
                )
                documents.append(doc)

        if not documents:
            print(f"No documents were processed successfully in this batch.")
            continue

        print(f"Successfully processed {len(documents)} documents in this batch")

        # Add documents to existing index using an Ingestion Pipeline for efficiency
        print("Adding documents to existing index...")

        try:
            # Setup an ingestion pipeline
            pipeline = IngestionPipeline(
                transformations=[
                    SentenceSplitter(chunk_size=1024, chunk_overlap=20),
                    OpenAIEmbedding(model="text-embedding-3-small"),
                ],
                vector_store=vector_store,
            )
            
            # Run the pipeline
            pipeline.run(documents=documents, show_progress=True)
            
            total_processed += len(documents)
            print(f"Successfully added {len(documents)} documents to the index.")

        except Exception as e:
            print(f"Failed to add documents to the index: {e}")
            continue

    print(f"\n=== SUMMARY ===")
    print(f"Total documents processed and added: {total_processed}")
    print(f"Previous index size: {existing_count}")
    
    # Get final stats
    try:
        final_stats = pinecone_index.describe_index_stats()
        print(f"Final index size: {final_stats.total_vector_count}")
        print(f"New documents added: {final_stats.total_vector_count - existing_count}")
    except Exception as e:
        print(f"Could not get final stats: {e}")
    
    print("Done! Your new documents have been successfully added to the existing index.")

if __name__ == "__main__":
    print("Starting document indexing process for new PDFs...")
    main()