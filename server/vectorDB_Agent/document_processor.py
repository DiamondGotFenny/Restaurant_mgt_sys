# document_processor.py

import os
import glob
from typing import List
from langchain.schema import Document
from langchain.text_splitter import NLTKTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from  logger_config import setup_logger
import nltk

# Ensure the necessary NLTK data is downloaded
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

class DocumentProcessor:
    def __init__(
        self,
        pdf_directory: str,
        chunk_size: int = 2000,
        chunk_overlap: int = 400,
        logger_file: str = "document_processor.log"
    ):
        """
        Initializes the DocumentProcessor with NLTKTextSplitter.

        Args:
            pdf_directory (str): Path to the directory containing PDF files.
            chunk_size (int): Maximum characters per chunk.
            chunk_overlap (int): Overlap to maintain context between chunks.
            logger_file (str): Path to the log file.
        """
        self.logger = setup_logger(logger_file)
        self.pdf_directory = pdf_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize NLTKTextSplitter
        self.text_splitter = NLTKTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator="\n",
            length_function=len
        )

    def load_and_split_documents(self) -> List[Document]:
        """
        Loads and splits all PDF documents from the specified directory.

        Returns:
            List[Document]: A list of split and annotated Document objects.
        """
        pdf_files = glob.glob(os.path.join(self.pdf_directory, "*.pdf"))
        if not pdf_files:
            self.logger.warning(f"No PDF files found in directory: {self.pdf_directory}")
            return []

        documents = []
        for pdf_file in pdf_files:
            self.logger.info(f"Loading PDF file: {pdf_file}")
            try:
                loader = PyPDFLoader(pdf_file)
                raw_docs = loader.load()
            except Exception as e:
                self.logger.error(f"Failed to load {pdf_file}: {e}")
                continue

            # Split documents
            split_docs = self.text_splitter.split_documents(raw_docs)
            self.logger.info(f"Loaded and split {len(split_docs)} documents from {pdf_file}.")

            # Annotate documents with metadata
            for idx, doc in enumerate(split_docs, start=1):
                annotated_doc = Document(
                    page_content=doc.page_content,
                    metadata={
                        "source": os.path.basename(pdf_file),
                        "page": idx  # Assuming each split corresponds to a page
                    }
                )
                documents.append(annotated_doc)

        self.logger.info(f"Total documents loaded and split: {len(documents)}")
        return documents