import glob
import os
from collections import defaultdict
from typing import List

from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..logger_config import setup_logger

class DocumentProcessor:
    def __init__(
        self,
        pdf_directory: str,
        chunk_size: int = 2000,
        chunk_overlap: int = 400,
        logger_file: str = "document_processor.log"
    ):
        """
        Initializes the DocumentProcessor with a deterministic splitter (no runtime downloads).

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

        # RecursiveCharacterTextSplitter avoids NLTK download side effects at import/runtime.
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True,
            separators=["\n\n", "\n", " ", ""],
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

        documents: List[Document] = []
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

            # Annotate documents with stable, trustworthy metadata.
            # Keep the real PDF page number from PyPDFLoader (0-indexed) and add a chunk id.
            chunk_counters: defaultdict[tuple[str, int | str], int] = defaultdict(int)
            source_name = os.path.basename(pdf_file)
            for doc in split_docs:
                raw_page = doc.metadata.get("page", "N/A")
                page = raw_page + 1 if isinstance(raw_page, int) else raw_page

                key = (source_name, page)
                chunk_counters[key] += 1
                chunk_index = chunk_counters[key]

                doc.metadata = {
                    **doc.metadata,
                    "source": source_name,
                    "page": page,
                    "chunk_index": chunk_index,
                    "chunk_id": f"{source_name}#p{page}c{chunk_index}",
                }
                documents.append(doc)

        self.logger.info(f"Total documents loaded and split: {len(documents)}")
        return documents
