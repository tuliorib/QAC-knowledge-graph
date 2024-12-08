from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, Dict, Protocol, runtime_checkable

import pymupdf  # PyMuPDF
from bs4 import BeautifulSoup
from docx import Document
import re


@runtime_checkable
class DocumentProcessor(Protocol):
    """Protocol defining the interface for document processors."""

    @abc.abstractmethod
    def read_document(self, file_path: Path) -> str:
        """Read and return the content of a document."""
        ...

    @abc.abstractmethod
    def clean_text(self, text: str) -> str:
        """Clean the extracted text."""
        ...


class BaseDocumentProcessor(DocumentProcessor):
    """Base class for document processors with common functionality."""

    def clean_text(self, text: str) -> str:
        """Clean the extracted text by removing extra whitespace and normalizing characters."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Normalize unicode characters
        text = text.encode('ascii', 'ignore').decode('ascii')
        return text.strip()


class PDFProcessor(BaseDocumentProcessor):
    """Processor for PDF documents."""

    def read_document(self, file_path: Path) -> str:
        """Read and return the content of a PDF document."""
        with pymupdf.open(file_path) as doc:
            text = "".join(page.get_text() for page in doc)
        return self.clean_text(text)


class HTMLProcessor(BaseDocumentProcessor):
    """Processor for HTML documents."""

    def read_document(self, file_path: Path) -> str:
        """Read and return the content of an HTML document."""
        with open(file_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'html.parser')
            text = soup.get_text(separator=' ')
        return self.clean_text(text)


class DOCXProcessor(BaseDocumentProcessor):
    """Processor for DOCX documents."""

    def read_document(self, file_path: Path) -> str:
        """Read and return the content of a DOCX document."""
        doc = Document(file_path)
        text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
        return self.clean_text(text)


class PlainTextProcessor(BaseDocumentProcessor):
    """Processor for plain text documents."""

    def read_document(self, file_path: Path) -> str:
        """Read and return the content of a plain text document."""
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return self.clean_text(text)


def get_document_processor(file_path: Path) -> DocumentProcessor:
    """Factory function to get the appropriate document processor based on file extension."""
    match file_path.suffix.lower():
        case ".pdf":
            return PDFProcessor()
        case ".html" | ".htm":
            return HTMLProcessor()
        case ".docx":
            return DOCXProcessor()
        case ".txt":
            return PlainTextProcessor()
        case _:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")


def process_document(file_path: Path) -> Dict[str, Any]:
    """Process a document and return its content and metadata."""
    processor = get_document_processor(file_path)
    content = processor.read_document(file_path)
    
    # Here you could add more metadata extraction if needed
    metadata = {
        "file_name": file_path.name,
        "file_type": file_path.suffix,
        "file_size": file_path.stat().st_size,
    }
    
    return {"content": content, "metadata": metadata}


if __name__ == "__main__":
    # Example usage
    file_path = Path("./data/raw/pymu.html")
    result = process_document(file_path)
    print(f"Processed {result['metadata']['file_name']}:")
    print(f"Content preview: {result['content'][:100]}...")
    print(f"Metadata: {result['metadata']}")