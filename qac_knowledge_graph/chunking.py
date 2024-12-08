from __future__ import annotations

import abc
from typing import List, Protocol, runtime_checkable, Optional
import re
from dataclasses import dataclass, field
import nltk
from nltk.tokenize import sent_tokenize
import ast

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet = True)

@runtime_checkable
class ChunkingStrategy(Protocol):
    """Protocol defining the interface for chunking strategies."""

    @abc.abstractmethod
    def create_chunks(self, content: str, chunk_size: int) -> List[str]:
        """Create chunks from the given content."""
        ...

@dataclass
class Chunk:
    """Represents a chunk of content with metadata."""
    content: str
    start_index: int
    end_index: int
    metadata: dict = field(default_factory=dict)

class BaseChunkingStrategy(ChunkingStrategy):
    """Base class for chunking strategies with common functionality."""

    def clean_text(self, text: str) -> str:
        """Clean the text by removing extra whitespace."""
        return re.sub(r'\s+', ' ', text).strip()

class TextChunkingStrategy(BaseChunkingStrategy):
    """Chunking strategy for regular text documents."""

    def create_chunks(self, content: str, chunk_size: int) -> List[Chunk]:
        cleaned_content = self.clean_text(content)
        sentences = sent_tokenize(cleaned_content)
        chunks = []
        current_chunk = []
        current_size = 0
        start_index = 0

        for sentence in sentences:
            sentence_size = len(sentence.split())
            if current_size + sentence_size > chunk_size and current_chunk:
                chunk_content = ' '.join(current_chunk)
                chunks.append(Chunk(chunk_content, start_index, start_index + len(chunk_content)))
                current_chunk = []
                current_size = 0
                start_index += len(chunk_content) + 1  # +1 for the space

            current_chunk.append(sentence)
            current_size += sentence_size

        if current_chunk:
            chunk_content = ' '.join(current_chunk)
            chunks.append(Chunk(chunk_content, start_index, start_index + len(chunk_content)))

        return chunks

class CodeChunkingStrategy(BaseChunkingStrategy):
    """Chunking strategy for code files."""

    def create_chunks(self, content: str, chunk_size: int) -> List[Chunk]:
        try:
            tree = ast.parse(content)
        except SyntaxError:
            # If parsing fails, fall back to line-based chunking
            return self._line_based_chunking(content, chunk_size)

        chunks = []
        current_chunk = []
        current_size = 0
        start_index = 0

        for node in ast.iter_child_nodes(tree):
            node_str = ast.get_source_segment(content, node)
            if node_str is None:
                continue
            node_size = len(node_str.split())

            if current_size + node_size > chunk_size and current_chunk:
                chunk_content = '\n'.join(current_chunk)
                chunks.append(Chunk(chunk_content, start_index, start_index + len(chunk_content)))
                current_chunk = []
                current_size = 0
                start_index += len(chunk_content) + 1  # +1 for the newline

            current_chunk.append(node_str)
            current_size += node_size

        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            chunks.append(Chunk(chunk_content, start_index, start_index + len(chunk_content)))

        return chunks

    def _line_based_chunking(self, content: str, chunk_size: int) -> List[Chunk]:
        lines = content.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        start_index = 0

        for line in lines:
            line_size = len(line.split())
            if current_size + line_size > chunk_size and current_chunk:
                chunk_content = '\n'.join(current_chunk)
                chunks.append(Chunk(chunk_content, start_index, start_index + len(chunk_content)))
                current_chunk = []
                current_size = 0
                start_index += len(chunk_content) + 1  # +1 for the newline

            current_chunk.append(line)
            current_size += line_size

        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            chunks.append(Chunk(chunk_content, start_index, start_index + len(chunk_content)))

        return chunks

class MixedContentChunkingStrategy(BaseChunkingStrategy):
    """Chunking strategy for documents with both text and code."""

    def __init__(self, text_strategy: TextChunkingStrategy, code_strategy: CodeChunkingStrategy):
        self.text_strategy = text_strategy
        self.code_strategy = code_strategy

    def create_chunks(self, content: str, chunk_size: int) -> List[Chunk]:
        # Split content into text and code blocks
        blocks = re.split(r'(```[\s\S]*?```)', content)
        chunks = []

        for block in blocks:
            if block.startswith('```') and block.endswith('```'):
                # Code block
                code_content = block[3:-3].strip()
                chunks.extend(self.code_strategy.create_chunks(code_content, chunk_size))
            else:
                # Text block
                chunks.extend(self.text_strategy.create_chunks(block, chunk_size))

        return chunks

def get_chunking_strategy(content_type: str) -> ChunkingStrategy:
    """Factory function to get the appropriate chunking strategy based on content type."""
    match content_type.lower():
        case "text":
            return TextChunkingStrategy()
        case "code":
            return CodeChunkingStrategy()
        case "mixed":
            return MixedContentChunkingStrategy(TextChunkingStrategy(), CodeChunkingStrategy())
        case _:
            raise ValueError(f"Unsupported content type: {content_type}")

def chunk_content(content: str, content_type: str, chunk_size: int) -> List[Chunk]:
    """Chunk the given content using the appropriate strategy."""
    strategy = get_chunking_strategy(content_type)
    return strategy.create_chunks(content, chunk_size)

if __name__ == "__main__":
    # Example usage
    text_content = """
    This is a sample text. It contains multiple sentences.
    We will use this to demonstrate text chunking.
    The TextChunkingStrategy should handle this content appropriately.
    """

    code_content = """
    def example_function():
        print("This is a sample function")
        for i in range(5):
            print(f"Iteration {i}")

    class ExampleClass:
        def __init__(self):
            self.value = 42

        def get_value(self):
            return self.value
    """

    mixed_content = """
    This is a mixed content example.
    It contains both text and code.

    ```python
    def greet(name):
        return f"Hello, {name}!"

    print(greet("World"))
    ```

    After the code block, we have more text.
    This demonstrates the MixedContentChunkingStrategy.
    """

    print("Text Chunking Example:")
    text_chunks = chunk_content(text_content, "text", 20)
    for i, chunk in enumerate(text_chunks):
        print(f"Chunk {i + 1}: {chunk.content}")

    print("\nCode Chunking Example:")
    code_chunks = chunk_content(code_content, "code", 50)
    for i, chunk in enumerate(code_chunks):
        print(f"Chunk {i + 1}: {chunk.content}")

    print("\nMixed Content Chunking Example:")
    mixed_chunks = chunk_content(mixed_content, "mixed", 30)
    for i, chunk in enumerate(mixed_chunks):
        print(f"Chunk {i + 1}: {chunk.content}")