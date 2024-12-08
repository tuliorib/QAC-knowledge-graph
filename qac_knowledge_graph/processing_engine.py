import asyncio
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, field
import logging

from document_processor import get_document_processor
from code_processor import get_code_processor
from chunking import get_chunking_strategy, Chunk
from metadata_extractor import extract_metadata, Metadata

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ProcessedChunk:
    chunk: Chunk
    metadata: Metadata

@dataclass
class ProcessingResult:
    file_path: Path
    content_type: str
    chunks: List[ProcessedChunk]
    global_metadata: Metadata
    processing_metadata: Dict[str, Any] = field(default_factory=dict)

class ProcessingEngine:
    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size

    async def process_document(self, file_path: Path) -> ProcessingResult:
        try:
            content_type = self._determine_content_type(file_path)
            content = await self._read_content(file_path, content_type)
            chunks = self._create_chunks(content, content_type)
            global_metadata = await self._extract_global_metadata(content, content_type)
            processed_chunks = await self._process_chunks_sequentially(chunks, content_type, global_metadata)

            result = ProcessingResult(
                file_path=file_path,
                content_type=content_type,
                chunks=processed_chunks,
                global_metadata=global_metadata,
                processing_metadata={
                    "num_chunks": len(processed_chunks),
                    "chunk_size": self.chunk_size,
                    "file_size": file_path.stat().st_size,
                }
            )

            logger.info(f"Successfully processed {file_path}")
            return result

        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            raise

    def _determine_content_type(self, file_path: Path) -> str:
        """Determine the content type based on file extension."""
        extension = file_path.suffix.lower()
        if extension in ['.py', '.js', '.java', '.cpp']:
            return 'code'
        elif extension in ['.txt', '.md', '.rst']:
            return 'text'
        elif extension in ['.pdf', '.docx', '.html']:
            return 'document'
        else:
            raise ValueError(f"Unsupported file type: {extension}")

    async def _read_content(self, file_path: Path, content_type: str) -> str:
        if content_type == 'code':
            processor = get_code_processor(file_path)
        else:
            processor = get_document_processor(file_path)

        return await asyncio.to_thread(processor.read_document, file_path)

    def _create_chunks(self, content: str, content_type: str) -> List[Chunk]:
        """Create chunks from the content."""
        chunking_strategy = get_chunking_strategy(content_type)
        return chunking_strategy.create_chunks(content, self.chunk_size)

    async def _extract_global_metadata(self, content: str, content_type: str) -> Metadata:
        """Extract global metadata from the entire content asynchronously."""
        strategy = "code" if content_type == "code" else "general"
        return await asyncio.to_thread(extract_metadata, content, strategy=strategy)

    async def _process_chunks_sequentially(self, chunks: List[Chunk], content_type: str, global_metadata: Metadata) -> List[ProcessedChunk]:
        """Process chunks sequentially with context awareness."""
        processed_chunks = []
        for chunk in chunks:
            metadata = await self._process_single_chunk(chunk, content_type, global_metadata, processed_chunks)
            processed_chunks.append(ProcessedChunk(chunk, metadata))
        return processed_chunks

    async def _process_single_chunk(self, chunk: Chunk, content_type: str, global_metadata: Metadata, previous_chunks: List[ProcessedChunk]) -> Metadata:
        """Process a single chunk with context awareness."""
        strategy = "code" if content_type == "code" else "general"
        context = {
            "global_metadata": global_metadata.__dict__,
            "previous_chunks": [pc.metadata.__dict__ for pc in previous_chunks[-3:]]  # Get metadata from last 3 chunks
        }
        return await asyncio.to_thread(extract_metadata, chunk.content, strategy=strategy, context=context)

    async def process_batch(self, file_paths: List[Path]) -> List[ProcessingResult]:
        """Process a batch of documents asynchronously."""
        tasks = [self.process_document(file_path) for file_path in file_paths]
        return await asyncio.gather(*tasks)

# Example usage
async def main():
    engine = ProcessingEngine(chunk_size=500)
    
    # Sample files (replace with actual file paths)
    file_paths = [
        Path("sample_text.txt"),
        Path("sample_code.py"),
        Path("sample_document.pdf")
    ]

    results = await engine.process_batch(file_paths)

    for result in results:
        print(f"Processed {result.file_path}:")
        print(f"Content Type: {result.content_type}")
        print(f"Number of Chunks: {len(result.chunks)}")
        print(f"Global Metadata: {result.global_metadata}")
        print(f"Processing Metadata: {result.processing_metadata}")
        print("First chunk preview:")
        if result.chunks:
            first_chunk = result.chunks[0]
            print(f"  Content: {first_chunk.chunk.content[:100]}...")
            print(f"  Metadata: {first_chunk.metadata}")
        print("\n")

if __name__ == "__main__":
    asyncio.run(main())