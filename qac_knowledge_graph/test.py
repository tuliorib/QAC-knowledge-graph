import json
from pathlib import Path
from document_processor import process_document
from chunking import chunk_content
from metadata_extractor import extract_metadata

def process_and_save(file_path: Path, output_dir: Path):
    # Process the document
    result = process_document(file_path)
    
    # Chunk the content
    chunks = chunk_content(result['content'], "text", chunk_size=1000)
    
    # Extract metadata for each chunk
    processed_chunks = []
    for i, chunk in enumerate(chunks):
        metadata = extract_metadata(chunk.content, strategy="general")
        processed_chunks.append({
            "chunk_number": i + 1,
            "content": chunk.content,
            "metadata": metadata.__dict__
        })
    
    # Prepare the output
    output = {
        "file_name": file_path.name,
        "file_metadata": result['metadata'],
        "chunks": processed_chunks
    }
    
    # Save to file
    output_file = output_dir / f"{file_path.stem}_processed.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    
    print(f"Processed {file_path.name} and saved results to {output_file}")

def main():
    raw_dir = Path("./data/raw")
    output_dir = Path("./processed")
    output_dir.mkdir(exist_ok=True)
    
    files_to_process = [
        # raw_dir / "graham.pdf",
        raw_dir / "pymu.html"
    ]
    
    for file_path in files_to_process:
        if file_path.exists():
            process_and_save(file_path, output_dir)
        else:
            print(f"File not found: {file_path}")

if __name__ == "__main__":
    main()