from __future__ import annotations

import abc
from pathlib import Path
from typing import Protocol, runtime_checkable, Tuple, Dict, Any
import tokenize
import io
import re
import ast

@runtime_checkable
class CodeProcessor(Protocol):
    """Protocol defining the interface for code processors."""

    @abc.abstractmethod
    def read_code(self, file_path: Path) -> str:
        """Read and return the content of a code file."""
        ...

    @abc.abstractmethod
    def extract_comments(self, code: str) -> Tuple[str, str]:
        """Extract comments from the code."""
        ...

    @abc.abstractmethod
    def extract_structure(self, code: str) -> Dict[str, Any]:
        """Extract structural information from the code."""
        ...


class BaseCodeProcessor(CodeProcessor):
    """Base class for code processors with common functionality."""

    def read_code(self, file_path: Path) -> str:
        """Read and return the content of a code file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()


class PythonProcessor(BaseCodeProcessor):
    """Processor for Python code files."""

    def extract_comments(self, code: str) -> Tuple[str, str]:
        """Extract comments from Python code."""
        comments = []
        code_without_comments = []
        
        try:
            tokens = tokenize.generate_tokens(io.StringIO(code).readline)
            for token in tokens:
                if token.type == tokenize.COMMENT:
                    comments.append(token.string)
                elif token.type != tokenize.STRING:
                    code_without_comments.append(token.string)
                elif token.string.startswith('"""') or token.string.startswith("'''"):
                    comments.append(token.string)
                else:
                    code_without_comments.append(token.string)
        except tokenize.TokenError:
            # Handle incomplete or invalid Python code
            pass

        return ' '.join(comments), ''.join(code_without_comments)

    def extract_structure(self, code: str) -> Dict[str, Any]:
        """Extract structural information from Python code."""
        try:
            tree = ast.parse(code)
            return {
                'imports': [node.names[0].name for node in ast.walk(tree) if isinstance(node, ast.Import)],
                'functions': [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)],
                'classes': [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)],
            }
        except SyntaxError:
            # Handle syntax errors in the code
            return {'error': 'Invalid Python syntax'}


class JavaScriptProcessor(BaseCodeProcessor):
    """Processor for JavaScript code files."""

    def extract_comments(self, code: str) -> Tuple[str, str]:
        """Extract comments from JavaScript code."""
        # Regular expression to match both single-line and multi-line comments
        comment_pattern = r'\/\/.*?$|\/\*.*?\*\/|\/\*[\s\S]*?\*\/|^\s*\*.*?$'
        comments = re.findall(comment_pattern, code, re.MULTILINE)
        code_without_comments = re.sub(comment_pattern, '', code, flags=re.MULTILINE)
        
        return ' '.join(comments), code_without_comments.strip()

    def extract_structure(self, code: str) -> Dict[str, Any]:
        """Extract structural information from JavaScript code."""
        # This is a simplified version. For more accurate results, consider using a JavaScript parser.
        function_pattern = r'\bfunction\s+(\w+)'
        class_pattern = r'\bclass\s+(\w+)'
        import_pattern = r'\b(?:import|require)\s*\(\s*[\'"](.+?)[\'"]\s*\)'

        return {
            'functions': re.findall(function_pattern, code),
            'classes': re.findall(class_pattern, code),
            'imports': re.findall(import_pattern, code),
        }


def get_code_processor(file_path: Path) -> CodeProcessor:
    """Factory function to get the appropriate code processor based on file extension."""
    match file_path.suffix.lower():
        case ".py":
            return PythonProcessor()
        case ".js":
            return JavaScriptProcessor()
        case _:
            raise ValueError(f"Unsupported code file type: {file_path.suffix}")


def process_code(file_path: Path) -> Dict[str, Any]:
    """Process a code file and return its content, comments, and structural information."""
    processor = get_code_processor(file_path)
    content = processor.read_code(file_path)
    comments, code_without_comments = processor.extract_comments(content)
    structure = processor.extract_structure(content)
    
    return {
        "content": content,
        "comments": comments,
        "code_without_comments": code_without_comments,
        "structure": structure,
        "metadata": {
            "file_name": file_path.name,
            "file_type": file_path.suffix,
            "file_size": file_path.stat().st_size,
        }
    }


if __name__ == "__main__":
    # Example usage
    file_path = Path("./data/raw/metadata_extractor.py")
    result = process_code(file_path)
    print(f"Processed {result['metadata']['file_name']}:")
    print(f"Content preview: {result['content'][:100]}...")
    print(f"Comments preview: {result['comments'][:100]}...")
    print(f"Structure: {result['structure']}")
    print(f"Metadata: {result['metadata']}")