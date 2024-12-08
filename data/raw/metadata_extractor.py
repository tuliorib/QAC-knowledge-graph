from typing import Dict, Any, List, Protocol, Callable
from dataclasses import dataclass, field
import json
import os
from openai import OpenAI

client = OpenAI()

@dataclass
class Metadata:
    """Represents extracted metadata."""
    concepts: List[str] = field(default_factory=list)
    subjects: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    people: List[str] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)
    organizations: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)
    title: str = ""
    author: str = ""
    citations: List[str] = field(default_factory=list)
    sentiment: str = ""
    key_sentences: List[str] = field(default_factory=list)
    entity_relationships: List[Dict[str, str]] = field(default_factory=list)
    additional_info: Dict[str, Any] = field(default_factory=dict)

class MetadataExtractor:
    """LLM-based metadata extractor with support for different strategies."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None):
        self.model = model
        self.client = self._initialize_client(api_key)

    def _initialize_client(self, api_key: str = None) -> OpenAI:
        """Initialize the OpenAI client with the API key."""
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable or provide it explicitly.")
        
        return OpenAI(api_key=api_key)

    def extract_metadata(self, content: str, strategy: str, context: Dict[str, Any] = None) -> Metadata:
        """Extract metadata using the specified strategy."""
        strategy_func = self.get_strategy(strategy)
        return strategy_func(content, context)

    def get_strategy(self, strategy: str) -> Callable:
        """Get the appropriate extraction strategy."""
        strategies = {
            "general": self.general_extraction_strategy,
            "academic": self.academic_extraction_strategy,
            "code": self.code_extraction_strategy,
            # Add more strategies as needed
        }
        return strategies.get(strategy, self.general_extraction_strategy)

    def general_extraction_strategy(self, content: str, context: Dict[str, Any] = None) -> Metadata:
        prompt = self._build_general_prompt(content, context)
        return self._execute_llm_call(prompt)

    def academic_extraction_strategy(self, content: str, context: Dict[str, Any] = None) -> Metadata:
        prompt = self._build_academic_prompt(content, context)
        return self._execute_llm_call(prompt)

    def code_extraction_strategy(self, content: str, context: Dict[str, Any] = None) -> Metadata:
        prompt = self._build_code_prompt(content, context)
        return self._execute_llm_call(prompt)

    def _build_general_prompt(self, content: str, context: Dict[str, Any] = None) -> str:
        return f"""Analyze the following text and provide metadata according to the specified format.
        Use the provided context for consistency if available.

        Context: {json.dumps(context) if context else "No context provided"}

        Text: {content[:10000]}  # Limiting to first 10000 characters

        Provide the following metadata in JSON format:

        1. concepts: Abstract ideas or themes discussed in the text (max 3)
        2. subjects: Specific areas or fields of study mentioned (max 3)
        3. topics: Particular issues or matters being discussed (max 3)
        4. people: Names of individuals mentioned
        5. dates: Any dates or time periods referenced
        6. organizations: Names of companies, institutions, or groups
        7. locations: Any places or geographical areas mentioned
        8. title: The title of the text (if applicable)
        9. author: The name of the author(s) (if mentioned)
        10. citations: Any references or citations
        11. sentiment: Overall sentiment of the text (positive, negative, or neutral)
        12. key_sentences: 1-2 important sentences that summarize main points
        13. entity_relationships: Relationships between entities in the format {{"subject": "entity1", "relationship": "verb", "object": "entity2"}}

        You are part of a bigger system, and you must always respond with a pure JSON response. The system will break if you don't. NEVER add a JSON code block syntax to your response."""

    def _build_academic_prompt(self, content: str, context: Dict[str, Any] = None) -> str:
        # Similar to _build_general_prompt, but with focus on academic-specific metadata
        pass

    def _build_code_prompt(self, content: str, context: Dict[str, Any] = None) -> str:
        # Similar to _build_general_prompt, but with focus on code-specific metadata
        pass

    def _execute_llm_call(self, prompt: str) -> Metadata:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes text and extracts key information."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )

        metadata_dict = json.loads(response.choices[0].message.content)
        return Metadata(**metadata_dict)
    
    

def extract_metadata(content: str, strategy: str = "general", context: Dict[str, Any] = None, api_key: str = None) -> Metadata:
    """Extract metadata from the given content using the specified strategy."""
    extractor = MetadataExtractor(api_key=api_key)
    return extractor.extract_metadata(content, strategy, context)


# Example usage
if __name__ == "__main__":
    content = "This is a sample text about artificial intelligence and its impact on society..."
    metadata = extract_metadata(content, strategy="general")
    print(json.dumps(metadata.__dict__, indent=2))