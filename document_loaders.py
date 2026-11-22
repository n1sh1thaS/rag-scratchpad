import os
import tempfile
from langchain_community.document_loaders import (
    TextLoader,
)

def load_text_file(temp_file_path: str) -> str:
    """
    Load a text file and return its content as a string.
    """
    try:
        loader = TextLoader(temp_file_path)
        documents = loader.load()

        for doc in documents:
            print(doc)
            print(doc.page_content)
    finally:
        os.remove(temp_file_path)


if __name__ == "__main__":
    # Create a temporary text file for demonstration
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(b"This is a sample text file for testing the TextLoader.")
        temp_file_path = temp_file.name

    # Load the text file and print its content
    load_text_file(temp_file_path)