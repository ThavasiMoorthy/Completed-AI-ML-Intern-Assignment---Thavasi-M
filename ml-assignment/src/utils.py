"""
Utility functions for data extraction, cleaning, and preprocessing.
Includes functions for downloading and processing Project Gutenberg texts.
"""
import re
import urllib.request
import urllib.error


def download_gutenberg_text(book_id: int, save_path: str = None) -> str:
    """
    Downloads a text from Project Gutenberg by book ID.
    
    Project Gutenberg uses a simple URL scheme:
    https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt
    
    Recommended books for this assignment:
    1. Alice's Adventures in Wonderland: 11
    2. Pride and Prejudice: 1342
    3. Frankenstein: 84
    4. A Tale of Two Cities: 98
    
    Args:
        book_id (int): The Project Gutenberg book ID
        save_path (str, optional): Path to save the downloaded text.
                                   If None, text is not saved to disk.
    
    Returns:
        str: The downloaded text content
    
    Raises:
        urllib.error.URLError: If download fails
    """
    url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
    
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            text = response.read().decode('utf-8')
            
            # Save to file if path provided
            if save_path:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                print(f"Text saved to {save_path}")
            
            return text
    except urllib.error.URLError as e:
        print(f"Error downloading text: {e}")
        print(f"Trying alternative URL format...")
        # Try alternative URL format
        url_alt = f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt"
        try:
            with urllib.request.urlopen(url_alt, timeout=10) as response:
                text = response.read().decode('utf-8')
                if save_path:
                    with open(save_path, 'w', encoding='utf-8') as f:
                        f.write(text)
                return text
        except urllib.error.URLError:
            raise urllib.error.URLError(f"Failed to download book {book_id} from Project Gutenberg")


def clean_gutenberg_text(text: str) -> str:
    """
    Cleans Project Gutenberg text by removing:
    - Project Gutenberg header and footer
    - License information
    - Extra whitespace
    
    Args:
        text (str): Raw text from Project Gutenberg
    
    Returns:
        str: Cleaned text ready for training
    """
    # Find the start marker (usually "*** START OF")
    start_markers = [
        "*** start of",
        "***start of",
        "*start of the project gutenberg",
        "start of the project gutenberg"
    ]
    
    start_idx = -1
    for marker in start_markers:
        idx = text.lower().find(marker)
        if idx != -1:
            # Find the first newline after the marker
            start_idx = text.find('\n', idx)
            break
    
    if start_idx == -1:
        # If no start marker found, try to find first sentence
        # Look for common patterns after metadata
        patterns = [
            r"\*\*\*.*?\n\n",
            r"chapter [i1]\s*\n",
            r"chapter one\s*\n"
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                start_idx = match.end()
                break
    
    # Find the end marker (usually "*** END OF")
    end_markers = [
        "*** end of",
        "***end of",
        "*end of the project gutenberg",
        "end of the project gutenberg"
    ]
    
    end_idx = len(text)
    for marker in end_markers:
        idx = text.lower().rfind(marker)
        if idx != -1:
            # Find the last newline before the marker
            end_idx = text.rfind('\n', 0, idx)
            break
    
    # Extract the main text content
    if start_idx != -1 and end_idx > start_idx:
        text = text[start_idx:end_idx]
    
    # Remove excessive whitespace (multiple newlines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove extra spaces
    text = re.sub(r' +', ' ', text)
    
    return text.strip()


def extract_text_from_file(file_path: str) -> str:
    """
    Extracts and returns text from a local file.
    
    Args:
        file_path (str): Path to the text file
    
    Returns:
        str: Content of the file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error reading file: {e}")


def preprocess_text_for_training(text: str, clean_gutenberg: bool = True) -> str:
    """
    Comprehensive text preprocessing pipeline.
    
    Args:
        text (str): Raw text
        clean_gutenberg (bool): Whether to apply Gutenberg-specific cleaning
    
    Returns:
        str: Preprocessed text ready for model training
    """
    # Clean Gutenberg headers/footers if specified
    if clean_gutenberg:
        text = clean_gutenberg_text(text)
    
    # Basic cleaning already handled in TrigramModel._clean_text
    # This function is here for additional preprocessing if needed
    
    return text
