from ngram_model import TrigramModel
import sys
import os

# Add parent directory to path for importing utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.utils import download_gutenberg_text, clean_gutenberg_text, extract_text_from_file
except ImportError:
    # Fallback if utils not available
    pass


def main():
    """
    Main function to train and generate text using the TrigramModel.
    Uses one of the recommended Project Gutenberg books as specified in the assignment.
    """
    model = None
    
    # Option 1: Download and train on Project Gutenberg text (recommended)
    # Using Alice's Adventures in Wonderland by Lewis Carroll (Book ID: 11)
    # Other recommended books:
    # - Pride and Prejudice: Book ID 1342
    # - Frankenstein: Book ID 84
    # - A Tale of Two Cities: Book ID 98
    print("Downloading Project Gutenberg text...")
    print("Book: Alice's Adventures in Wonderland by Lewis Carroll")
    try:
        # Alice's Adventures in Wonderland (book ID: 11)
        text = download_gutenberg_text(11)
        cleaned_text = clean_gutenberg_text(text)
        
        # For larger corpora (like Project Gutenberg), use unk_threshold=1
        # This replaces words that appear only once with <unk> to handle rare words
        model = TrigramModel(unk_threshold=1)
        model.fit(cleaned_text)
        print("Training complete!")
        print(f"Trained on {len(cleaned_text.split())} words")
    except Exception as e:
        print(f"Error downloading Project Gutenberg text: {e}")
        print("Falling back to example corpus...")
        
        # Option 2: Fallback to example corpus if download fails
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(os.path.dirname(script_dir), "data")
        corpus_path = os.path.join(data_dir, "example_corpus.txt")
        
        if os.path.exists(corpus_path):
            print("Training on example corpus...")
            with open(corpus_path, "r", encoding='utf-8') as f:
                text = f.read()
            # For small corpora, use unk_threshold=0 to avoid replacing words with <unk>
            model = TrigramModel(unk_threshold=0)
            model.fit(text)
        else:
            print("Example corpus not found. Cannot proceed.")
            return
    
    if model is None or not model.trained:
        print("Model training failed. Cannot generate text.")
        return

    # Generate new text
    print("\nGenerating text...")
    print("-" * 50)
    for i in range(3):
        generated_text = model.generate(max_length=50)
        print(f"\nGenerated Text {i+1}:")
        print(generated_text)
        print("-" * 50)


if __name__ == "__main__":
    main()
