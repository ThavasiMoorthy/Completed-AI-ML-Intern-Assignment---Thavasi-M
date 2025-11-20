import random
import re
from collections import defaultdict
from typing import Dict, List, Tuple


class TrigramModel:
    """
    A trigram (N=3) language model that learns word sequences from text
    and can generate new text probabilistically.
    """
    
    # Special tokens for sentence boundaries and unknown words
    START_TOKEN = "<start>"
    END_TOKEN = "<end>"
    UNK_TOKEN = "<unk>"
    
    def __init__(self, unk_threshold=1):
        """
        Initializes the TrigramModel.
        
        Args:
            unk_threshold (int): Words appearing fewer times than this 
                                will be replaced with <unk> during training.
                                Default is 1 (only unseen words become unknown).
        """
        # Nested dictionary to store trigram counts: counts[w1][w2][w3] = count
        self.counts: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )
        
        # Store vocabulary for later use
        self.vocab: Dict[str, int] = defaultdict(int)
        
        # Threshold for unknown words
        self.unk_threshold = unk_threshold
        
        # Flag to check if model has been trained
        self.trained = False

    def _clean_text(self, text: str) -> str:
        """
        Cleans the input text by converting to lowercase and normalizing whitespace.
        We preserve sentence structure by keeping periods.
        
        Args:
            text (str): Raw input text
            
        Returns:
            str: Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Normalize whitespace (replace multiple spaces/tabs/newlines with single space)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenizes text into words, splitting on whitespace and punctuation.
        Handles sentence boundaries.
        
        Args:
            text (str): Cleaned text
            
        Returns:
            List[str]: List of tokens (words)
        """
        # Split on whitespace
        tokens = text.split()
        return tokens

    def _add_padding(self, tokens: List[str]) -> List[str]:
        """
        Adds padding tokens at the beginning and end of sentences.
        For trigrams, we need 2 start tokens and 1 end token per sentence.
        
        Args:
            tokens (List[str]): List of word tokens
            
        Returns:
            List[str]: Tokens with padding added
        """
        if not tokens:
            return []
        
        # Add two start tokens at the beginning
        padded = [self.START_TOKEN, self.START_TOKEN]
        
        # Add end token after each sentence (assuming sentence ends with period)
        # For simplicity, we'll add end token at the end of the entire sequence
        # and treat the sequence as one long sentence
        padded.extend(tokens)
        padded.append(self.END_TOKEN)
        
        return padded

    def _handle_unknown_words(self, tokens: List[str]) -> List[str]:
        """
        Replaces rare words (below threshold) with <unk> token.
        First pass counts all words, second pass replaces rare ones.
        
        Args:
            tokens (List[str]): List of tokens
            
        Returns:
            List[str]: Tokens with rare words replaced by <unk>
        """
        # First pass: count word frequencies (excluding special tokens)
        word_counts = defaultdict(int)
        for token in tokens:
            if token not in [self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN]:
                word_counts[token] += 1
        
        # Second pass: replace words below threshold with <unk>
        processed_tokens = []
        for token in tokens:
            if token in [self.START_TOKEN, self.END_TOKEN]:
                processed_tokens.append(token)
            elif word_counts[token] <= self.unk_threshold:
                processed_tokens.append(self.UNK_TOKEN)
                self.vocab[self.UNK_TOKEN] += 1
            else:
                processed_tokens.append(token)
                self.vocab[token] += 1
        
        return processed_tokens

    def fit(self, text: str):
        """
        Trains the trigram model on the given text.

        Args:
            text (str): The text to train the model on.
        """
        if not text or not text.strip():
            self.trained = True
            return
        
        # Step 1: Clean the text
        cleaned_text = self._clean_text(text)
        
        # Step 2: Tokenize into words
        tokens = self._tokenize(cleaned_text)
        
        # Step 3: Handle unknown words (replace rare words with <unk>)
        tokens = self._handle_unknown_words(tokens)
        
        # Step 4: Add padding tokens
        padded_tokens = self._add_padding(tokens)
        
        # Step 5: Count trigrams
        # For each position i, create trigram (w[i], w[i+1], w[i+2])
        for i in range(len(padded_tokens) - 2):
            w1 = padded_tokens[i]
            w2 = padded_tokens[i + 1]
            w3 = padded_tokens[i + 2]
            
            # Increment count for this trigram
            self.counts[w1][w2][w3] += 1
        
        self.trained = True

    def _get_next_word_probabilities(self, w1: str, w2: str) -> List[Tuple[str, float]]:
        """
        Converts trigram counts to probabilities for the next word given context (w1, w2).
        
        Args:
            w1 (str): First word of context
            w2 (str): Second word of context
            
        Returns:
            List[Tuple[str, float]]: List of (word, probability) tuples
        """
        if w1 not in self.counts or w2 not in self.counts[w1]:
            # Context not seen during training - return empty probabilities
            return []
        
        # Get all possible third words for this context
        trigram_context = self.counts[w1][w2]
        
        # Calculate total count for this context
        total_count = sum(trigram_context.values())
        
        if total_count == 0:
            return []
        
        # Convert counts to probabilities
        probabilities = [
            (word, count / total_count)
            for word, count in trigram_context.items()
        ]
        
        return probabilities

    def _sample_next_word(self, w1: str, w2: str) -> str:
        """
        Probabilistically samples the next word given context (w1, w2).
        
        Args:
            w1 (str): First word of context
            w2 (str): Second word of context
            
        Returns:
            str: Sampled next word, or <end> if no valid context found
        """
        probabilities = self._get_next_word_probabilities(w1, w2)
        
        if not probabilities:
            # If context not found, return end token to stop generation
            return self.END_TOKEN
        
        # Extract words and their probabilities
        words = [word for word, _ in probabilities]
        probs = [prob for _, prob in probabilities]
        
        # Weighted random choice based on probabilities
        next_word = random.choices(words, weights=probs, k=1)[0]
        
        return next_word

    def generate(self, max_length=50, seed_text: Tuple[str, str] = None) -> str:
        """
        Generates new text using the trained trigram model.

        Args:
            max_length (int): The maximum length of the generated text.
            seed_text (Tuple[str, str], optional): Initial context (w1, w2) to start generation.
                                                  If None, starts with <start> <start>.

        Returns:
            str: The generated text (without special tokens).
        """
        if not self.trained:
            return ""
        
        # If no counts exist, return empty string
        if not self.counts:
            return ""
        
        # Initialize generation with start tokens or seed text
        if seed_text is None:
            w1, w2 = self.START_TOKEN, self.START_TOKEN
        else:
            w1, w2 = seed_text
        
        generated_words = []
        length = 0
        
        # Generate words until we hit end token or max_length
        while length < max_length:
            next_word = self._sample_next_word(w1, w2)
            
            # Stop if we generate end token
            if next_word == self.END_TOKEN:
                break
            
            # Add word to generated sequence (skip special tokens in output)
            if next_word not in [self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN]:
                generated_words.append(next_word)
            elif next_word == self.UNK_TOKEN:
                # Include unknown token as a placeholder word
                generated_words.append(next_word)
            
            # Update context: shift window forward
            w1, w2 = w2, next_word
            length += 1
        
        # Join words with spaces to form generated text
        return " ".join(generated_words)
