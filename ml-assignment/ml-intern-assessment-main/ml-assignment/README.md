# Trigram Language Model

This directory contains the core assignment files for the Trigram Language Model - a probabilistic language model that learns word sequences from text and generates new text.

## Overview

The implementation includes:
- **TrigramModel**: A trigram (N=3) language model that learns word patterns from text
- **Text Cleaning**: Functions to preprocess and clean text data
- **Project Gutenberg Integration**: Utilities to download and process classic literature texts
- **Probabilistic Generation**: Text generation using weighted random sampling

## Project Structure

```
ml-assignment/
├── src/
│   ├── ngram_model.py      # Main TrigramModel implementation
│   ├── generate.py         # Script to train and generate text
│   └── utils.py            # Utility functions for data extraction and cleaning
├── data/
│   └── example_corpus.txt  # Example training corpus
├── tests/
│   └── test_ngram.py       # Unit tests
├── evaluation.md           # Design choices summary
└── README.md              # This file
```

## Installation

1. Install required dependencies:
```bash
pip install pytest
```

2. Navigate to the project directory:
```bash
cd ml-assignment
```

## How to Run

### Basic Usage

1. **Train on example corpus and generate text:**
```bash
cd src
python generate.py
```

This will:
- Load the example corpus from `data/example_corpus.txt`
- Train the trigram model
- Generate 3 samples of new text

### Using Project Gutenberg Texts

To train on a Project Gutenberg book, modify `src/generate.py`:

```python
from utils import download_gutenberg_text, clean_gutenberg_text

# Download and clean text
text = download_gutenberg_text(11)  # Alice's Adventures in Wonderland
cleaned_text = clean_gutenberg_text(text)

# Train model
model = TrigramModel()
model.fit(cleaned_text)

# Generate text
generated_text = model.generate(max_length=100)
print(generated_text)
```

**Recommended Project Gutenberg Books:**
- Alice's Adventures in Wonderland: Book ID 11
- Pride and Prejudice: Book ID 1342
- Frankenstein: Book ID 84
- A Tale of Two Cities: Book ID 98

### Using Your Own Text

```python
from src.ngram_model import TrigramModel

# Create model
model = TrigramModel(unk_threshold=1)

# Train on your text
with open("your_text_file.txt", "r", encoding='utf-8') as f:
    text = f.read()
model.fit(text)

# Generate text
generated_text = model.generate(max_length=50)
print(generated_text)
```

### Python API

```python
from src.ngram_model import TrigramModel

# Initialize model
model = TrigramModel(unk_threshold=1)

# Train on text
model.fit("Your training text here...")

# Generate text
text = model.generate(max_length=100)

# Generate with custom seed
text = model.generate(max_length=100, seed_text=("the", "cat"))
```

## Running Tests

Run the test suite:
```bash
cd ml-assignment
pytest tests/
```

Or run specific test:
```bash
pytest tests/test_ngram.py::test_fit_and_generate
```

## Key Features

1. **Text Cleaning**: Automatic lowercasing and whitespace normalization
2. **Unknown Word Handling**: Rare words replaced with `<unk>` token
3. **Padding**: Special tokens for sentence boundaries (`<start>`, `<end>`)
4. **Probabilistic Generation**: Weighted random sampling from trigram distribution
5. **Project Gutenberg Support**: Download and clean classic literature texts

## Model Parameters

- `unk_threshold` (int): Words appearing ≤ this count become `<unk>`. Default: 1
- `max_length` (int): Maximum number of words in generated text. Default: 50

## Design Choices

Please see `evaluation.md` for a detailed explanation of design decisions, including:
- N-gram storage structure
- Text preprocessing approach
- Padding strategy
- Unknown word handling
- Probabilistic sampling algorithm
