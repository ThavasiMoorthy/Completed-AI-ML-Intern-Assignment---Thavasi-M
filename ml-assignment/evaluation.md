# Evaluation: Design Choices for Trigram Language Model

## Overview
This document summarizes the key design decisions made while implementing the trigram (N=3) language model from scratch. The implementation focuses on clean architecture, efficient data structures, and probabilistic text generation.

## 1. N-Gram Storage Structure

**Decision**: Used nested dictionaries (`defaultdict`) to store trigram counts.

**Structure**: `counts[w1][w2][w3] = count`, where:
- `w1`, `w2`, `w3` are the three consecutive words forming a trigram
- `count` is the frequency of occurrence of this trigram

**Rationale**:
- **Efficiency**: O(1) lookup time for accessing trigram counts
- **Flexibility**: Easy to add new trigrams without pre-allocation
- **Memory**: Only stores observed trigrams, not all possible combinations
- **Simplicity**: Intuitive mapping from the mathematical concept to code structure

**Alternative Considered**: A single dictionary with tuple keys `(w1, w2, w3)` could also work, but nested dictionaries provide better structure for accessing context probabilities.

## 2. Text Cleaning and Preprocessing

**Decisions**:
1. **Lowercasing**: All text converted to lowercase for consistency
2. **Whitespace Normalization**: Multiple spaces/tabs/newlines collapsed to single spaces
3. **Tokenization**: Simple whitespace-based splitting (preserves punctuation within words)

**Rationale**:
- Lowercasing reduces vocabulary size and prevents duplicate learning of capitalized vs. lowercase words
- Preserving sentence structure by not aggressively removing punctuation maintains linguistic patterns
- Simple tokenization is sufficient for this task and avoids over-complication

**Note**: More advanced cleaning (stemming, lemmatization) was avoided to keep the implementation focused and maintain word-level patterns.

## 3. Padding Strategy

**Decision**: Used special tokens `<start>` and `<end>` to mark sentence boundaries.

**Implementation**:
- Each sequence begins with two `<start>` tokens: `[<start>, <start>, w1, w2, w3, ..., <end>]`
- Two start tokens are needed because trigrams require a 2-word context
- Each sequence ends with one `<end>` token

**Rationale**:
- Allows the model to learn sentence-initial patterns (first words after start)
- Enables natural sentence termination during generation
- Standard approach in n-gram language modeling

## 4. Unknown Word Handling

**Decision**: Words appearing ≤ threshold times (default: 1) are replaced with `<unk>` token.

**Implementation**:
1. First pass: Count frequency of all words in the corpus
2. Second pass: Replace words below threshold with `<unk>`
3. Special tokens (`<start>`, `<end>`) are never replaced

**Rationale**:
- Prevents vocabulary explosion from rare words
- Allows model to handle words not seen during training (during generation)
- Threshold approach is standard in language modeling (similar to min_count in word2vec)

**Trade-off**: Setting threshold > 1 reduces vocabulary size further but loses information about rare words. Default of 1 was chosen to balance vocabulary size and information retention.

## 5. Generation Algorithm and Probabilistic Sampling

**Decision**: Used weighted random sampling from probability distribution derived from trigram counts.

**Implementation**:
1. **Probability Calculation**: For context (w1, w2), compute P(w3 | w1, w2) = count(w1, w2, w3) / Σ count(w1, w2, *), where * is any possible third word
2. **Sampling**: Use `random.choices()` with weights equal to probabilities to sample the next word
3. **Context Window**: Maintain sliding window of last 2 words (w1, w2) as context

**Algorithm Flow**:
```
1. Initialize: w1 = <start>, w2 = <start>
2. While not <end> and length < max_length:
   a. Get probabilities for next word given (w1, w2)
   b. Sample next word from distribution
   c. Add word to output (if not special token)
   d. Update context: w1 = w2, w2 = next_word
```

**Rationale**:
- **Probabilistic vs. Deterministic**: Using probabilities ensures diversity in generated text rather than always choosing the most likely word (which would be repetitive)
- **Context-Aware**: Each word choice depends on the previous 2 words, maintaining local coherence
- **Stopping Condition**: Generation stops when `<end>` token is sampled or max_length reached

**Alternative Considered**: Greedy decoding (always pick most likely word) was rejected because it produces repetitive, uninteresting text.

## 6. Handling Edge Cases

**Decisions**:
1. **Empty Text**: If training text is empty, model sets trained flag but has no counts
2. **Unknown Context**: If generation encounters unseen context (w1, w2), return `<end>` to stop generation
3. **No Training Data**: Generate returns empty string if model hasn't been trained

**Rationale**: Graceful degradation prevents crashes and provides predictable behavior.

## 7. Project Gutenberg Integration

**Decision**: Created utility functions for downloading and cleaning Project Gutenberg texts.

**Features**:
- Automatic download of texts by book ID
- Removal of Project Gutenberg headers and footers
- License information removal
- Whitespace normalization

**Rationale**: Makes it easy to train on high-quality, copyright-free texts without manual preprocessing.

## Summary

The implementation prioritizes:
1. **Simplicity**: Clear, readable code with straightforward data structures
2. **Correctness**: Proper probabilistic sampling and probability calculations
3. **Robustness**: Handles edge cases and provides graceful error handling
4. **Extensibility**: Easy to modify thresholds, add preprocessing steps, or extend to higher N-grams

The model successfully learns word sequences from text and generates novel text that reflects the patterns in the training corpus.
