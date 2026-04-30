import os
import re
import numpy as np
import pickle
from collections import Counter

# --- CONFIG ---
input_file = 'TinyStories-train.txt'
context_len = 256
train_ratio = 0.9
min_freq = 10
# regex from your code
pattern = r"<\|.*?\|>|\n|[a-zA-Z]+|[.,!?\"']"
counts = Counter()


# --- STEP 1: Pass 1 - Build Vocab (Streaming) ---
print("Pass 1: Building Vocab...")
# vocab = set(['<|startoftext|>', '<|pad|>', '<|unk|>']) # Added pad token
story_count = 0

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        tokens = re.findall(pattern, line.lower())
        counts.update(tokens)
        if '<|endoftext|>' in tokens:
            story_count += 1
            if story_count % 100000 == 0: print(f"  Scanned {story_count} stories...")

vocab = [word for word, count in counts.items() if count > min_freq]
vocab = sorted(list(set(vocab + ['<|startoftext|>', '<|pad|>', '<|unk|>'])))

word_to_i = {word: i for i, word in enumerate(vocab)}
itos = {i: word for i, word in enumerate(vocab)}
vocab_size = len(vocab)

print(f"Original Vocab: {len(counts)}")
print(f"Filtered Vocab (min_freq={min_freq}): {vocab_size}")

word_to_i = {word: i for i, word in enumerate(vocab)}
itos = {i: word for i, word in enumerate(vocab)}

start_idx = word_to_i['<|startoftext|>']
end_idx = word_to_i['<|endoftext|>']
pad_idx = word_to_i['<|pad|>'] # Use this instead of -1 for uint16 compatibility
unknown_idx = word_to_i['<|unk|>']
print(f"Vocab size: {vocab_size} | Total stories: {story_count}")

# --- STEP 2: Pass 2 - Encode and Write to Binaries ---
print("Pass 2: Encoding and writing to disk...")

# We create 4 files: X and Y for both Train and Val
# Each "row" is exactly context_len (256) uint16 integers
n_train_stories = int(story_count * train_ratio)
n_val_stories = story_count - n_train_stories

# Pre-allocate files
tr_X = np.memmap('train_X.bin', dtype=np.uint16, mode='w+', shape=(n_train_stories, context_len))
tr_Y = np.memmap('train_Y.bin', dtype=np.uint16, mode='w+', shape=(n_train_stories, context_len))
val_X = np.memmap('val_X.bin', dtype=np.uint16, mode='w+', shape=(n_val_stories, context_len))
val_Y = np.memmap('val_Y.bin', dtype=np.uint16, mode='w+', shape=(n_val_stories, context_len))

def get_encoded_pair(tokens):
    """Your exact build_dataset logic for a single story"""
    # Create context and target
    context = [start_idx] + [word_to_i.get(w, unknown_idx) for w in tokens[:-1]]
    target = [word_to_i.get(w, unknown_idx) for w in tokens[:-1]] + [end_idx]
    
    # Truncate
    context = context[:context_len]
    target = target[:context_len]
    
    # Pad (using pad_idx instead of -1 because uint16 is unsigned)
    context += [pad_idx] * (context_len - len(context))
    target += [pad_idx] * (context_len - len(target))
    
    return context, target

with open(input_file, 'r', encoding='utf-8') as f:
    current_story_tokens = []
    curr_idx = 0
    
    for line in f:
        tokens = re.findall(pattern, line.lower())
        current_story_tokens.extend(tokens)
        
        if '<|endoftext|>' in tokens:
            # We found the end of a story, process it
            x, y = get_encoded_pair(current_story_tokens)
            
            if curr_idx < n_train_stories:
                tr_X[curr_idx] = x
                tr_Y[curr_idx] = y
            else:
                v_idx = curr_idx - n_train_stories
                val_X[v_idx] = x
                val_Y[v_idx] = y
            
            curr_idx += 1
            current_story_tokens = [] # Reset for next story
            
            if curr_idx % 50000 == 0:
                print(f"  Processed {curr_idx}/{story_count} stories...")

# Flush to disk
for f in [tr_X, tr_Y, val_X, val_Y]: f.flush()

# Save metadata
with open('meta.pkl', 'wb') as f:
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'word_to_i': word_to_i,
        'pad_idx': pad_idx,
        'unknown_idx': unknown_idx,
        'start_idx': start_idx,
        'end_idx': end_idx
    }
    pickle.dump(meta, f)

print("Done! Files created: train_X.bin, train_Y.bin, val_X.bin, val_Y.bin, meta.pkl")