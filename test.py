import multiprocessing as mp
from collections import Counter
from functools import reduce


def process_chunk(chunk):
    # Use Counter to count frequency of each alphabetic character
    return Counter(char.lower() for char in chunk if char.isalpha())


def combine_counters(counter1, counter2):
    # Combine two Counter objects
    return counter1 + counter2


def count_character_frequency_parallel(
    filename, chunk_size=50 * 1024 * 1024
):  # 50MB chunks
    with open(filename, "r", encoding="utf-8") as file:
        chunks = []
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            chunks.append(chunk)

    with mp.Pool() as pool:
        results = pool.map(process_chunk, chunks)

    # Combine all counters into a single counter
    return reduce(combine_counters, results, Counter())


char_frequencies = count_character_frequency_parallel(
    "/mnt/data/AI/kaggle/tr_corpus_clean.txt"
)

# Print characters and their frequencies in descending order of frequency
for char, freq in sorted(char_frequencies.items(), key=lambda x: (-x[1], x[0])):
    print(f"{char}: {freq}")

# Alternatively, print just the most common characters
print("\nMost common characters:")
for char, freq in char_frequencies.most_common(10):
    print(f"{char}: {freq}")
