import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

"""
TASK 6: Cosine Similarity
---------------------------
Part A: Implement cosine_similarity_manual(v1, v2) WITHOUT
        using numpy.  Use only Python loops / math.
Part B: Implement cosine_similarity_numpy(v1, v2) using numpy.


Both should return a float between -1 and 1.


Then embed these two pairs and print which pair is more similar:
  Pair 1: "dog" vs "puppy"
  Pair 2: "dog" vs "automobile"


Formula:
  cosine_similarity = (v1 · v2) / (||v1|| × ||v2||)


HINT:
  dot product: sum(a*b for a, b in zip(v1, v2))
  magnitude  : sum(x**2 for x in v) ** 0.5
  numpy equiv: np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
"""


import math
import numpy as np

def cosine_similarity_manual(v1: list, v2: list) -> float:
    """Computes cosine similarity using pure Python."""
    dot_product = sum(a * b for a, b in zip(v1, v2))
    magnitude_v1 = math.sqrt(sum(x * x for x in v1))
    magnitude_v2 = math.sqrt(sum(x * x for x in v2))
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0.0
    return dot_product / (magnitude_v1 * magnitude_v2)

def cosine_similarity_numpy(v1: list, v2: list) -> float:
    """Computes cosine similarity using numpy."""
    v1 = np.array(v1)
    v2 = np.array(v2)
    return float(
        np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    )

from langchain_openai import OpenAIEmbeddings
def compare_word_pairs() -> dict:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    dog = embeddings.embed_query("dog")
    puppy = embeddings.embed_query("puppy")
    automobile = embeddings.embed_query("automobile")
    sim_dog_puppy = cosine_similarity_numpy(dog, puppy)
    sim_dog_auto = cosine_similarity_numpy(dog, automobile)
    return {
        "dog_vs_puppy": sim_dog_puppy,
        "dog_vs_automobile": sim_dog_auto,
        "more_similar_pair": (
            "dog vs puppy"
            if sim_dog_puppy > sim_dog_auto
            else "dog vs automobile"
        ),
    }


