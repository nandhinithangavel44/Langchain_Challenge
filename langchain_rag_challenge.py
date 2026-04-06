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

"""
TASK 7: Batch Embedding with Chunking
----------------------------------------
Given a long text document, split it into overlapping chunks
using RecursiveCharacterTextSplitter, then embed all chunks
in a single batch call.  Return:
  {
    "num_chunks"   : int,
    "chunk_size"   : int,   # configured chunk size
    "overlap"      : int,   # configured overlap
    "embedding_dim": int,
    "chunks"       : list[str]
  }

Use chunk_size=200, chunk_overlap=40.

HINT:
  from langchain.text_splitter import RecursiveCharacterTextSplitter
  splitter = RecursiveCharacterTextSplitter(
      chunk_size=200, chunk_overlap=40
  )
  chunks = splitter.split_text(long_text)
  vectors = embeddings.embed_documents(chunks)
"""

SAMPLE_DOCUMENT = """
LangChain is a framework for developing applications powered by language models.
It provides tools for prompt management, chains, agents, and memory.
LangChain integrates with many LLM providers including OpenAI, Anthropic, and Cohere.
The framework also supports vector stores, document loaders, and output parsers.
RAG (Retrieval-Augmented Generation) is a technique that enhances LLM responses
by fetching relevant documents from a knowledge base at query time.
pgvector is a PostgreSQL extension that enables efficient storage and similarity
search of high-dimensional vector embeddings directly inside a relational database.
LangSmith is an observability platform for LangChain applications that provides
tracing, evaluation, and debugging of LLM pipelines.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

def batch_embed_with_chunks(
    text: str,
    chunk_size: int,
    overlap: int
) -> dict:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )

    chunks = splitter.split_text(text)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectors = embeddings.embed_documents(chunks)

    return {
        "num_chunks": len(chunks),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "embedding_dim": len(vectors[0]),
        "chunks": chunks
    }

"""
TASK 8: Compare Two Embedding Models
--------------------------------------
Embed the same sentence using two different OpenAI models:
  Model A: text-embedding-3-small   (1536 dims)
  Model B: text-embedding-3-large   (3072 dims)


For the sentence:  "Vector databases power semantic search."


Return a dict:
  {
    "sentence"   : str,
    "model_a"    : {"model": str, "dims": int, "first_3": list[float]},
    "model_b"    : {"model": str, "dims": int, "first_3": list[float]},
    "dim_ratio"  : float   # model_b_dims / model_a_dims
  }


HINT:
  OpenAIEmbeddings(model="text-embedding-3-small")
  OpenAIEmbeddings(model="text-embedding-3-large")
  embeddings.embed_query(sentence) → single vector (list of floats)
"""
from langchain_openai import OpenAIEmbeddings

def compare_embedding_models(sentence: str) -> dict:
    """Embeds a sentence with two models and compares their dimensions."""
    model_a = OpenAIEmbeddings(model="text-embedding-3-small")
    model_b = OpenAIEmbeddings(model="text-embedding-3-large")
    vec_a = model_a.embed_query(sentence)
    vec_b = model_b.embed_query(sentence)

    return {
        "sentence": sentence,
        "model_a": {
            "model": "text-embedding-3-small",
            "dims": len(vec_a),
            "first_3": vec_a[:3],
        },
        "model_b": {
            "model": "text-embedding-3-large",
            "dims": len(vec_b),
            "first_3": vec_b[:3],
        },
        "dim_ratio": len(vec_b) / len(vec_a),
    }




