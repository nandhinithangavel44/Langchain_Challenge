# TASK 19 — Create a LangSmith Dataset
"""
# ─────────────────────────────────────────────────────────────
TASK 19: Create a LangSmith Dataset and Add Examples
------------------------------------------------------
Use the LangSmith SDK to:
  1. Create a dataset named "rag-eval-dataset".
  2. Add 3 question-answer example pairs to it.
  3. Return the dataset id as a string.


Examples to add:
  Q: "What does RAG stand for?"
     A: "Retrieval-Augmented Generation"
  Q: "What PostgreSQL extension enables vector search?"
     A: "pgvector"
  Q: "What LangChain tool provides observability?"
     A: "LangSmith"


HINT:
  from langsmith import Client
  client = Client()


  dataset = client.create_dataset("rag-eval-dataset")
  client.create_examples(
      inputs=[{"question": q} for q in questions],
      outputs=[{"answer": a} for a in answers],
      dataset_id=dataset.id
  )
"""


from langsmith import Client
from langsmith.utils import LangSmithConflictError

def create_langsmith_dataset():
    client = Client()
    dataset_name = "rag-eval-dataset"

    try:
        dataset = client.create_dataset(dataset_name)
    except LangSmithConflictError:
        dataset = client.read_dataset(dataset_name=dataset_name)


    client.create_examples(
        inputs=[
            {"question": "What is LangChain?"},
            {"question": "What is LCEL used for?"},
            {"question": "What distance metrics does pgvector support?"},
        ],
        outputs=[
            {"answer": "framework"},
            {"answer": "composition"},
            {"answer": "distance metrics"},
        ],
        dataset_id=dataset.id,
    )


    return dataset.id

print("\n[Task 19] Create LangSmith Dataset")
dataset_id = create_langsmith_dataset()
print(f"  Dataset ID: {dataset_id}")