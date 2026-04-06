#TASK 18: LangSmith Tracing
"""
-----------------------------
Instrument a simple LCEL chain so every invocation is
traced in LangSmith.  Your function should:
  1. Set LANGCHAIN_TRACING_V2=true and LANGCHAIN_PROJECT.
  2. Build the same basic LCEL chain from Task 1.
  3. Add run_name and tags to the invocation config.
  4. Return the response AND the run_id of the trace.


Expected return:
  {"answer": str, "run_id": str}


HINT:
  from langchain_core.tracers.context import collect_runs


  with collect_runs() as cb:
      result = chain.invoke(
          {"topic": topic},
          config={"run_name": "task18_trace", "tags": ["challenge"]}
      )
  run_id = str(cb.traced_runs[0].id)
"""

import os
from langchain_core.tracers.context import collect_runs
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

def traced_chain(topic: str) -> dict:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "rag-challenge"

    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt = ChatPromptTemplate.from_template("Explain {topic} briefly.")
    chain = prompt | llm | StrOutputParser()

    with collect_runs() as cb:
        answer = chain.invoke(
            {"topic": topic},
            config={"run_name": "task18_trace", "tags": ["challenge"]},
        )

    run_id = str(cb.traced_runs[0].id)
    return {"answer": answer, "run_id": run_id}

print("[Task 18] Traced Chain")
traced = traced_chain("embeddings")
print(f"  Answer : {str(traced.get('answer', ''))[:80]}")
print(f"  Run ID : {traced.get('run_id')}")