
"""
TASK 20: LangSmith Evaluation (evaluate)
------------------------------------------
Run an automated evaluation of your RAG pipeline using the
dataset created in Task 19.


Steps:
  1. Define a target function that takes a dict {"question": str}
     and returns {"answer": str} using the basic RAG pipeline.
  2. Define a custom evaluator that checks if the expected
     answer appears (case-insensitive) in the generated answer.
  3. Run the evaluation using langsmith.evaluate().
  4. Return the evaluation results summary dict:
     {"dataset": str, "num_examples": int, "pass_rate": float}


HINT:
  from langsmith.evaluation import evaluate, LangChainStringEvaluator


  def target(inputs: dict) -> dict:
      return {"answer": basic_rag_pipeline(RAG_DOCUMENTS, inputs["question"])}


  results = evaluate(
      target,
      data="rag-eval-dataset",
      evaluators=[...],
      experiment_prefix="rag-challenge-eval",
  )
"""


from langsmith.evaluation import evaluate

def run_langsmith_evaluation() -> dict:
    def target(inputs: dict) -> dict:
        return {
            "answer": basic_rag_pipeline(
                RAG_DOCUMENTS,
                inputs["question"]
            )
        }


    def evaluator(run, example):
        expected = example.outputs["answer"].lower()
        predicted = run.outputs["answer"].lower()

        passed = expected in predicted

        return {
            "key": "answer_match",
            "score": 1.0 if passed else 0.0,
        }

    results = evaluate(
        target,
        data="rag-eval-dataset",
        evaluators=[evaluator],
        experiment_prefix="rag-challenge-eval",
    )

    return {
        "dataset": "rag-eval-dataset",
        "num_examples": len(results),
        "pass_rate": sum(r.get("score", 0) for r in results) / len(results),
    }

print("\n[Task 20] Run LangSmith Evaluation")
eval_summary = run_langsmith_evaluation()
print(f"  Dataset     : {eval_summary.get('dataset')}")
print(f"  # Examples  : {eval_summary.get('num_examples')}")
print(f"  Pass rate   : {eval_summary.get('pass_rate')}")