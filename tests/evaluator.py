# tests/evaluator.py
from ragas import evaluation
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
from pipeline import MultimodalRAGPipeline
import pandas as pd

class RAGEvaluator:
    """
    Automated evaluation using Ragas.
    Metrics:
    - Faithfulness: Is the answer derived *only* from context? (Detects hallucinations)
    - Answer Relevancy: Does the answer actually address the user's question?
    """
    def __init__(self, pipeline: MultimodalRAGPipeline):
        self.pipeline = pipeline

    def run_evaluation(self, test_set: list):
        questions = [item["question"] for item in test_set]
        ground_truths = [[item["ground_truth"]] for item in test_set]
        
        answers = []
        contexts = []

        print("[EVAL] Running queries through the pipeline...")
        for q in questions:
            res = self.pipeline.query(q)
            answers.append(res["answer"])
            
            raw_contexts = [cit["chunk_text_preview"] for cit in res["citations"]]
            contexts.append(raw_contexts if raw_contexts else ["No context retrieved"])

        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }
        dataset = Dataset.from_dict(data)
        
        print("[EVAL] Computing Ragas metrics via LLM judge...")
        result = evaluation.evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy]
        )
        
        df = result.to_pandas()
        print("\n=== EVALUATION RESULTS ===")
        print(df[["question", "faithfulness", "answer_relevancy"]])
        return df

if __name__ == "__main__":
    pipe = MultimodalRAGPipeline()
    evaluator = RAGEvaluator(pipe)
    
    mock_test = [{
        "question": "What is the chunk token size limit?",
        "ground_truth": "The chunk size limit is configured via chunk_size, defaulting to 512 tokens."
    }]
    try:
        evaluator.run_evaluation(mock_test)
    except Exception as e:
        print(f"[EVAL] Skipping execution (database empty or offline): {e}")
