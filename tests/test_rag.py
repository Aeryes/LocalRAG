import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from src.app import app  # Import your actual RAG Agent

TEST_CASES = [
    {
        "input": "What is in the requirements.txt file?",
        "expected_context": ["langchain", "qdrant-client", "streamlit", "arize-phoenix"]
    },
    {
        "input": "What database are we using?",
        "expected_context": ["Qdrant", "vector database", "docker container"]
    }
]


@pytest.mark.parametrize("case", TEST_CASES)
def test_rag_accuracy(case, local_judge):
    """
    Runs the RAG pipeline and asserts that the answer is:
    1. Faithful (Not a hallucination)
    2. Relevant (Answers the question)
    """

    result = app.invoke({"question": case["input"], "hallucination_count": 0})

    actual_output = result["generation"]
    retrieved_context = result["documents"]

    print(f"\nQUERY: {case['input']}")
    print(f"OUTPUT: {actual_output}")

    test_case = LLMTestCase(
        input=case["input"],
        actual_output=actual_output,
        retrieval_context=retrieved_context
    )

    faithfulness = FaithfulnessMetric(
        threshold=0.5,
        model=local_judge,
        include_reason=True
    )
    relevancy = AnswerRelevancyMetric(
        threshold=0.5,
        model=local_judge,
        include_reason=True
    )

    assert_test(test_case, [faithfulness, relevancy])