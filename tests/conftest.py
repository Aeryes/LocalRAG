import pytest
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

class LocalOllamaJudge(DeepEvalBaseLLM):
    def __init__(self, model_name="llama3:8b"):
        self.model_name = model_name
        self.llm = ChatOllama(
            model=model_name,
            base_url="http://ollama:11434",
            temperature=0
        )

    def load_model(self):
        return self.llm

    def generate(self, prompt: str) -> str:
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content

    async def a_generate(self, prompt: str) -> str:
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return response.content

    def get_model_name(self):
        return self.model_name

@pytest.fixture
def local_judge():
    return LocalOllamaJudge()