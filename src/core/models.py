from langchain_ollama import ChatOllama

from .types import BaseModel, MessageList


class Llama3(BaseModel):
    def __init__(self, temperature: float = 0.5, num_predict: int = 1000):
        self.model = ChatOllama(
            model="llama3",
            temperature=temperature,
            num_predict=num_predict)

    def __call__(self, message_list: MessageList) -> str:
        return self.model.invoke(message_list)
