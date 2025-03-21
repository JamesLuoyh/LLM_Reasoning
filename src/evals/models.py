from typing import Any

from langchain_ollama import ChatOllama

from evals.objects import Answer, LanguageModel, MessageList


class Llama3(LanguageModel):

    def __init__(self, temperature: float = 0.7,
                 num_predict: int = 2048, structured: bool = False):
        self.model = ChatOllama(
            model="llama3.1",
            temperature=temperature,
            num_predict=num_predict)
        self.structured = structured

    def __call__(self, message_list: MessageList) -> str:
        if self.structured:
            bound_llm = self.model.with_structured_output(Answer)

            return bound_llm.invoke(message_list)

        return self.model.invoke(message_list)

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}
