# Code adapted from OpenAI (2024)
# Source: https://github.com/openai/simple-evals/
# Licensed under the MIT License

from typing import Any

from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

Message = dict[str, Any]
MessageList = list[Message]

GEMINI_API_KEY = ""
GEMINI2_FLASH = "gemini-2.0-flash-001"


def count_tokens(contents: list):
    client = genai.Client(api_key=GEMINI_API_KEY)
    return client.models.count_tokens(
        model=GEMINI2_FLASH, contents=contents).total_tokens


class LanguageModel:
    """
    Base class for language model.
    """

    # Invokes model with message_list
    def __call__(self, message_list: MessageList) -> str:
        raise NotImplementedError

    def select_model_type(self, model_type: str,
                          temperature: float, num_predict: int) -> Any:
        if model_type == "llama3":
            return ChatOllama(model="llama3.1",
                              temperature=temperature, num_predict=num_predict)
        elif model_type == "gemini2_flash":
            return ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=temperature,
                max_tokens=num_predict,
                timeout=None,
                max_retries=2,
                google_api_key=GEMINI_API_KEY,
            )


class EvalResult(BaseModel):
    """
    Result of running an evaluation (usually consisting of many samples)
    """

    score: list[float] | None = Field(description="top-line metric")
    metrics: dict[str, float] | None = Field(description="other metrics")
    htmls: list[str] = Field(description="strings of valid html")
    convos: list[MessageList] = Field(description="sampled conversations")
    input_tokens: float | None = Field(
        description="number of input prompt tokens in total")
    output_tokens: float | None = Field(
        description="number of output prompt tokens in total")


class SingleEvalResult(BaseModel):
    """
    Result of evaluating a single sample
    """

    score: list[float] | None = Field(description="top-line metric")
    metrics: dict[str, float] = Field(
        description="other metrics", default_factory=dict)
    html: str | None = None
    convo: MessageList | None = Field(
        description="sampled conversation", default=None)
    input_tokens: int | None = Field(
        description="number of input prompt tokens in total")
    output_tokens: int | None = Field(
        description="number of output prompt tokens in total")


class Answer(BaseModel):
    """
    Answer structure
    """

    answer: str = Field(description="The answer to the question")


class Eval:
    """
    Base class for defining an evaluation.
    """

    def __call__(self, model: LanguageModel) -> EvalResult:
        raise NotImplementedError


class AggregatedSolutionWrapper:
    """
    Wrapper for the aggregated solution so it can be called in evals
    """

    def __init__(self, solution: Any, input_tokens: int, output_tokens: int):
        self.solution = solution
        self.in_tokens = input_tokens
        self.out_tokens = output_tokens

    @property
    def content(self):
        return self.solution

    @property
    def input_tokens(self):
        return self.in_tokens

    @property
    def output_tokens(self):
        return self.out_tokens
