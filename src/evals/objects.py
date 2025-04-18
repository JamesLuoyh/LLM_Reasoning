# Code adapted from OpenAI (2024)
# Source: https://github.com/openai/simple-evals/
# Licensed under the MIT License

from typing import Any

from pydantic import BaseModel, Field

Message = dict[str, Any]
MessageList = list[Message]


class LanguageModel:
    """
    Base class for language model.
    """

    # Invokes model with message_list
    def __call__(self, message_list: MessageList) -> str:
        raise NotImplementedError


class EvalResult(BaseModel):
    """
    Result of running an evaluation (usually consisting of many samples)
    """

    score: float | None = Field(description="top-line metric")
    metrics: dict[str, float] | None = Field(description="other metrics")
    htmls: list[str] = Field(description="strings of valid html")
    convos: list[MessageList] = Field(description="sampled conversations")


class SingleEvalResult(BaseModel):
    """
    Result of evaluating a single sample
    """

    score: float | None = Field(description="top-line metric")
    metrics: dict[str, float] = Field(
        description="other metrics", default_factory=dict)
    html: str | None = None
    convo: MessageList | None = Field(
        description="sampled conversation", default=None)


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

    def __init__(self, solution: Any):
        self.solution = solution

    @property
    def content(self):
        return self.solution
