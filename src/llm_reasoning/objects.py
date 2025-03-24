from typing import Any, List, Literal, NamedTuple, Optional, Union

from pydantic import BaseModel, Field


# the plan is a list of reasoning steps to be taken to solve the problem
class Plan(BaseModel):
    """Submit a Plan to be executed. The plan is a list of reasoning steps to be taken to solve the problem"""

    draft: str = Field(
        description="This is a place for drafting the plan before submitting.")

    reasoning: str = Field(
        description="The reasoning behind the submitted plan. Explain how you arrived at the plan.")

    plan_items: List[str] = Field(
        description="The plan to be submitted for execution, itemized into digestable chunks in a list of strings."
    )


class ExecutionStep(BaseModel):
    """Submit an execution of a planned step."""

    draft: str = Field(
        description="This is a place for drafting the plan execution before submitting.")

    reasoning: str = Field(
        description="The reasoning behind the submitted execution. Explain step-by-step of the execution."
    )

    execution: str = Field(
        description="The final output of the execution for the planned step.")

    solved: bool = Field(
        description="Set the boolean to true if the entirely problem is solved correctly.")

    final_answer: str = Field(
        description="Output the final answer for the entirely problem.")


class Evaluation(BaseModel):
    """Submit the evaluation of the proposed plan and the evaluation of the executed steps of the plan."""

    reasoning: str = Field(
        description="The reasoning behind the submitted evaluations. Explain how you arrived at these evaluations."
    )

    rank: List[int] = Field(
        description="Rank your prefereces of the plans and their associated initial step executions in a list of indices. The most preferred appears first."
    )
