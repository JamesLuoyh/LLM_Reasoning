from typing import List

from pydantic import BaseModel, Field, field_validator


class Generation(BaseModel):
    """The output of the reasoning agent."""

    scratch: str = Field(
        description="Use this space for making drafts or providing additional comments.")
    reasoning: str = Field(
        description="The reasoning behind the solution. There is no limit to the reasoning steps. Think as thoroughly as possible."
    )
    solution: str = Field(
        description="The final solution to the problem. We use automatic evaluation so your answer key should be as simple and standard as possible for the purpose of matching."
    )

    @field_validator("solution", mode="before")
    def convert_solution_to_string(cls, value):
        # Convert integers to strings
        if isinstance(value, int):
            return str(value)
        return value


class Vote(BaseModel):
    """Submit the vote of the proposed reasonings and solutions."""

    scratch: str = Field(
        description="Use this space for making drafts or providing additional comments.")
    explanation: str = Field(
        description="Show your explanation for why the following preference rank is given.")
    preference: List[int] = Field(
        description="Rank your prefereces of the generations in a list of indices. The most preferred appears first."
    )


class Verification(BaseModel):
    """Submit the vote of the proposed reasonings and solutions."""

    scratch: str = Field(
        description="Use this space for making drafts or providing additional comments.")
    verification_steps: str = Field(
        description="The verification steps for verifying the solution.")

    correct: bool = Field(
        description="If the solution is correct, return True. Otherwise, return False.")
