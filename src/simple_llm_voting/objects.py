from typing import List
from pydantic import BaseModel, Field, field_validator

class Generation(BaseModel):
    """The output of the reasoning agent."""
    reasoning: str = Field(
        description="The reasoning behind the solution."
    )
    solution: str = Field(
        description="The final solution to the problem"
    )

    @field_validator("solution", mode="before")
    def convert_solution_to_string(cls, value):
        # Convert integers to strings
        if isinstance(value, int):
            return str(value)
        return value

class Vote(BaseModel):
    """Submit the vote of the proposed reasonings and solutions."""

    preference: List[int] = Field(
        description="Rank your prefereces of the generations in a list of indices. The most preferred appears first."
    )