import operator
from typing import Optional, Dict, Any
from typing_extensions import Annotated, TypedDict
from langgraph.graph import StateGraph

from langchain_core.runnables import RunnableConfig
from langgraph.constants import Send
from langgraph.checkpoint.memory import MemorySaver
from typing import List, Literal, Union, NamedTuple, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from objects import *
from langchain_ollama import OllamaLLM

from agents import *
import numpy as np

def update(
    existing: Optional[list] = None,
    updates: Optional[Union[list, Literal["clear"]]] = None,
) -> List[list]:
    if existing is None:
        existing = []
    if updates is None:
        return existing
    if updates == "clear":
        return []
    # Concatenate the lists
    return existing + updates


def updateSolved(
    existing: Optional[list] = None,
    updates: Optional[Union[list, Literal["clear"]]] = None,
) -> List[bool]:
    if existing is None:
        existing = []
    if updates is None:
        return existing
    if updates == "clear":
        return []
    # Concatenate the lists
    return existing + updates


class ToTState(TypedDict):
    problem: str
    plans: List[List[str]]
    historical_plans: List[List[str]]
    executed_steps: List[List[str]]
    ranks: List[int]
    depth: Annotated[int, operator.add]
    solved: List[bool]
    solution: List[Any]
    evaluated_ranks: List[int]


class Configuration(TypedDict):
    max_depth: int
    beam_size: int
    n_reviewers: int


def _ensure_configurable(config: RunnableConfig) -> Configuration:
    """Get params that configure the search algorithm."""
    configurable = config.get("configurable", {})
    return {
        **configurable,
        "max_depth": configurable.get("max_depth", 10),
        "beam_size": configurable.get("beam_size", 3),
        "n_reviewers": configurable.get("n_reviewers", 3),
    }




def plan(state: ToTState, *, config: RunnableConfig) -> Dict[str, Any]:
    """Generate the next state."""
    configurable = _ensure_configurable(config)

    plans = []
    print(state["problem"])
    for i in range(configurable["beam_size"]):
        # try:
        plan_submission = None
        while plan_submission is None:
            try:
                plan_submission = planner().invoke(
                    {
                        "problem": state["problem"],
                        "executed_steps": state.get("executed_steps", [""] * configurable["beam_size"])[i],
                        "historical_plans": state.get("historical_plans", [""] * configurable["beam_size"])[i],
                        "k": configurable["max_depth"] - state.get("depth", 0),
                    },
                    config=config,
                )
            except Exception as e:
                print(f"plan() Exception: {e}")
            print("plan():", i, plan_submission)
        plans.append(plan_submission.plan_items)
        

    return {"plans": plans}

def execute(state: ToTState, *, config: RunnableConfig) -> Dict[str, Any]:
    """Generate the next state."""
    configurable = _ensure_configurable(config)

    executed_steps = state.get("executed_steps", [""] * len(state["plans"]))
    solved = []
    solution = []
    for i in range(len(state["plans"])):
        execute_submission = None
        while execute_submission is None:
            try:
                execute_submission = executer().invoke(
                    {
                        "problem": state["problem"],
                        "executed_steps": executed_steps[i],
                        "plan": state["plans"][i],
                    },
                    config=config,
                )
            except Exception as e:
                print(f"execute() Exception: {e}")
            print("execute()", i, execute_submission)
        # append the new executed step to the previous executed steps
        executed_steps[i] += execute_submission.execution
        solved.append(execute_submission.solved)
        solution.append(execute_submission.final_answer)
    return {"executed_steps": executed_steps, "solved": solved}


def review(state: ToTState, *, config: RunnableConfig) -> Dict[str, Any]:
    """Generate the next state."""
    configurable = _ensure_configurable(config)

    ranks = []
    assert(len(state["plans"]) == len(state["executed_steps"]))
    print("review", state["problem"])
    for i in range(configurable["n_reviewers"]):
        review_submission = None
        while review_submission is None or sorted(review_submission.rank) != list(range(len(state["plans"]))):
            try:
                review_submission = reviewer().invoke(
                    {
                        "problem": state["problem"],
                        "executed_steps": state["executed_steps"],
                        "plans": state["plans"],
                        "solved": state["solved"]
                    },
                    config=config,
                )
            except Exception as e:
                print(f"review() Exception: {e}")
            print("review()", i, review_submission)
        ranks.append(review_submission.rank)
    
    # Only keep the first one for now.
    return {"evaluated_ranks": ranks}


def aggregate(
    state: ToTState, *, config: RunnableConfig
) -> Dict[str, Any]:
    # Here to implement the aggregation strategy
    scores = np.zeros(len(state["plans"]))
    reviews = np.array(state["evaluated_ranks"], dtype=np.int32)
    for i in range(len(state["plans"])):
        scores[reviews[:, i]] += len(state["plans"]) - i

    argmax = np.argmax(scores)

    # only keep the first one for now
    historical_plans = state.get("historical_plans", [[]] * len(state["plans"]))
    historical_plans[argmax].append(state["plans"][argmax][0])
    
    configurable = _ensure_configurable(config)

    return {
        # Update the starting point for the next iteration
        "historical_plans": [historical_plans[argmax]] * configurable["beam_size"],
        # Clear the old memory
        "executed_steps": [state["executed_steps"][argmax]] * configurable["beam_size"],
        # Increment the depth by 1
        "depth": 1,
    }


def should_terminate(
    state: ToTState, config: RunnableConfig
) -> Union[Literal["__end__"], Send]:
    configurable = _ensure_configurable(config)
    solved = state["solved"][0]
    if solved or state["depth"] >= configurable["max_depth"]:
        return "__end__"
    return Send("plan", {**state})
