import operator
from typing import Any, Dict, List, Literal, NamedTuple, Optional, Union

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import Send
from langgraph.graph import StateGraph
from typing_extensions import Annotated, TypedDict

from llm_reasoning.objects import *


def planner(llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a planning agent for solving challenging reasoning problems. You will be provided a problem. "
                "Usually, you will also be provided some initial steps that other solver has attempted so far. "
                "Your goal is to come up with a plan following the executed steps so far so that the solution can be found in "
                "as few steps as possible. Each step needs to be high digestible so that other solvers who pick it up can finish "
                "the step with little ambiguity. Aware that the previous executed steps may not always be correct so your plan can also "
                "be exploring new approaches. There is a limit to the maximum number of steps you can generate.",
            ),
            (
                "user",
                "Generate the plan following the previous executed steps for solving the problem: {problem}. Previous planned steps: {historical_plans}."
                " Previous steps executed:{executed_steps}. You may propose up to {k} steps. ",
            ),
        ],
    ).partial(candidate="")

    bound_llm = llm.with_structured_output(Plan)
    return prompt | bound_llm


def executer(llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a problem solving agent for tackling challenging reasoning problems. You will be provided a problem "
                "and a specified plan for solving the problem. Usually, you will also be provided some initial steps that other solver has executed so far. "
                "Your goal is to execute the first step in the plan exactly as specified. After that, identify whether the executed step also solves the entire problem. "
                "If the entire problem is solved, output your final answer as an integer.",
            ),
            (
                "user",
                "Execute the first step specified in the plan following the provided steps already executed. The problem: {problem}. "
                "The executed steps: {executed_steps}. The plan that follows the executed steps has the following items: {plan}.",
            ),
        ],
    )

    bound_llm = llm.with_structured_output(ExecutionStep)
    return prompt | bound_llm


def reviewer(llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a reviewer of multiple problem solving agents for tackling challenging reasoning problems. You will be provided a problem "
                "and multiple specified plans for solving the problem. Usually, you will also be provided some initial steps that other solver has attempted so far corresponding to each plan. "
                "Your goal is to first examine the plans and the initial executions and judge how likely each plan is on the right track to solving the problem. "
                "Some of the atempts may claim that the problem is solved. In this case, evaluate whether they have solved it correctly. If not, they should be placed to the end of the list. "
                "You will then be asked to generate a ranked list of the quality of the plans from your most preferred to your least preferred using the 0-based indices. The most preferred comes first.",
            ),
            (
                "user",
                "Generate a ranked list for the plans and their corresponding executions. Problem: {problem}. Plans: {plans}. Executions: {executed_steps}. Solved: {solved}. "
                "The i-th plan corresponds to the i-th execution and the solved indicators.",
            ),
        ],
    )

    bound_llm = llm.with_structured_output(Evaluation)
    return prompt | bound_llm
