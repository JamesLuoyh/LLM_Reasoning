from typing import Any, Dict, List, TypedDict

import numpy as np
from langchain_core.runnables import RunnableConfig

from evals.common import is_equiv
from evals.constants import INVALID_ANS
from llm_aggregation.agents import generator, verifier, voter


class State(TypedDict):
    problem: str
    reasonings: List[str]
    solutions: List[str]
    preferences: List[List[int]]
    aggregated_solution: str
    scores: List[int]


class Configuration(TypedDict):
    n_generators: int
    n_voters: int
    n_verifiers: int


def _ensure_configurable(config: RunnableConfig) -> Configuration:
    """Get params that configure the search algorithm."""
    configurable = config.get("configurable", {})
    return {
        **configurable,
        "n_generators": configurable.get("n_generators", 5),
        "n_voters": configurable.get("n_voters", 5),
        "n_verifiers": configurable.get("n_verifiers", 5),
        "retries": configurable.get("retries", 5),
    }


def generate(state: State, *, config: RunnableConfig,
             llm: Any, debug: bool = False) -> Dict[str, Any]:
    """Generate the next state."""
    configurable = _ensure_configurable(config)

    reasonings = []
    solutions = []
    if debug:
        print(state["problem"])
    for i in range(configurable["n_generators"]):
        generation = None
        retries = 0
        while generation is None and retries < configurable["retries"]:
            try:
                generation = generator(llm).invoke(
                    {
                        "problem": state["problem"],
                    },
                    config=config,
                )
            except Exception as e:
                print(f"generate() Exception: {e}")
            retries += 1
            if debug:
                print("generate():", i, generation)
        if generation:
            reasonings.append(generation.reasoning)
            solutions.append(generation.solution)
        else:
            reasonings.append(INVALID_ANS)
            solutions.append(INVALID_ANS)

    return {"reasonings": reasonings, "solutions": solutions}


def vote(state: State, *, config: RunnableConfig,
         llm: Any, debug: bool = False) -> Dict[str, Any]:
    """Vote on the generated solutions."""
    configurable = _ensure_configurable(config)

    preferences = []
    for i in range(configurable["n_voters"]):
        vote = None
        retries = 0
        while (
            vote is None or sorted(vote.preference) != list(
                range(len(state["solutions"])))
        ) and retries < configurable["retries"]:
            try:
                vote = voter(llm).invoke(
                    {
                        "problem": state["problem"],
                        "reasonings": state["reasonings"],
                        "solutions": state["solutions"],
                        "n_generators": configurable["n_generators"],
                    },
                    config=config,
                )
            except Exception as e:
                print(f"vote() Exception: {e}")
            retries += 1
            if debug:
                print("vote():", i, vote)
        if vote:
            preferences.append(vote.preference)
        else:
            preferences.append(None)

    return {"preferences": preferences}


def majority_vote(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Majority vote on the generated solutions."""
    configurable = _ensure_configurable(config)

    preferences = state["preferences"]
    if not preferences:
        return {"aggregated_solution": ""}

    n_generators = configurable["n_generators"]
    votes = [0] * n_generators
    for preference in preferences:
        if preference:
            votes[preference[0]] += 1

    majority_vote_index = votes.index(max(votes))
    return {"aggregated_solution": state["solutions"][majority_vote_index]}


def borda_count(state: State, config: RunnableConfig,
                debug: bool = False) -> Dict[str, Any]:
    "Borda count on the generated solutions."

    configurable = _ensure_configurable(config)
    preferences = state["preferences"]
    print("preference: ", preferences)
    if not preferences:
        return {"aggregated_solution": ""}

    n_generators = configurable["n_generators"]
    print("n_generators ", n_generators)
    borda_counts = {i: 0 for i in range(n_generators)}
    for preference in preferences:
        if preference:
            for i, rank in enumerate(preference):
                borda_counts[rank] += n_generators - i - 1

    # Find the solution with the highest Borda count
    max_key = max(borda_counts, key=borda_counts.get)

    if debug:
        print("preferences: ", preferences)
        print("borda_count: ", borda_counts)
        print("max_key: ", max_key)
        print("aggregated_solution: ", state["solutions"][max_key])
    return {"aggregated_solution": state["solutions"][max_key]}


def best_of_n(state: State, debug: bool = False) -> Dict[str, Any]:
    """Best of n on the generated solutions."""
    assert len(state["preferences"]) == 1

    if not state["preferences"]:
        return {"aggregated_solution": ""}

    best_of_n_index = state["preferences"][0][0]

    if debug:
        print("best_of_n: ", best_of_n_index)
        print("aggregated_solution: ", state["solutions"][best_of_n_index])

    return {"aggregated_solution": state["solutions"][best_of_n_index]}


def self_consistency(state: State) -> Dict[str, Any]:
    """Output the most common final answer among generators."""
    solutions = state["solutions"]
    if not solutions or len(solutions) == 0:
        return {"aggregated_solution": ""}
    scores = {}
    max_count = 1
    max_key = solutions[0]
    for solution in solutions:
        matched = False
        for key in scores.keys():
            matched = is_equiv(key, solution)
            if matched:
                scores[key] += 1
                max_count = max(max_count, scores[key])
                if scores[key] == max_count:
                    max_key = key
                break
        if not matched:
            scores[solution] = 1
    return {"aggregated_solution": max_key}


def verifications(state: State, *, config: RunnableConfig,
                  llm: Any, debug: bool = False) -> Dict[str, Any]:
    "Verify candidate solutions one by one"
    solutions = state["solutions"]
    configurable = _ensure_configurable(config)
    scores = [0] * len(solutions)
    for i, solution in enumerate(solutions):
        for j in range(configurable["n_verifiers"]):
            verification = None
            retries = 0
            while verification is None and retries < configurable["retries"]:
                try:
                    verification = verifier(llm).invoke(
                        {"problem": state["problem"],
                         "reasoning": state["reasonings"][i],
                            "solution": solution},
                        config=config,
                    )
                except Exception as e:
                    print(f"verifications() Exception: {e}")
                retries += 1
                if debug:
                    print("verifications():", i, verification)
            if verification and verification.correct:
                scores[i] += 1
    return {"scores": scores}


def aggregate_verifications(state: State) -> Dict[str, Any]:
    "Output the solution with the highest score. Tie-break if necessary."
    scores = state["scores"]
    if not scores:
        return {"aggregated_solution": ""}
    # TODO: tie breaking answers with similar scores
    return {"aggregated_solution": state["solutions"][np.argmax(scores)]}
