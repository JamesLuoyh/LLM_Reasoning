from typing import TypedDict, List, Dict, Any
from langchain_core.runnables import RunnableConfig

from simple_llm_voting.agents import generator, voter

class State(TypedDict):
    problem: str 
    reasonings: List[str]
    solutions: List[str]
    preferences: List[int]
    aggregated_solution: str
    
class Configuration(TypedDict):
    n_generators: int
    n_voters: int

def _ensure_configurable(config: RunnableConfig) -> Configuration:
    """Get params that configure the search algorithm."""
    configurable = config.get("configurable", {})
    return {
        **configurable,
        "n_generators": configurable.get("n_generators", 5),
        "n_voters": configurable.get("n_voters", 5),
    }

def generate(state: State, *, config: RunnableConfig, llm: Any, debug: bool = False) -> Dict[str, Any]:

    """Generate the next state."""
    configurable = _ensure_configurable(config)

    reasonings = []
    solutions = []
    if debug: 
        print(state["problem"])
    for i in range(configurable["n_generators"]):
        generation = None 
        while generation is None:
            try: 
                generation = generator(llm).invoke(
                    {
                        "problem": state["problem"],
                    },
                    config=config,
                )
            except Exception as e:
                print(f"generate() Exception: {e}")
            if debug: 
                print("generate():", i, generation)
        reasonings.append(generation.reasoning)
        solutions.append(generation.solution)
    
    return {"reasonings": reasonings, "solutions": solutions}

def vote(state: State, *, config: RunnableConfig, llm: Any, debug: bool = False) -> Dict[str, Any]:
    """Vote on the generated solutions."""
    configurable = _ensure_configurable(config)

    preferences = []
    for i in range(configurable["n_voters"]):
        vote = None 
        while vote is None or sorted(
            vote.preference) != list(range(len(state["solutions"]))):
            try: 
                vote = voter(llm).invoke(
                    {
                        "problem": state["problem"],
                        "reasonings": state["reasonings"],
                        "solutions": state["solutions"],
                    },
                    config=config,
                )
            except Exception as e:
                print(f"vote() Exception: {e}")
            if debug: 
                print("vote():", i, vote)
        preferences.append(vote.preference)

    return {"preferences": preferences}

def majority_vote(state: State) -> Dict[str, Any]:
    """Majority vote on the generated solutions."""

    preferences = state["preferences"]
    n_generators = len(preferences[0])
    votes = [0] * n_generators
    for preference in preferences:
        for i in range(len(preference)):
            votes[preference[i]] += 1
    
    majority_vote_index = votes.index(max(votes))
    return {"aggregated_solution": state["solutions"][majority_vote_index]}
