import getpass
import os

from .graph import *


def _set_env(var: str, pwd: str):
    if not os.environ.get(var):
        os.environ[var] = pwd  # getpass.getpass(f"{var}: ")


_set_env("OPENAI_API_KEY", "YOUR API KEY")
# To visualize the algorithm
trace = False
if trace:
    _set_env("LANGSMITH_TRACING", "true")
    _set_env("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    _set_env("LANGSMITH_PROJECT", "pr-fixed-bottom-13")
    _set_env("LANGSMITH_API_KEY", "YOUR API KEY")
    os.environ["LANGSMITH_PROJECT"] = "ToT Tutorial"


# Create the graph
builder = StateGraph(state_schema=ToTState, config_schema=Configuration)

# Add nodes
builder.add_node(plan)
builder.add_node(execute)
builder.add_node(review)
builder.add_node(aggregate)


# Add edges
builder.add_edge("plan", "execute")
builder.add_edge("execute", "review")
builder.add_edge("review", "aggregate")
builder.add_conditional_edges(
    "aggregate",
    should_terminate,
    path_map=[
        "plan",
        "__end__"])

# Set entry point
builder.add_edge("__start__", "plan")

# Compile the graph
graph = builder.compile(checkpointer=MemorySaver())


config = {
    "configurable": {
        "thread_id": "test_1",
        "max_depth": 10,
    }
}

# puzzles = load_data()
counter = 1
for step in graph.stream(
    {
        "problem": "Substance X reacts violently with liquid Y with the release of a gas W whose molecule contains the same number of neutrons and protons, and a precipitate G forms, which, when heated, releases B. The melting point of B (under normal conditions) is very close to 277 K. The product of the reaction of a certain keto acid with the substance X contains 2 atoms of oxygen. The substance X and especially its very close analog is used as a reagent in organic chemistry. Indicate the sum of the atomic weights of the lightest and heaviest elements in the substance X."
    },
    config,
):
    print("step", counter, step)
    counter += 1

final_state = graph.get_state(config)
winning_solution = final_state.values["solution"][0]
search_depth = final_state.values["depth"]
if final_state.values["solved"][0]:
    print(
        f"Found a winning solution in {search_depth} steps: {winning_solution}")
else:
    print(
        f"Failed to find a winning solution in {search_depth} steps. Best guess: {winning_solution}")
