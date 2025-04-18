import os
from functools import partial
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

import simple_llm_voting.graph as voting_graph
from evals.objects import AggregatedSolutionWrapper, Answer, LanguageModel, MessageList
from llm_reasoning.graph import *


def _set_env(var: str, pwd: str):
    if not os.environ.get(var):
        os.environ[var] = pwd


class Gemini2_flash(LanguageModel):
    def __init__(self, temperature: float = 1.5,
                 num_predict: int = 2048, structured: bool = False):
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-001",
            temperature=temperature,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            google_api_key="Your API Key",
            # other params...
        )
        self.structured = structured

    def __call__(self, message_list: MessageList) -> str:
        if self.structured:
            bound_llm = self.model.with_structured_output(Answer)

            return bound_llm.invoke(message_list)

        return self.model.invoke(message_list)

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}


class Llama3(LanguageModel):

    def __init__(self, temperature: float = 0.7,
                 num_predict: int = 2048, structured: bool = False):
        self.model = ChatOllama(
            model="llama3.1",
            temperature=temperature,
            num_predict=num_predict)
        self.structured = structured

    def __call__(self, message_list: MessageList) -> str:
        if self.structured:
            bound_llm = self.model.with_structured_output(Answer)

            return bound_llm.invoke(message_list)

        return self.model.invoke(message_list)

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}


class ToT(LanguageModel):
    def __init__(self, temperature: float = 0.7, trace=False):

        self.llm = ChatOllama(model="llama3.1")
        # llm = ChatOllama(model="llama3-groq-tool-use")
        # llm = ChatOpenAI(model="gpt-4o-mini")

        # getpass.getpass(f"{var}: ")

        # _set_env("OPENAI_API_KEY", "YOUR API KEY")
        # To visualize the algorithm
        if trace:
            _set_env("LANGSMITH_TRACING", "true")
            _set_env("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
            _set_env("LANGSMITH_PROJECT", "pr-fixed-bottom-13")
            _set_env("LANGSMITH_API_KEY", "YOUR API KEY")
            os.environ["LANGSMITH_PROJECT"] = "ToT Tutorial"

        self.builder = StateGraph(
            state_schema=ToTState,
            config_schema=Configuration)
        # Add nodes
        self.builder.add_node("plan", partial(plan, llm=self.llm))
        self.builder.add_node("execute", partial(execute, llm=self.llm))
        self.builder.add_node("review", partial(review, llm=self.llm))
        self.builder.add_node(aggregate)

        # Add edges
        self.builder.add_edge("plan", "execute")
        self.builder.add_edge("execute", "review")
        self.builder.add_edge("review", "aggregate")
        self.builder.add_conditional_edges(
            "aggregate", should_terminate, path_map=[
                "plan", "__end__"])

        # Set entry point
        self.builder.add_edge("__start__", "plan")

        # Compile the graph
        self.graph = self.builder.compile(checkpointer=MemorySaver())

        self.config = {
            "configurable": {
                "thread_id": "test_1",
                "max_depth": 10,
                "recursion_limit": 100}}

    def __call__(self, message_list: MessageList) -> str:
        counter = 1
        assert len(message_list) == 1 and message_list[0]["content"]
        for step in self.graph.invoke(
                {"problem": message_list[0]["content"]}, config=self.config):
            print("step", counter, step)
            counter += 1

        final_state = self.graph.get_state(self.config)
        winning_solution = final_state.values["solution"][0]
        search_depth = final_state.values["depth"]
        if final_state.values["solved"][0]:
            print(
                f"Found a winning solution in {search_depth} steps: {winning_solution}")
        else:
            print(
                f"Failed to find a winning solution in {search_depth} steps. Best guess: {winning_solution}")
        return winning_solution

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}


class VoteLLM(LanguageModel):
    def __init__(self, temperature: float = 0.7, num_predict: int = 2048,
                 trace: bool = False, debug: bool = False):

        self.llm = ChatOllama(
            model="llama3.1",
            temperature=temperature,
            num_predict=num_predict)
        self.debug = debug
        # llm = ChatOllama(model="llama3-groq-tool-use")
        # llm = ChatOpenAI(model="gpt-4o-mini")

        def _set_env(var: str, pwd: str):
            if not os.environ.get(var):
                os.environ[var] = pwd  # getpass.getpass(f"{var}: ")

        # _set_env("OPENAI_API_KEY", "YOUR API KEY")
        # To visualize the algorithm
        if trace:
            _set_env("LANGSMITH_TRACING", "true")
            _set_env("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
            _set_env("LANGSMITH_PROJECT", "pr-fixed-bottom-13")
            _set_env("LANGSMITH_API_KEY", "YOUR API KEY")
            os.environ["LANGSMITH_PROJECT"] = "ToT Tutorial"

        self.builder = StateGraph(
            state_schema=voting_graph.State,
            config_schema=voting_graph.Configuration)
        # Add nodes
        self.builder.add_node(
            "generate",
            partial(
                voting_graph.generate,
                llm=self.llm,
                debug=self.debug))
        self.builder.add_node(
            "vote",
            partial(
                voting_graph.vote,
                llm=self.llm,
                debug=self.debug))
        self.builder.add_node("aggregate", partial(voting_graph.majority_vote))

        # Add edges
        self.builder.add_edge("generate", "vote")
        self.builder.add_edge("vote", "aggregate")
        self.builder.add_edge("aggregate", "__end__")

        # Set entry point
        self.builder.add_edge("__start__", "generate")

        # Compile the graph
        self.graph = self.builder.compile(checkpointer=MemorySaver())

        self.config = {
            "configurable": {
                "thread_id": "test_1",
                "recursion_limit": 100}}

    def __call__(self, message_list: MessageList) -> str:
        counter = 1
        assert len(message_list) == 1 and message_list[0]["content"]
        for step in self.graph.invoke(
                {"problem": message_list[0]["content"]}, config=self.config):
            if self.debug:
                print("step", counter, step)
            counter += 1

        final_state = self.graph.get_state(self.config)
        aggregated_solution = final_state.values["aggregated_solution"]
        if self.debug:
            print(f"Found a solution: {aggregated_solution}")

        aggregated_solution = AggregatedSolutionWrapper(aggregated_solution)
        return aggregated_solution

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}
