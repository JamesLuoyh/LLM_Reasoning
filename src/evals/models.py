import os
from functools import partial
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

import llm_aggregation.graph as aggregation_graph
from evals.objects import (
    GEMINI2_FLASH,
    GEMINI_API_KEY,
    AggregatedSolutionWrapper,
    Answer,
    LanguageModel,
    MessageList,
    count_tokens,
)
from llm_aggregation.objects import Generation
from llm_reasoning.graph import *


def set_langsmith_env():
    def _set_env(var: str, pwd: str):
        if not os.environ.get(var):
            os.environ[var] = pwd

    _set_env("LANGSMITH_TRACING", "true")
    _set_env("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    _set_env("LANGSMITH_PROJECT", "pr-fixed-bottom-13")
    _set_env("LANGSMITH_API_KEY", "YOUR API KEY")


def _set_env(var: str, pwd: str):
    if not os.environ.get(var):
        os.environ[var] = pwd  # getpass.getpass(f"{var}: ")


os.environ["LANGSMITH_PROJECT"] = "ToT Tutorial"


class Gemini2_flash(LanguageModel):
    def __init__(self, temperature: float = 0.7, num_predict: int = 2048,
                 structured: bool = True, max_retries=2):
        self.model = ChatGoogleGenerativeAI(
            model=GEMINI2_FLASH,
            temperature=temperature,
            max_tokens=None,
            timeout=None,
            max_retries=max_retries,
            google_api_key=GEMINI_API_KEY,
        )
        self.structured = structured
        self.max_retries = max_retries

    def __call__(self, message_list: MessageList) -> str:
        if self.structured:
            bound_llm = self.model.with_structured_output(Generation)
            retries = 0
            generation = None
            while generation is None and retries < self.max_retries:
                generation = bound_llm.invoke(message_list)
                retries += 1
            output_tokens = count_tokens(
                [generation.solution, generation.reasoning, generation.scratch])
            input_tokens = count_tokens(
                [item for m in message_list for item in m.values()])
            return AggregatedSolutionWrapper(
                generation.solution, output_tokens=output_tokens, input_tokens=input_tokens
            )

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

        # To visualize the algorithm
        if trace:
            set_langsmith_env()

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
                "thread_id": "tree_of_thought",
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


class SelfConsistency(LanguageModel):
    def __init__(
        self,
        model_type: str,
        temperature: float = 0.7,
        num_predict: int = 2048,
        allow_duplicates: bool = True,
        trace: bool = False,
        debug: bool = False,
        n_generators: int = 5,
    ):

        self.llm = self.select_model_type(model_type, temperature, num_predict)
        self.debug = debug

        if not allow_duplicates:
            equality_checker = Gemini2_flash(temperature, structured=False)
        else:
            equality_checker = None

        # To visualize the algorithm
        if trace:
            set_langsmith_env()

        self.builder = StateGraph(
            state_schema=aggregation_graph.State,
            config_schema=aggregation_graph.Configuration)
        # Add nodes
        self.builder.add_node(
            "generate",
            partial(
                aggregation_graph.generate,
                llm=self.llm,
                debug=self.debug))
        self.builder.add_node(
            "aggregate", partial(
                aggregation_graph.self_consistency))

        # Add edges
        self.builder.add_edge("generate", "aggregate")
        self.builder.add_edge("aggregate", "__end__")

        # Set entry point
        self.builder.add_edge("__start__", "generate")

        # Compile the graph
        self.graph = self.builder.compile(checkpointer=MemorySaver())

        self.config = {
            "configurable": {"thread_id": "self_consistency", "recursion_limit": 100},
            "allow_duplicates": allow_duplicates,
            "equality_checker": equality_checker,
            "n_generators": n_generators,
        }

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

        return AggregatedSolutionWrapper(
            aggregated_solution,
            output_tokens=final_state.values["output_tokens"],
            input_tokens=final_state.values["input_tokens"],
        )

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}


class MajorityVote(LanguageModel):
    def __init__(
        self,
        model_type: str,
        temperature: float = 0.7,
        num_predict: int = 2048,
        allow_duplicates: bool = True,
        trace: bool = False,
        debug: bool = False,
        n_generators: int = 5,
    ):

        self.llm = self.select_model_type(model_type, temperature, num_predict)
        self.debug = debug

        if not allow_duplicates:
            equality_checker = Gemini2_flash(temperature, structured=False)
        else:
            equality_checker = None

        # To visualize the algorithm
        if trace:
            set_langsmith_env()

        self.builder = StateGraph(
            state_schema=aggregation_graph.State,
            config_schema=aggregation_graph.Configuration)
        # Add nodes
        self.builder.add_node(
            "generate",
            partial(
                aggregation_graph.generate,
                llm=self.llm,
                debug=self.debug))
        self.builder.add_node(
            "vote",
            partial(
                aggregation_graph.vote,
                llm=self.llm,
                debug=self.debug))
        self.builder.add_node(
            "aggregate", partial(
                aggregation_graph.majority_vote))

        # Add edges
        self.builder.add_edge("generate", "vote")
        self.builder.add_edge("vote", "aggregate")
        self.builder.add_edge("aggregate", "__end__")

        # Set entry point
        self.builder.add_edge("__start__", "generate")

        # Compile the graph
        self.graph = self.builder.compile(checkpointer=MemorySaver())

        self.config = {
            "configurable": {"thread_id": "majority_vote", "recursion_limit": 100},
            "allow_duplicates": allow_duplicates,
            "equality_checker": equality_checker,
            "n_generators": n_generators,
        }

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
        return AggregatedSolutionWrapper(
            aggregated_solution,
            output_tokens=final_state.values["output_tokens"],
            input_tokens=final_state.values["input_tokens"],
        )

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}


class BordaCount(LanguageModel):
    def __init__(
        self,
        model_type: str,
        temperature: float = 0.7,
        num_predict: int = 2048,
        allow_duplicates: bool = True,
        trace: bool = False,
        debug: bool = False,
        n_generators: int = 5,
        verifier_temperature: float = 0.7,
        n_verifiers: int = 5,
    ):

        self.llm = self.select_model_type(
            model_type, verifier_temperature, num_predict)
        self.debug = debug

        if not allow_duplicates:
            equality_checker = Gemini2_flash(temperature, structured=False)
        else:
            equality_checker = None

        # To visualize the algorithm
        if trace:
            set_langsmith_env()

        self.builder = StateGraph(
            state_schema=aggregation_graph.State,
            config_schema=aggregation_graph.Configuration)
        # Add nodes
        self.builder.add_node(
            "generate",
            partial(
                aggregation_graph.generate,
                llm=self.llm,
                debug=self.debug))
        self.builder.add_node(
            "vote",
            partial(
                aggregation_graph.vote,
                llm=self.llm,
                debug=self.debug))
        self.builder.add_node(
            "aggregate",
            partial(
                aggregation_graph.borda_count,
                debug=self.debug))

        # Add edges
        self.builder.add_edge("generate", "vote")
        self.builder.add_edge("vote", "aggregate")
        self.builder.add_edge("aggregate", "__end__")

        # Set entry point
        self.builder.add_edge("__start__", "generate")

        # Compile the graph
        self.graph = self.builder.compile(checkpointer=MemorySaver())

        self.config = {
            "configurable": {"thread_id": "borda_count", "recursion_limit": 100},
            "allow_duplicates": allow_duplicates,
            "equality_checker": equality_checker,
            "n_generators": n_generators,
            "n_voters": n_verifiers,
        }

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

        return AggregatedSolutionWrapper(
            aggregated_solution,
            output_tokens=final_state.values["output_tokens"],
            input_tokens=final_state.values["input_tokens"],
        )

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}


class BestOfN(LanguageModel):
    def __init__(
        self,
        model_type: str,
        temperature: float = 0.7,
        num_predict: int = 2048,
        allow_duplicates: bool = True,
        trace: bool = False,
        debug: bool = False,
        n_generators: int = 5,
    ):

        self.llm = self.select_model_type(model_type, temperature, num_predict)
        self.debug = debug

        if not allow_duplicates:
            equality_checker = Gemini2_flash(temperature, structured=False)
        else:
            equality_checker = None

        # To visualize the algorithm
        if trace:
            set_langsmith_env()

        self.builder = StateGraph(
            state_schema=aggregation_graph.State,
            config_schema=aggregation_graph.Configuration)
        # Add nodes
        self.builder.add_node(
            "generate",
            partial(
                aggregation_graph.generate,
                llm=self.llm,
                debug=self.debug))
        self.builder.add_node(
            "vote",
            partial(
                aggregation_graph.vote,
                llm=self.llm,
                debug=self.debug))

        self.builder.add_node(
            "aggregate",
            partial(
                aggregation_graph.best_of_n,
                debug=self.debug))

        # Add edges
        self.builder.add_edge("generate", "vote")
        self.builder.add_edge("vote", "aggregate")
        self.builder.add_edge("aggregate", "__end__")

        # Set entry point
        self.builder.add_edge("__start__", "generate")

        # Compile the graph
        self.graph = self.builder.compile(checkpointer=MemorySaver())

        self.config = {
            "configurable": {"thread_id": "best_of_n", "recursion_limit": 100},
            "n_voters": 1,
            "allow_duplicates": allow_duplicates,
            "equality_checker": equality_checker,
            "n_generators": n_generators,
        }

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

        return AggregatedSolutionWrapper(
            aggregated_solution,
            output_tokens=final_state.values["output_tokens"],
            input_tokens=final_state.values["input_tokens"],
        )

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}


class ScaleVerification(LanguageModel):
    def __init__(
        self,
        model_type: str,
        temperature: float = 0.7,
        num_predict: int = 2048,
        allow_duplicates: bool = True,
        trace: bool = False,
        debug: bool = False,
        n_generators: int = 5,
        verifier_temperature: float = 0.7,
        n_verifiers: int = 5,
    ):
        # TODO make the generator and the verifier different llms
        self.llm = self.select_model_type(
            model_type, verifier_temperature, num_predict)
        self.debug = debug
        # llm = ChatOllama(model="llama3-groq-tool-use")
        # llm = ChatOpenAI(model="gpt-4o-mini")

        if not allow_duplicates:
            equality_checker = Gemini2_flash(temperature, structured=False)
        else:
            equality_checker = None

        # To visualize the algorithm
        if trace:
            set_langsmith_env()

        self.builder = StateGraph(
            state_schema=aggregation_graph.State,
            config_schema=aggregation_graph.Configuration)
        # Add nodes
        self.builder.add_node(
            "generate",
            partial(
                aggregation_graph.generate,
                llm=self.llm,
                debug=self.debug))
        self.builder.add_node(
            "verify",
            partial(
                aggregation_graph.verifications,
                llm=self.llm,
                debug=self.debug))
        self.builder.add_node(
            "aggregate", partial(
                aggregation_graph.aggregate_verifications))

        # Add edges
        self.builder.add_edge("generate", "verify")
        self.builder.add_edge("verify", "aggregate")
        self.builder.add_edge("aggregate", "__end__")

        # Set entry point
        self.builder.add_edge("__start__", "generate")

        # Compile the graph
        self.graph = self.builder.compile(checkpointer=MemorySaver())

        self.config = {
            "configurable": {
                "thread_id": "scale_verification",
                "recursion_limit": 100,
            },
            "allow_duplicates": allow_duplicates,
            "equality_checker": equality_checker,
            "n_generators": n_generators,
            "n_verifiers": n_verifiers,
        }

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

        return AggregatedSolutionWrapper(
            aggregated_solution,
            output_tokens=final_state.values["output_tokens"],
            input_tokens=final_state.values["input_tokens"],
        )

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}
