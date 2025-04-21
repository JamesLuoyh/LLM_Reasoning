from langchain_core.prompts import ChatPromptTemplate

from simple_llm_voting.objects import Generation, Verification, Vote


def generator(llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a reasoning agent for solving challenging reasoning problems. You will be provided a problem. "
                "Your goal is to come up with a reasoning and a solution for the problem. "
                "The reasoning should be highly digestible so that other solvers who pick it up can understand it with little ambiguity. "
                "Your reasoning should be detailed but concise. "
                "You must return your response in JSON format with the following fields: "
                "- reasoning: a string containing your entire reasoning process. "
                "- solution: a string that contains the final solution to the problem, optionally formatted using LaTeX syntax. "
                "Here is an example of the required JSON format: "
                "{{"
                '  "reasoning": "First, I identified the numbers. Then, I added them together.", '
                '  "solution": "4"'
                "}}",
            ),
            (
                "user",
                "Solve the following problem step by step. "
                "Problem: {problem}. "
                "Your response should be in JSON format with the following fields: "
                "{{"
                '  "reasoning": "First, I identified the numbers. Then, I added them together.", '
                '  "solution": "4"'
                "}}. "
                "Solutions that include \\boxed will be considered invalid.",
            ),
        ],
    )

    bound_llm = llm.with_structured_output(Generation)
    return prompt | bound_llm


def voter(llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a reviewer of multiple problem solving agents for tackling challenging reasoning problems. You will be provided a problem "
                "and multiple specified reasonings and final solutions for solving the problem. "
                "Evaluate whether they have solved it correctly. If not, they should be placed to the end of the list. "
                "You will then be asked to generate a ranked list of the quality of the reasoning and solutions "
                "from your most preferred to your least preferred using the 0-based indices. The most preferred comes first. "
                "When ranking, consider the following criteria: "
                "- Correctness: Is the solution correct and does the reasoning logically lead to the solution? "
                "- Clarity: Is the reasoning easy to understand and free of ambiguity? "
                "- Logical Coherence: Does the reasoning follow a clear and logical progression? "
                "Provide a justification for each ranking based on these criteria.",
            ),
            (
                "user",
                "Evaluate the generated reasonings and solutions. Problem: {problem}. Reasonings: {reasonings}. Solutions: {solutions}. "
                "The i-th reasoning corresponds to the i-th solution. REMEMBER, 0-BASED INDEXING. Make sure the length is of size {n_generators}. "
                "For each ranking, explain why you ranked it in that position based on correctness, clarity, and logical coherence.",
            ),
        ],
    )

    bound_llm = llm.with_structured_output(Vote)
    return prompt | bound_llm


def verifier(llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                'An examiner has presented a math problem along with a "candidate solution". Your purpose is to assist in evaluating whether the *final answer* in the candidate solution is correct. ',
            ),
            (
                "user",
                "We will guide you through a series of exercises to fulfill this purpose. The question is a real exam problem, which means there is always one unique correct answer. Here is the math question. {problem} Here is the candidate solution. {solution}. Here is the reasoning steps. {reasoning} \n"
                "Now we will proceed to analyzing the correctness of the candidate solution. We have two goals: identify if there is an error, and if there is an error, repair the candidate solution and identify whether the final answer has changed. For now, we will focus on methodically combing through the candidate solution and checking each step for a potential error. "
                "[Steps to Follow] You will proceed through the candidate solution, one baby step at a time. In your exhaustive investigation of the solution, you will first form a short list of potential errors. Each entry in this list should be written as a self-contained, standalone mathematical claim that the candidate solution relies on being correct, but which you find suspicious upon first inspection. After forming this list, you will then proceed to validate each potential error by performing a detailed error validation check. "
                "At the end, if you found a fatal error, then the solution is incorrect and you should return **False**. Otherwise, say **True** if the final answer is correct.",
            ),
        ],
    )

    bound_llm = llm.with_structured_output(Verification)
    return prompt | bound_llm
