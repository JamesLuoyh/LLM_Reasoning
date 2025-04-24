# Code adapted from OpenAI (2024)
# Source: https://github.com/openai/simple-evals/
# Licensed under the MIT License

import random
import re

import pandas

from evals import common
from evals.common import ANSWER_PATTERN, HTML_JINJA
from evals.constants import INVALID_ANS
from evals.objects import Eval, EvalResult, LanguageModel, SingleEvalResult
from evals.templates import MATH_QUERY_TEMPLATE, MATH_QUERY_TEMPLATE_WITHOUT_ANSWER_LINE


class MathEval(Eval):
    def __init__(
        self,
        equality_checker: LanguageModel,
        num_examples: int | None = None,
        idx_examples: list | None = None,
        n_repeats: int = 16,
        answer_format: bool = True,
    ):
        df = pandas.read_csv("data/math_500_test.csv")
        examples = [row.to_dict() for _, row in df.iterrows()]
        if idx_examples is not None and len(idx_examples) > 0:
            self.examples = [examples[i] for i in idx_examples]
            print(self.examples)
        else:
            if num_examples:
                assert n_repeats == 1, "n_repeats only supported for num_examples = None"
                rng = random.Random(0)
                examples = rng.sample(examples, num_examples)
            self.examples = examples * n_repeats
        print(self.examples)
        self.equality_checker = equality_checker
        self.answer_format = answer_format

    def __call__(self, sampler: LanguageModel) -> EvalResult:
        def fn(row: dict):
            if self.answer_format:  # Answer format is for eval that does not use langchain
                prompt_messages = [
                    sampler._pack_message(
                        content=MATH_QUERY_TEMPLATE.format(
                            **row), role="user")]
            else:
                prompt_messages = [
                    sampler._pack_message(
                        content=MATH_QUERY_TEMPLATE_WITHOUT_ANSWER_LINE.format(
                            **row), role="user")
                ]
            # try:
            response_text = sampler(prompt_messages).content
            # except Exception as e:
            #     print(f"Error in model invocation: {e}")
            #     response_text = ""

            if self.answer_format:
                match = re.search(ANSWER_PATTERN, response_text)
                extracted_answer = match.group(1) if match else None
            else:
                extracted_answer = response_text

            target_match = re.search(r"\\boxed\{(.*)\}", row["Answer"])
            target_answer = target_match.group(1) if target_match else None
            score = common.is_equiv(target_answer, extracted_answer)
            if score < 1 and extracted_answer is not None and extracted_answer != INVALID_ANS:
                # use gemini2 flash to double check
                print("Here")
                score = common.check_equality(
                    self.equality_checker, target_answer, extracted_answer)
                print(score)
            score = float(score)
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=target_answer,
                extracted_answer=extracted_answer,
            )
            convo = prompt_messages + \
                [dict(content=response_text, role="assistant")]
            return SingleEvalResult(html=html, score=score, convo=convo)

        results = common.map_with_progress(fn, self.examples)
        return common.aggregate_results(results)
