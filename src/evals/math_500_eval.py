import random
import re

import pandas

from evals import common
from evals.common import ANSWER_PATTERN, HTML_JINJA
from evals.objects import Eval, EvalResult, LanguageModel, SingleEvalResult
from evals.templates import MATH_QUERY_TEMPLATE


class MathEval(Eval):
    def __init__(
        self,
        equality_checker: LanguageModel,
        num_examples: int | None = None,
        n_repeats: int = 16,
    ):
        df = pandas.read_csv(
            f"https://openaipublic.blob.core.windows.net/simple-evals/math_500_test.csv")
        examples = [row.to_dict() for _, row in df.iterrows()]
        if num_examples:
            assert n_repeats == 1, "n_repeats only supported for num_examples = None"
            rng = random.Random(0)
            examples = rng.sample(examples, num_examples)
        self.examples = examples * n_repeats
        self.equality_checker = equality_checker

    def __call__(self, sampler: LanguageModel) -> EvalResult:
        def fn(row: dict):
            prompt_messages = [
                sampler._pack_message(
                    content=MATH_QUERY_TEMPLATE.format(
                        **row), role="user")]
            response_text = sampler(prompt_messages).content
            match = re.search(ANSWER_PATTERN, response_text)
            extracted_answer = match.group(1) if match else None
            target_match = re.search(r"\\boxed\{(.*)\}", row["Answer"])
            target_answer = target_match.group(1) if target_match else None
            score = float(
                common.check_equality(
                    self.equality_checker,
                    target_answer,
                    extracted_answer))
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
