# Code adapted from OpenAI (2024)
# Source: https://github.com/openai/simple-evals/
# Licensed under the MIT License

import os
from collections import defaultdict
from multiprocessing.pool import ThreadPool
from typing import Any, Tuple, Union

import jinja2
import numpy as np
from tqdm import tqdm

from evals.objects import EvalResult, LanguageModel, Message, SingleEvalResult
from evals.templates import EQUALITY_TEMPLATE, QUERY_TEMPLATE_MULTICHOICE

from .latex_normalize import _str_is_mat, _str_to_interval, split_matrix, split_tuple, string_normalize
from .latex_parser import are_equal_under_sympy

ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?"
ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"
MULTILINGUAL_ANSWER_PATTERN_TEMPLATE = "(?i){}[ \t]*([A-D]|[أ-د]|[অ]|[ব]|[ড]|[ঢ]|[Ａ]|[Ｂ]|[Ｃ]|[Ｄ])"

HTML_JINJA = """
<h3>Prompt conversation</h3>
{% for message in prompt_messages %}
{{ message_to_html(message) | safe }}
{% endfor %}
<h3>Sampled message</h3>
{{ message_to_html(next_message) | safe }}
<h3>Results</h3>
<p>Correct Answer: {{ correct_answer }}</p>
<p>Extracted Answer: {{ extracted_answer }}</p>
<p>Score: {{ score }}</p>
"""


def format_multichoice_question(row):
    return QUERY_TEMPLATE_MULTICHOICE.format(**row)


def check_equality(sampler: LanguageModel, expr1: str, expr2: str):
    prompt = EQUALITY_TEMPLATE % {"expression1": expr1, "expression2": expr2}
    response = sampler([dict(content=prompt, role="user")]).content
    return response.lower().strip() == "yes"


def is_equiv(expression_1: str, expression_2: str,
             verbose: bool = False, fast: bool = False):
    if expression_1 is None and expression_2 is None:
        print("WARNING: Both None")
        return True
    if expression_1 is None or expression_2 is None:
        return False

    str_pass = expression_1.strip() == expression_2.strip()

    if str_pass:
        return True

    def default_is_equiv(
            expression_1_normalized: Union[str, set], expression_2_normalized: Union[str, set]) -> bool:
        if verbose:
            print(expression_1_normalized, expression_2_normalized)

        if isinstance(expression_1_normalized,
                      str) and expression_1_normalized == expression_2_normalized:
            return True

        try:
            # e.g., gt = 30^\\circ -> {30, pi/6}, gv = pi/6
            if isinstance(expression_1_normalized, set) or isinstance(
                    expression_2_normalized, set):
                if isinstance(expression_1_normalized, str):
                    expression_1_normalized = {expression_1_normalized}
                if isinstance(expression_2_normalized, str):
                    expression_2_normalized = {expression_2_normalized}
                for gt_norm in expression_1_normalized:
                    for gv_norm in expression_2_normalized:
                        if is_latex_equiv(gt_norm, gv_norm, verbose=verbose):
                            return True
                return False
            else:
                return is_latex_equiv(
                    expression_1_normalized, expression_2_normalized, verbose=verbose)

        except Exception as e:
            return False

    expression_1_normalized = string_normalize(expression_1)
    expression_2_normalized = string_normalize(expression_2)
    default_equiv = default_is_equiv(
        expression_1_normalized,
        expression_2_normalized)
    if fast or default_equiv:
        return default_equiv
    else:
        expression_1_normalized = string_normalize(
            expression_1, remove_mid_std_space=False)
        expression_2_normalized = string_normalize(
            expression_2, remove_mid_std_space=False)
        default_equiv_space = default_is_equiv(
            expression_1_normalized, expression_2_normalized)
        if default_equiv_space:
            return True

        expression_1_normalized = string_normalize(
            expression_1, lower_case=False)
        expression_2_normalized = string_normalize(
            expression_2, lower_case=False)
        default_equiv_case = default_is_equiv(
            expression_1_normalized, expression_2_normalized)
        if default_equiv_case:
            return True

        raw_equiv = are_equal_under_sympy(expression_1, expression_2)
        return raw_equiv


def is_latex_equiv(
    expression_1_normalized: str,
    expression_2_normalized: str,
    verbose: bool = False,
) -> bool:
    if len(expression_2_normalized) == 0:
        return False

    is_correct, splitted = False, False
    if (
        "(" in expression_1_normalized
        or ")" in expression_1_normalized
        or "[" in expression_1_normalized
        or "]" in expression_1_normalized
    ):
        is_correct, splitted = is_equiv_possible_intervals(
            expression_1_normalized, expression_2_normalized, verbose)

    if not is_correct:
        is_correct, splitted = is_equiv_possible_tuple(
            expression_1_normalized, expression_2_normalized, verbose)

    if not is_correct and (_str_is_mat(expression_1_normalized)
                           or _str_is_mat(expression_2_normalized)):
        is_correct, splitted = is_equiv_possible_matrix(
            expression_1_normalized, expression_2_normalized, verbose)

    if not is_correct and not splitted:
        is_correct = are_equal_under_sympy(
            expression_1_normalized, expression_2_normalized, verbose)
    return is_correct


def is_equiv_possible_intervals(
    expression_1_normalized: str,
    expression_2_normalized: str,
    verbose: bool = False,
) -> Tuple[bool, bool]:
    gt_interval = _str_to_interval(expression_1_normalized)
    gv_interval = _str_to_interval(expression_2_normalized)

    splitted = True
    if gt_interval is None and gv_interval is None:
        splitted = False

    if gt_interval is not None and gv_interval is not None and gt_interval.compare(
            gv_interval) == 0:
        return True, splitted

    return False, splitted


def is_equiv_possible_tuple(
    expression_1_normalized: str,
    expression_2_normalized: str,
    verbose: bool = False,
) -> Tuple[bool, bool]:
    # split "(,,,)" or "[,,,]" into list, split ",,," into set
    expression_1_elems = split_tuple(expression_1_normalized)
    expression_2_elems = split_tuple(expression_2_normalized)

    if verbose:
        print(expression_1_elems, expression_2_elems)

    splitted = True
    if isinstance(expression_1_elems, str) and isinstance(
            expression_2_elems, str):
        if expression_1_elems == expression_1_normalized and expression_2_elems == expression_2_normalized:
            return False, False
        else:
            return is_equiv(expression_1_elems,
                            expression_2_elems, verbose), splitted

    is_correct = False
    if len(expression_1_elems) != len(
            expression_2_elems) and not "\\in" in expression_2_elems:
        is_correct = False
    elif not isinstance(expression_1_elems, type(expression_2_elems)):
        is_correct = False
    elif isinstance(expression_1_elems, (list, tuple)):
        for expression_1_elem, expression_2_elem in zip(
                expression_1_elems, expression_2_elems):
            if not is_equiv(expression_1_elem, expression_2_elem, verbose):
                return False, splitted
        return True, splitted
    elif isinstance(expression_1_elems, set):
        gt_found_matches = [False] * len(expression_1_elems)
        gv_found_matches = [False] * len(expression_2_elems)
        for i, expression_1_elem in enumerate(expression_1_elems):
            if not gt_found_matches[i]:
                for j, expression_2_elem in enumerate(expression_2_elems):
                    if not gv_found_matches[j] and is_equiv(
                            expression_1_elem, expression_2_elem, verbose):
                        gt_found_matches[i] = True
                        gv_found_matches[j] = True
                        break
        return all(gt_found_matches), splitted

    return is_correct, splitted


def is_equiv_possible_matrix(
    expression_1_normalized: str,
    expression_2_normalized: str,
    verbose: bool = False,
) -> Tuple[bool, bool]:
    gt_matrix = split_matrix(expression_1_normalized)
    gv_matrix = split_matrix(expression_2_normalized)

    splitted = True
    if isinstance(gt_matrix, str) and isinstance(gv_matrix, str):
        if gt_matrix == expression_1_normalized and gv_matrix == expression_2_normalized:
            return False, False
        else:
            return is_equiv(gt_matrix, gv_matrix), splitted

    elif isinstance(gt_matrix, list) and isinstance(gv_matrix, list):
        # check num of rows are equal
        if len(gt_matrix) != len(gv_matrix):
            return False, splitted

        for gt_col, gv_col in zip(gt_matrix, gv_matrix):
            if isinstance(gt_col, str) and isinstance(
                    gv_col, str) and is_equiv(gt_col, gv_col):
                continue

            elif isinstance(gt_col, list) and isinstance(gv_col, list):
                # check num of cols are equal
                if len(gt_col) != len(gv_col):
                    return False, splitted

                for gt_col_item, gv_col_item in zip(gt_col, gv_col):
                    if not is_equiv(gt_col_item, gv_col_item):
                        return False, splitted
            else:
                return False, splitted

        return True, splitted

    else:
        return False, splitted


def _compute_stat(values: list, stat: str):
    if stat == "mean":
        return np.mean(values)
    elif stat == "std":
        return np.std(values)
    elif stat == "min":
        return np.min(values)
    elif stat == "max":
        return np.max(values)
    else:
        raise ValueError(f"Unknown {stat=}")


def aggregate_results(
    single_eval_results: list[SingleEvalResult],
    default_stats: tuple[str] = ("mean", "std"),
    name2stats: dict[str, tuple[str]] | None = None,
) -> EvalResult:
    """
    Aggregate results from multiple evaluations into a single EvalResult.
    """
    name2stats = name2stats or {}
    name2values = defaultdict(list)
    htmls = []
    convos = []
    for single_eval_result in single_eval_results:
        for name, value in single_eval_result.metrics.items():
            name2values[name].append(value)
        if single_eval_result.score is not None:
            name2values["score"].append(single_eval_result.score)
        htmls.append(single_eval_result.html)
        convos.append(single_eval_result.convo)
    final_metrics = {}
    for name, values in name2values.items():
        stats = name2stats.get(name, default_stats)
        for stat in stats:
            key = name if stat == "mean" else f"{name}:{stat}"
            final_metrics[key] = _compute_stat(values, stat)
    return EvalResult(score=final_metrics.pop("score", None),
                      metrics=final_metrics, htmls=htmls, convos=convos)


def map_with_progress(f: callable, xs: list[Any], num_threads: int = 2):
    """
    Apply f to each element of xs, using a ThreadPool, and show progress.
    """
    if os.getenv("debug"):
        return list(map(f, tqdm(xs, total=len(xs))))
    else:
        with ThreadPool(min(num_threads, len(xs))) as pool:
            return list(tqdm(pool.imap(f, xs), total=len(xs)))


jinja_env = jinja2.Environment(
    loader=jinja2.BaseLoader(),
    undefined=jinja2.StrictUndefined,
    autoescape=jinja2.select_autoescape(["html", "xml"]),
)
_message_template = """
<div class="message {{ role }}">
    <div class="role">
    {{ role }}
    {% if variant %}<span class="variant">({{ variant }})</span>{% endif %}
    </div>
    <div class="content">
    <pre>{{ content }}</pre>
    </div>
</div>
"""


def message_to_html(message: Message) -> str:
    """
    Generate HTML snippet (inside a <div>) for a message.
    """
    return jinja_env.from_string(_message_template).render(
        role=message["role"], content=message["content"], variant=message.get(
            "variant", None)
    )


jinja_env.globals["message_to_html"] = message_to_html


_report_template = """<!DOCTYPE html>
<html>
    <head>
        <style>
            .message {
                padding: 8px 16px;
                margin-bottom: 8px;
                border-radius: 4px;
            }
            .message.user {
                background-color: #B2DFDB;
                color: #00695C;
            }
            .message.assistant {
                background-color: #B39DDB;
                color: #4527A0;
            }
            .message.system {
                background-color: #EEEEEE;
                color: #212121;
            }
            .role {
                font-weight: bold;
                margin-bottom: 4px;
            }
            .variant {
                color: #795548;
            }
            table, th, td {
                border: 1px solid black;
            }
            pre {
                white-space: pre-wrap;
            }
        </style>
    </head>
    <body>
    {% if metrics %}
    <h1>Metrics</h1>
    <table>
    <tr>
        <th>Metric</th>
        <th>Value</th>
    </tr>
    <tr>
        <td><b>Score</b></td>
        <td>{{ score | float | round(3) }}</td>
    </tr>
    {% for name, value in metrics.items() %}
    <tr>
        <td>{{ name }}</td>
        <td>{{ value }}</td>
    </tr>
    {% endfor %}
    </table>
    {% endif %}
    <h1>Examples</h1>
    {% for html in htmls %}
    {{ html | safe }}
    <hr>
    {% endfor %}
    </body>
</html>
"""


def make_report(eval_result: EvalResult) -> str:
    """
    Create a standalone HTML report from an EvalResult.
    """
    return jinja_env.from_string(_report_template).render(
        score=eval_result.score,
        metrics=eval_result.metrics,
        htmls=eval_result.htmls,
    )


def make_report_from_example_htmls(htmls: list[str]):
    """
    Create a standalone HTML report from a list of example htmls
    """
    return jinja_env.from_string(_report_template).render(
        score=None, metrics={}, htmls=htmls)


def normalize_response(response: str) -> str:
    """
    Normalize the response by removing markdown and LaTeX formatting that may prevent a match.
    """

    return (
        response.replace("**", "")
        .replace("$\\boxed{", "")
        .replace("}$", "")
        .replace("\\$", "")
        .replace("$\\text{", "")
        .replace("$", "")
        .replace("\\mathrm{", "")
        .replace("\\{", "")
        .replace("\\text", "")
        .replace("\\(", "")
        .replace("\\mathbf{", "")
        .replace("{", "")
        .replace("\\boxed", "")
    )


def normalize_extracted_answer(extracted_answer: str) -> str:
    return (
        # In arabic these are the letters used for A-D in multiple choice
        # questions
        extracted_answer.replace("أ", " A")
        .replace("ب", " B")
        .replace("ج", " C")
        .replace("د", " D")
        # In Bengali these are the letters used for A-D in multiple choice
        # questions
        .replace("অ", " A")
        .replace("ব", " B")
        .replace("ড", " C")
        .replace("ঢ", " D")
        # In Japanese these are the letters sometimes used for A-D in multiple
        # choice questions
        .replace("Ａ", " A")
        .replace("Ｂ", " B")
        .replace("Ｃ", " C")
        .replace("Ｄ", " D")
        .strip()
    )
