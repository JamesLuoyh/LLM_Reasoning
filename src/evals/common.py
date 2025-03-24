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


def is_equiv(ground_truth: str, given: str,
             verbose: bool = False, fast: bool = False):
    if ground_truth is None and given is None:
        print("WARNING: Both None")
        return True
    if ground_truth is None or given is None:
        return False

    str_pass = ground_truth.strip() == given.strip()

    if str_pass:
        return True

    def default_is_equiv(
            ground_truth_normalized: Union[str, set], given_normalized: Union[str, set]) -> bool:
        if verbose:
            print(ground_truth_normalized, given_normalized)

        if isinstance(ground_truth_normalized,
                      str) and ground_truth_normalized == given_normalized:
            return True

        try:
            # e.g., gt = 30^\\circ -> {30, pi/6}, gv = pi/6
            if isinstance(ground_truth_normalized, set) or isinstance(
                    given_normalized, set):
                if isinstance(ground_truth_normalized, str):
                    ground_truth_normalized = {ground_truth_normalized}
                if isinstance(given_normalized, str):
                    given_normalized = {given_normalized}
                for gt_norm in ground_truth_normalized:
                    for gv_norm in given_normalized:
                        if is_latex_equiv(gt_norm, gv_norm, verbose=verbose):
                            return True
                return False
            else:
                return is_latex_equiv(
                    ground_truth_normalized, given_normalized, verbose=verbose)

        except Exception as e:
            return False

    ground_truth_normalized = string_normalize(ground_truth)
    given_normalized = string_normalize(given)
    default_equiv = default_is_equiv(ground_truth_normalized, given_normalized)
    if fast or default_equiv:
        return default_equiv
    else:
        ground_truth_normalized = string_normalize(
            ground_truth, remove_mid_std_space=False)
        given_normalized = string_normalize(given, remove_mid_std_space=False)
        default_equiv_space = default_is_equiv(
            ground_truth_normalized, given_normalized)
        if default_equiv_space:
            return True

        ground_truth_normalized = string_normalize(
            ground_truth, lower_case=False)
        given_normalized = string_normalize(given, lower_case=False)
        default_equiv_case = default_is_equiv(
            ground_truth_normalized, given_normalized)
        if default_equiv_case:
            return True

        raw_equiv = are_equal_under_sympy(ground_truth, given)
        return raw_equiv


def is_latex_equiv(
    ground_truth_normalized: str,
    given_normalized: str,
    verbose: bool = False,
) -> bool:
    if len(given_normalized) == 0:
        return False

    is_correct, splitted = False, False
    if (
        "(" in ground_truth_normalized
        or ")" in ground_truth_normalized
        or "[" in ground_truth_normalized
        or "]" in ground_truth_normalized
    ):
        is_correct, splitted = is_equiv_possible_intervals(
            ground_truth_normalized, given_normalized, verbose)

    if not is_correct:
        is_correct, splitted = is_equiv_possible_tuple(
            ground_truth_normalized, given_normalized, verbose)

    if not is_correct and (_str_is_mat(
            ground_truth_normalized) or _str_is_mat(given_normalized)):
        is_correct, splitted = is_equiv_possible_matrix(
            ground_truth_normalized, given_normalized, verbose)

    if not is_correct and not splitted:
        is_correct = are_equal_under_sympy(
            ground_truth_normalized, given_normalized, verbose)
    return is_correct


def is_equiv_possible_intervals(
    ground_truth_normalized: str,
    given_normalized: str,
    verbose: bool = False,
) -> Tuple[bool, bool]:
    gt_interval = _str_to_interval(ground_truth_normalized)
    gv_interval = _str_to_interval(given_normalized)

    splitted = True
    if gt_interval is None and gv_interval is None:
        splitted = False

    if gt_interval is not None and gv_interval is not None and gt_interval.compare(
            gv_interval) == 0:
        return True, splitted

    return False, splitted


def is_equiv_possible_tuple(
    ground_truth_normalized: str,
    given_normalized: str,
    verbose: bool = False,
) -> Tuple[bool, bool]:
    # split "(,,,)" or "[,,,]" into list, split ",,," into set
    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)

    if verbose:
        print(ground_truth_elems, given_elems)

    splitted = True
    if isinstance(ground_truth_elems, str) and isinstance(given_elems, str):
        if ground_truth_elems == ground_truth_normalized and given_elems == given_normalized:
            return False, False
        else:
            return is_equiv(ground_truth_elems, given_elems, verbose), splitted

    is_correct = False
    if len(ground_truth_elems) != len(
            given_elems) and not "\\in" in given_elems:
        is_correct = False
    elif not isinstance(ground_truth_elems, type(given_elems)):
        is_correct = False
    elif isinstance(ground_truth_elems, (list, tuple)):
        for ground_truth_elem, given_elem in zip(
                ground_truth_elems, given_elems):
            if not is_equiv(ground_truth_elem, given_elem, verbose):
                return False, splitted
        return True, splitted
    elif isinstance(ground_truth_elems, set):
        gt_found_matches = [False] * len(ground_truth_elems)
        gv_found_matches = [False] * len(given_elems)
        for i, ground_truth_elem in enumerate(ground_truth_elems):
            if not gt_found_matches[i]:
                for j, given_elem in enumerate(given_elems):
                    if not gv_found_matches[j] and is_equiv(
                            ground_truth_elem, given_elem, verbose):
                        gt_found_matches[i] = True
                        gv_found_matches[j] = True
                        break
        return all(gt_found_matches), splitted

    return is_correct, splitted


def is_equiv_possible_matrix(
    ground_truth_normalized: str,
    given_normalized: str,
    verbose: bool = False,
) -> Tuple[bool, bool]:
    gt_matrix = split_matrix(ground_truth_normalized)
    gv_matrix = split_matrix(given_normalized)

    splitted = True
    if isinstance(gt_matrix, str) and isinstance(gv_matrix, str):
        if gt_matrix == ground_truth_normalized and gv_matrix == given_normalized:
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
