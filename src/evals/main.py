# Code adapted from OpenAI (2024)
# Source: https://github.com/openai/simple-evals/
# Licensed under the MIT License

import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd

from evals import common
from evals.math_500_eval import MathEval
from evals.models import *


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluations using different models/model structures")
    parser.add_argument(
        "--list-model-structures",
        action="store_true",
        help="List available model structures")
    parser.add_argument(
        "--model-structure",
        type=str,
        help="Select a model structure by name")
    parser.add_argument(
        "--model-type",
        type=str,
        default="gemini2_flash",
        help="Select a model type by name",
    )
    parser.add_argument(
        "--allow-duplicates",
        action="store_false",
        help="Allow duplicates in the generated solutions")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode")
    parser.add_argument(
        "--examples",
        type=int,
        help="Number of examples to use (overrides default)")
    parser.add_argument(
        "--idx_examples",
        type=int,
        nargs="+",
        help="A list of examples from the dataset to test")

    parser.add_argument(
        "--n_repeats",
        type=int,
        help="Number of times to repeat per example")

    parser.add_argument(
        "--verifier_temperature", type=float, default=0.7, help="The temperature of the verifiers or voters"
    )

    parser.add_argument(
        "--n_verifiers",
        type=int,
        default=5,
        help="The number of verifiers")

    parser.add_argument("--seed", type=int, default=5, help="The random seed")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    model_types = ["llama3", "gemini2_flash"]

    if args.model_type:
        if args.model_type not in model_types:
            print(f"Model type {args.model_type} not found")
            return
    model_structures = {
        # Baseline models
        "base_llama3": Llama3(temperature=0.7, num_predict=2048, structured=False),
        "ToT": ToT(temperature=0.7),
        "majority_vote": MajorityVote(
            model_type=args.model_type,
            temperature=0.7,
            num_predict=2048,
            allow_duplicates=args.allow_duplicates,
            debug=args.debug,
        ),
        "borda_count": BordaCount(
            model_type=args.model_type,
            temperature=0.7,
            num_predict=2048,
            allow_duplicates=args.allow_duplicates,
            debug=args.debug,
            verifier_temperature=args.verifier_temperature,
            n_verifiers=args.n_verifiers,
        ),
        "best_of_n": BestOfN(
            model_type=args.model_type,
            temperature=0.7,
            num_predict=2048,
            allow_duplicates=args.allow_duplicates,
            debug=args.debug,
        ),
        "self_consistency": SelfConsistency(
            model_type=args.model_type,
            temperature=0.7,
            num_predict=2048,
            allow_duplicates=args.allow_duplicates,
            debug=args.debug,
        ),
        "gemini2_flash": Gemini2_flash(temperature=1.5),
        "verification": ScaleVerification(
            model_type=args.model_type,
            temperature=0.7,
            num_predict=2048,
            allow_duplicates=args.allow_duplicates,
            debug=args.debug,
            verifier_temperature=args.verifier_temperature,
            n_verifiers=args.n_verifiers,
        ),
    }

    if args.list_model_structures:
        print("Available model structures:")
        for model_structure_name in model_structures.keys():
            print(f" - {model_structure_name}")
        return

    if args.model_structure:
        if args.model_structure not in model_structures:
            print(f"Model structure {args.model_structure} not found")
            return
        model_structure = {
            args.model_structure: model_structures[args.model_structure]}

    equality_checker = Gemini2_flash(
        temperature=1.5, structured=False)  # Llama3()

    def get_evals(eval_name, debug_mode):
        num_examples = args.examples if args.examples is not None else (
            5 if debug_mode else None)
        # Set num_examples = None to reproduce full evals
        n_repeats = 10
        if args.n_repeats:
            n_repeats = args.n_repeats
        elif debug_mode or num_examples:
            n_repeats = 1

        match eval_name:
            case "math":
                return MathEval(
                    equality_checker=equality_checker,
                    num_examples=num_examples,
                    idx_examples=args.idx_examples,
                    n_repeats=n_repeats,
                    answer_format=True if args.model_structure.startswith(
                        "base") else False,
                )
            case _:
                raise Exception(f"Unrecognized eval type: {eval_name}")

    evals = {eval_name: get_evals(eval_name, args.debug)
             for eval_name in ["math"]}
    print(evals)
    debug_suffix = "_DEBUG" if args.debug else ""
    print(debug_suffix)
    mergekey2resultpath = {}

    root_report_foldername = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "reports")
    if not os.path.isdir(root_report_foldername):
        os.makedirs(root_report_foldername)

    for model_name, sampler in model_structure.items():
        for eval_name, eval_obj in evals.items():
            result = eval_obj(sampler)
            # ^^^ how to use a sampler
            file_stem = (
                f"{eval_name}_{model_name}_ng_5_gtemp_0.7_nv_{
                    args.n_verifiers}_vtemp_{
                    args.verifier_temperature}"
            )
            dateTimeObj = datetime.now()
            date = dateTimeObj.strftime("_%m_%d_%H_%M_%S")
            report_filename = f"/{root_report_foldername}/{file_stem}{debug_suffix}{date}.html"
            print(f"Writing report to {report_filename}")
            with open(report_filename, "w") as fh:
                fh.write(common.make_report(result))
            metrics = result.metrics | {"score": result.score}
            print(metrics)
            result_filename = f"/{root_report_foldername}/{file_stem}{debug_suffix}.json"
            with open(result_filename, "w") as f:
                f.write(json.dumps(metrics, indent=2))
            print(f"Writing results to {result_filename}")
            mergekey2resultpath[f"{file_stem}"] = result_filename
    merge_metrics = []
    for eval_model_name, result_filename in mergekey2resultpath.items():
        try:
            result = json.load(open(result_filename, "r+"))
        except Exception as e:
            print(e, result_filename)
            continue
        result = result.get("f1_score", result.get("score", None))
        eval_name = eval_model_name[: eval_model_name.find("_")]
        model_name = eval_model_name[eval_model_name.find("_") + 1:]
        merge_metrics.append(
            {"eval_name": eval_name, "model_name": model_name, "metric": result})
    merge_metrics_df = pd.DataFrame(merge_metrics).pivot(
        index=["model_name"], columns="eval_name")
    print("\nAll results: ")
    print(merge_metrics_df.to_markdown())
    return merge_metrics


if __name__ == "__main__":
    main()
