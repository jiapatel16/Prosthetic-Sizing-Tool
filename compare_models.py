from pathlib import Path
import json
import re

import numpy as np


MODEL_ANALYSIS_PATH = Path(
    "models/model_analysis.json"
)

COMBINED_REPORT_PATH = Path(
    "reports/combined_photo_report.txt"
)

COMPARISON_PATH = Path(
    "models/model_comparison.json"
)

REPORT_PATH = Path(
    "reports/model_comparison_report.txt"
)


def load_model_analyses() -> list[dict]:
    if not MODEL_ANALYSIS_PATH.exists():
        raise FileNotFoundError(
            "Could not find models/model_analysis.json. "
            "Run analyze_model_library.py first."
        )

    with MODEL_ANALYSIS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        analyses = json.load(file)

    if not isinstance(analyses, list):
        raise ValueError(
            "model_analysis.json must contain a list."
        )

    return analyses


def load_combined_report() -> str:
    if not COMBINED_REPORT_PATH.exists():
        raise FileNotFoundError(
            "Could not find "
            "reports/combined_photo_report.txt. "
            "Run combine_views.py first."
        )

    return COMBINED_REPORT_PATH.read_text(
        encoding="utf-8"
    )


def extract_mm_value(
    report: str,
    label: str,
) -> float:
    pattern = (
        rf"{re.escape(label)}:\s*"
        rf"([\d.]+)\s*mm"
    )

    match = re.search(
        pattern,
        report,
        re.IGNORECASE,
    )

    if match is None:
        raise ValueError(
            f"Could not find '{label}' "
            "in the combined report."
        )

    return float(
        match.group(1)
    )


def load_recipient_measurements() -> dict:
    report = load_combined_report()

    return {
        "length_mm": extract_mm_value(
            report,
            "Average estimate",
        ),
        "width_mm": extract_mm_value(
            report,
            "Maximum top-view width",
        ),
        "thickness_mm": extract_mm_value(
            report,
            "Maximum side-view thickness",
        ),
    }


def calculate_uniform_scale(
    recipient: np.ndarray,
    model: np.ndarray,
) -> float:
    denominator = float(
        np.dot(model, model)
    )

    if denominator <= 0:
        raise ValueError(
            "Model dimensions must be positive."
        )

    scale = float(
        np.dot(recipient, model)
        / denominator
    )

    return scale


def calculate_errors(
    recipient: np.ndarray,
    scaled_model: np.ndarray,
) -> dict:
    absolute_errors = np.abs(
        scaled_model - recipient
    )

    percentage_errors = (
        absolute_errors
        / recipient
        * 100.0
    )

    mean_percentage_error = float(
        np.mean(percentage_errors)
    )

    maximum_percentage_error = float(
        np.max(percentage_errors)
    )

    score = max(
        0.0,
        100.0 - mean_percentage_error,
    )

    return {
        "length_error_mm":
            float(absolute_errors[0]),

        "width_error_mm":
            float(absolute_errors[1]),

        "thickness_error_mm":
            float(absolute_errors[2]),

        "length_error_percent":
            float(percentage_errors[0]),

        "width_error_percent":
            float(percentage_errors[1]),

        "thickness_error_percent":
            float(percentage_errors[2]),

        "mean_error_percent":
            mean_percentage_error,

        "maximum_error_percent":
            maximum_percentage_error,

        "geometric_similarity_score":
            score,
    }


def compare_model(
    model: dict,
    recipient_measurements: dict,
) -> dict:
    dimensions = model.get(
        "dimensions_mm",
        {}
    )

    model_length = float(
        dimensions.get("length", 0)
    )

    model_width = float(
        dimensions.get("width", 0)
    )

    model_thickness = float(
        dimensions.get("thickness", 0)
    )

    if (
        model_length <= 0
        or model_width <= 0
        or model_thickness <= 0
    ):
        raise ValueError(
            "The model has invalid dimensions."
        )

    recipient_vector = np.array(
        [
            recipient_measurements["length_mm"],
            recipient_measurements["width_mm"],
            recipient_measurements["thickness_mm"],
        ],
        dtype=float,
    )

    model_vector = np.array(
        [
            model_length,
            model_width,
            model_thickness,
        ],
        dtype=float,
    )

    scale = calculate_uniform_scale(
        recipient_vector,
        model_vector,
    )

    scaled_model = (
        model_vector * scale
    )

    errors = calculate_errors(
        recipient_vector,
        scaled_model,
    )

    return {
        "name":
            model.get("name", "Unknown"),

        "device_folder":
            model.get(
                "device_folder",
                "Unknown",
            ),

        "part_type":
            model.get(
                "part_type",
                "Unknown",
            ),

        "stl_path":
            model.get("stl_path", ""),

        "original_dimensions_mm": {
            "length": model_length,
            "width": model_width,
            "thickness": model_thickness,
        },

        "experimental_uniform_scale":
            scale,

        "experimental_scale_percent":
            scale * 100.0,

        "scaled_dimensions_mm": {
            "length":
                float(scaled_model[0]),

            "width":
                float(scaled_model[1]),

            "thickness":
                float(scaled_model[2]),
        },

        **errors,

        "warning": (
            "This is an external bounding-box "
            "comparison only. It does not measure "
            "the internal socket cavity or confirm fit."
        ),
    }


def compare_all_models(
    analyses: list[dict],
    recipient_measurements: dict,
) -> tuple[list[dict], list[dict]]:
    comparisons = []
    failures = []

    for model in analyses:
        try:
            comparison = compare_model(
                model,
                recipient_measurements,
            )

            comparisons.append(
                comparison
            )

        except (
            ValueError,
            TypeError,
            KeyError,
        ) as error:
            failures.append(
                {
                    "name":
                        model.get(
                            "name",
                            "Unknown",
                        ),
                    "error":
                        str(error),
                }
            )

    comparisons.sort(
        key=lambda result:
            result[
                "geometric_similarity_score"
            ],
        reverse=True,
    )

    return comparisons, failures


def save_json_results(
    recipient: dict,
    comparisons: list[dict],
    failures: list[dict],
) -> None:
    COMPARISON_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "recipient_measurements_mm":
            recipient,

        "comparison_type":
            "Experimental external bounding-box screening",

        "comparisons":
            comparisons,

        "failures":
            failures,
    }

    with COMPARISON_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=4,
        )


def build_text_report(
    recipient: dict,
    comparisons: list[dict],
    failures: list[dict],
) -> str:
    lines = [
        "========================================",
        " PROZI EXPERIMENTAL MODEL COMPARISON",
        "========================================",
        "",
        "Recipient Measurements",
        "----------------------",
        f"Length: {recipient['length_mm']:.1f} mm",
        f"Top-view width: {recipient['width_mm']:.1f} mm",
        (
            "Side-view thickness: "
            f"{recipient['thickness_mm']:.1f} mm"
        ),
        "",
        "Ranked Geometric Screening Results",
        "----------------------------------",
    ]

    if not comparisons:
        lines.append(
            "No models were successfully compared."
        )

    for rank, result in enumerate(
        comparisons,
        start=1,
    ):
        scaled = result[
            "scaled_dimensions_mm"
        ]

        lines.extend(
            [
                "",
                f"{rank}. {result['name']}",
                f"Part: {result['part_type']}",
                (
                    "Geometric similarity score: "
                    f"{result['geometric_similarity_score']:.1f}%"
                ),
                (
                    "Experimental uniform scale: "
                    f"{result['experimental_scale_percent']:.1f}%"
                ),
                (
                    "Scaled external dimensions: "
                    f"{scaled['length']:.1f} × "
                    f"{scaled['width']:.1f} × "
                    f"{scaled['thickness']:.1f} mm"
                ),
                (
                    "Mean dimensional error: "
                    f"{result['mean_error_percent']:.1f}%"
                ),
                (
                    "Length error: "
                    f"{result['length_error_mm']:.1f} mm"
                ),
                (
                    "Width error: "
                    f"{result['width_error_mm']:.1f} mm"
                ),
                (
                    "Thickness error: "
                    f"{result['thickness_error_mm']:.1f} mm"
                ),
            ]
        )

    if failures:
        lines.extend(
            [
                "",
                "Models Not Compared",
                "-------------------",
            ]
        )

        for failure in failures:
            lines.append(
                f"- {failure['name']}: "
                f"{failure['error']}"
            )

    lines.extend(
        [
            "",
            "Critical Limitation",
            "-------------------",
            (
                "This ranking compares the recipient "
                "measurements with each STL's external "
                "bounding box."
            ),
            (
                "It does not measure the internal socket "
                "or gauntlet cavity, padding allowance, "
                "comfort, pressure, range of motion, "
                "device eligibility, or clinical fit."
            ),
            (
                "The scale values and scores are "
                "experimental screening outputs only. "
                "They must not be used as independent "
                "prosthetic recommendations."
            ),
            "",
        ]
    )

    return "\n".join(
        lines
    )


def save_text_report(
    report: str,
) -> None:
    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )


def print_top_results(
    recipient: dict,
    comparisons: list[dict],
) -> None:
    print()
    print("========================================")
    print(" PROZI EXPERIMENTAL MODEL COMPARISON")
    print("========================================")

    print()
    print("Recipient")

    print(
        f"Length: "
        f"{recipient['length_mm']:.1f} mm"
    )

    print(
        f"Width: "
        f"{recipient['width_mm']:.1f} mm"
    )

    print(
        f"Thickness: "
        f"{recipient['thickness_mm']:.1f} mm"
    )

    print()
    print("Top results")

    for rank, result in enumerate(
        comparisons[:5],
        start=1,
    ):
        print()
        print(
            f"{rank}. {result['name']}"
        )

        print(
            f"Part: {result['part_type']}"
        )

        print(
            f"Geometric score: "
            f"{result['geometric_similarity_score']:.1f}%"
        )

        print(
            f"Experimental scale: "
            f"{result['experimental_scale_percent']:.1f}%"
        )

        print(
            f"Mean error: "
            f"{result['mean_error_percent']:.1f}%"
        )


def main() -> None:
    try:
        analyses = load_model_analyses()

        recipient = (
            load_recipient_measurements()
        )

        comparisons, failures = (
            compare_all_models(
                analyses,
                recipient,
            )
        )

        save_json_results(
            recipient,
            comparisons,
            failures,
        )

        report = build_text_report(
            recipient,
            comparisons,
            failures,
        )

        save_text_report(
            report
        )

        print_top_results(
            recipient,
            comparisons,
        )

        print()
        print(
            f"JSON results saved to: "
            f"{COMPARISON_PATH}"
        )

        print(
            f"Text report saved to: "
            f"{REPORT_PATH}"
        )

        print()
        print(
            "Important: These are experimental "
            "external-geometry comparisons, not "
            "prosthetic fit recommendations."
        )

    except (
        FileNotFoundError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        print(
            f"Model comparison failed: {error}"
        )


if __name__ == "__main__":
    main()