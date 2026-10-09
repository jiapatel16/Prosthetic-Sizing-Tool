from pathlib import Path
import re


TOP_REPORT_PATH = Path(
    "reports/top_photo_geometry_report.txt"
)

SIDE_REPORT_PATH = Path(
    "reports/side_photo_geometry_report.txt"
)

COMBINED_REPORT_PATH = Path(
    "reports/combined_photo_report.txt"
)


def read_report(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}"
        )

    return path.read_text(
        encoding="utf-8"
    )


def extract_number(
    report: str,
    label: str,
) -> float:
    pattern = (
        rf"{re.escape(label)}:\s*"
        rf"(?:[\d.]+\s*px\s*\|\s*)?"
        rf"([\d.]+)\s*mm"
    )

    match = re.search(
        pattern,
        report,
        re.IGNORECASE,
    )

    if match is None:
        raise ValueError(
            f"Could not find '{label}' in the report."
        )

    return float(
        match.group(1)
    )


def extract_text(
    report: str,
    label: str,
) -> str:
    pattern = (
        rf"{re.escape(label)}:\s*(.+)"
    )

    match = re.search(
        pattern,
        report,
        re.IGNORECASE,
    )

    if match is None:
        return "Not available"

    return match.group(1).strip()


def main() -> None:
    try:
        top_report = read_report(
            TOP_REPORT_PATH
        )

        side_report = read_report(
            SIDE_REPORT_PATH
        )

        top_length = extract_number(
            top_report,
            "Estimated length",
        )

        side_length = extract_number(
            side_report,
            "Estimated length",
        )

        top_max_width = extract_number(
            top_report,
            "Maximum width",
        )

        side_max_width = extract_number(
            side_report,
            "Maximum width",
        )

        top_middle_width = extract_number(
            top_report,
            "Middle width",
        )

        side_middle_width = extract_number(
            side_report,
            "Middle width",
        )

        top_proximal_width = extract_number(
            top_report,
            "Proximal width",
        )

        side_proximal_width = extract_number(
            side_report,
            "Proximal width",
        )

        top_shape = extract_text(
            top_report,
            "Shape classification",
        )

        side_shape = extract_text(
            side_report,
            "Shape classification",
        )

        average_length = (
            top_length + side_length
        ) / 2.0

        length_difference = abs(
            top_length - side_length
        )

        combined_report = (
            "====================================\n"
            " PROZI COMBINED PHOTO ANALYSIS\n"
            "====================================\n"
            "\n"
            "Residual-Limb Length\n"
            "--------------------\n"
            f"Top-view estimate: {top_length:.1f} mm\n"
            f"Side-view estimate: {side_length:.1f} mm\n"
            f"Average estimate: {average_length:.1f} mm\n"
            f"Difference between views: "
            f"{length_difference:.1f} mm\n"
            "\n"
            "Cross-Section Measurements\n"
            "--------------------------\n"
            f"Maximum top-view width: "
            f"{top_max_width:.1f} mm\n"
            f"Maximum side-view thickness: "
            f"{side_max_width:.1f} mm\n"
            "\n"
            f"Middle top-view width: "
            f"{top_middle_width:.1f} mm\n"
            f"Middle side-view thickness: "
            f"{side_middle_width:.1f} mm\n"
            "\n"
            f"Proximal top-view width: "
            f"{top_proximal_width:.1f} mm\n"
            f"Proximal side-view thickness: "
            f"{side_proximal_width:.1f} mm\n"
            "\n"
            "Shape Analysis\n"
            "--------------\n"
            f"Top-view shape: {top_shape}\n"
            f"Side-view shape: {side_shape}\n"
            "\n"
            "Interpretation\n"
            "--------------\n"
            "The top view estimates medial-lateral width.\n"
            "The side view estimates anterior-posterior "
            "thickness.\n"
            "The average of the two length measurements is "
            "reported as an experimental combined estimate.\n"
            "\n"
            "Important Note\n"
            "--------------\n"
            "These measurements are experimental "
            "computer-vision estimates and must be "
            "validated against direct physical "
            "measurements before prosthetic sizing.\n"
        )

        COMBINED_REPORT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        COMBINED_REPORT_PATH.write_text(
            combined_report,
            encoding="utf-8",
        )

        print(combined_report)

        print(
            f"Combined report saved to: "
            f"{COMBINED_REPORT_PATH}"
        )

    except (
        FileNotFoundError,
        ValueError,
    ) as error:
        print(
            f"Could not combine reports: {error}"
        )


if __name__ == "__main__":
    main()