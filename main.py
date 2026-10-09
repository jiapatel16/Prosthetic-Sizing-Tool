from pathlib import Path
import os
import subprocess
import sys

REPORTS_FOLDER = Path("reports")
MODELS_FOLDER = Path("models")

PHOTO_PIPELINE = Path("photo_pipeline.py")
COMPARE_MODELS = Path("compare_models.py")
ANALYZE_LIBRARY = Path("analyze_model_library.py")
MODEL_LIBRARY = Path("model_library.py")

SIDE_REPORT = REPORTS_FOLDER / "side_photo_geometry_report.txt"
TOP_REPORT = REPORTS_FOLDER / "top_photo_geometry_report.txt"
COMBINED_REPORT = REPORTS_FOLDER / "combined_photo_report.txt"
COMPARISON_REPORT = REPORTS_FOLDER / "model_comparison_report.txt"
MODEL_ANALYSIS = MODELS_FOLDER / "model_analysis.json"


def print_header() -> None:
    print()
    print("=" * 48)
    print("                    PROZI")
    print(" Experimental Prosthetic Geometry Tool")
    print("=" * 48)
    print()


def pause() -> None:
    input("\nPress Enter to return to the main menu...")


def run_script(script: Path, *arguments: str) -> bool:
    if not script.exists():
        print(f"\nCould not find: {script}")
        return False

    print()
    print(f"Running {script.name} {' '.join(arguments)}...")
    print("-" * 48)

    result = subprocess.run(
        [sys.executable, str(script), *arguments],
        check=False,
    )

    if result.returncode != 0:
        print()
        print(f"{script.name} did not finish successfully.")
        return False

    return True


def open_file(path: Path) -> None:
    if not path.exists():
        print(f"\nCould not find: {path}")
        return

    try:
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        elif os.name == "nt":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
        print(f"\nOpened: {path}")
    except OSError as error:
        print(f"\nCould not open the file automatically: {error}")
        print(f"File location: {path}")


def start_new_recipient_analysis() -> None:
    print()
    print("NEW RECIPIENT ANALYSIS")
    print("-" * 48)
    print("Prozi will process TWO different images:")
    print("  1. Side view")
    print("  2. Top view")
    print()
    print("For EACH image, you will:")
    print("  • paint segmentation corrections")
    print("  • calibrate using the ruler")
    print("  • click the sleeve boundary")
    print("  • measure the residual limb")
    print()

    input("Press Enter to begin the SIDE view...")

    if not run_script(PHOTO_PIPELINE, "--view", "side"):
        pause()
        return

    if not SIDE_REPORT.exists():
        print("\nThe side-view report was not created.")
        pause()
        return

    input("\nPress Enter to begin the TOP view...")

    if not run_script(PHOTO_PIPELINE, "--view", "top"):
        pause()
        return

    if not TOP_REPORT.exists():
        print("\nThe top-view report was not created.")
        pause()
        return

    if not COMBINED_REPORT.exists():
        print()
        print("Both view reports exist, but the combined report is missing.")
        print("Run combine_views.py once.")
        pause()
        return

    print()
    print("=" * 48)
    print("TWO-VIEW PHOTO ANALYSIS COMPLETED")
    print("=" * 48)
    print(f"Side report: {SIDE_REPORT}")
    print(f"Top report: {TOP_REPORT}")
    print(f"Combined report: {COMBINED_REPORT}")

    if not MODEL_ANALYSIS.exists():
        response = input(
            "\nAnalyze the model library now? Enter Y or N: "
        ).strip().lower()
        if response == "y":
            if not run_script(ANALYZE_LIBRARY):
                pause()
                return

    if MODEL_ANALYSIS.exists():
        response = input(
            "\nUse BOTH views to compare against the model library now? "
            "Enter Y or N: "
        ).strip().lower()
        if response == "y":
            run_script(COMPARE_MODELS)

    pause()


def compare_current_recipient() -> None:
    if not SIDE_REPORT.exists() or not TOP_REPORT.exists():
        print("\nBoth side and top reports are required.")
        print("Complete a new recipient analysis first.")
        pause()
        return

    if not COMBINED_REPORT.exists():
        print("\nThe combined two-view report is missing.")
        pause()
        return

    if not MODEL_ANALYSIS.exists():
        print("\nThe model library has not been analyzed yet.")
        pause()
        return

    run_script(COMPARE_MODELS)
    pause()


def update_model_library() -> None:
    print()
    print("MODEL LIBRARY")
    print("-" * 48)
    print("1 = Register, scan, or display model files")
    print("2 = Analyze all registered STL models")
    print("3 = Return")

    choice = input("\nChoose an option: ").strip()

    if choice == "1":
        run_script(MODEL_LIBRARY)
    elif choice == "2":
        run_script(ANALYZE_LIBRARY)
    elif choice != "3":
        print("\nPlease choose 1, 2, or 3.")

    if choice != "3":
        pause()


def view_reports() -> None:
    available = [
        ("Combined two-view report", COMBINED_REPORT),
        ("Model comparison report", COMPARISON_REPORT),
        ("Side-view report", SIDE_REPORT),
        ("Top-view report", TOP_REPORT),
    ]
    existing = [(name, path) for name, path in available if path.exists()]

    print()
    print("SAVED REPORTS")
    print("-" * 48)

    if not existing:
        print("No reports found.")
        pause()
        return

    for index, (name, path) in enumerate(existing, start=1):
        print(f"{index} = {name}")
        print(f"    {path}")

    print(f"{len(existing) + 1} = Return")
    choice = input("\nChoose a report: ").strip()

    try:
        selected = int(choice) - 1
    except ValueError:
        pause()
        return

    if 0 <= selected < len(existing):
        open_file(existing[selected][1])
        pause()


def show_status() -> None:
    print()
    print("PROJECT STATUS")
    print("-" * 48)
    print(f"Side view: {'Ready' if SIDE_REPORT.exists() else 'Missing'}")
    print(f"Top view: {'Ready' if TOP_REPORT.exists() else 'Missing'}")
    print(
        "Combined two-view report: "
        f"{'Ready' if COMBINED_REPORT.exists() else 'Missing'}"
    )
    print(
        "Model analysis: "
        f"{'Ready' if MODEL_ANALYSIS.exists() else 'Missing'}"
    )
    print(
        "Model comparison: "
        f"{'Ready' if COMPARISON_REPORT.exists() else 'Missing'}"
    )
    pause()


def main() -> None:
    while True:
        print_header()
        print("1 = Start new TWO-VIEW recipient analysis")
        print("2 = Compare current two-view result")
        print("3 = Manage or analyze model library")
        print("4 = View saved reports")
        print("5 = View project status")
        print("6 = Exit")

        choice = input("\nChoose an option: ").strip()

        if choice == "1":
            start_new_recipient_analysis()
        elif choice == "2":
            compare_current_recipient()
        elif choice == "3":
            update_model_library()
        elif choice == "4":
            view_reports()
        elif choice == "5":
            show_status()
        elif choice == "6":
            print("\nClosing Prozi.")
            break
        else:
            print("\nPlease enter a number from 1 to 6.")
            pause()


if __name__ == "__main__":
    main()