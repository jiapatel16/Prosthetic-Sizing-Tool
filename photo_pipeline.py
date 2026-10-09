from pathlib import Path
import argparse
import shutil
import subprocess
import sys

RESULTS_FOLDER = Path("results")
REPORTS_FOLDER = Path("reports")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--view",
        choices=("side", "top"),
        help="Force the workflow to use the side or top view.",
    )
    return parser.parse_args()


def choose_view() -> str:
    while True:
        print()
        print("==============================")
        print(" PROZI PHOTO WORKFLOW")
        print("==============================")
        print("1 = Side view")
        print("2 = Top view")

        choice = input("Choose a view: ").strip()

        if choice == "1":
            return "side"
        if choice == "2":
            return "top"

        print("Please enter 1 or 2.")


def run_script(
    script_name: str,
    input_text: str | None = None,
) -> bool:
    print()
    print(f"Running {script_name}...")
    print("-" * 40)

    result = subprocess.run(
        [sys.executable, script_name],
        input=input_text,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        print()
        print(f"{script_name} did not finish successfully.")
        return False

    return True


def save_file(source: Path, destination: Path) -> None:
    if not source.exists():
        print(f"Warning: Could not find {source}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        destination.unlink()

    shutil.move(str(source), str(destination))
    print(f"Saved: {destination}")


def save_view_results(view: str) -> None:
    print()
    print(f"Saving {view}-view results...")

    result_files = {
        RESULTS_FOLDER / "limb_mask.png":
            RESULTS_FOLDER / f"{view}_limb_mask.png",
        RESULTS_FOLDER / "detected_limb.jpg":
            RESULTS_FOLDER / f"{view}_detected_limb.jpg",
        RESULTS_FOLDER / "segmented_limb.png":
            RESULTS_FOLDER / f"{view}_segmented_limb.png",
        RESULTS_FOLDER / "residual_limb_mask.png":
            RESULTS_FOLDER / f"{view}_residual_limb_mask.png",
        RESULTS_FOLDER / "residual_limb_detected.jpg":
            RESULTS_FOLDER / f"{view}_residual_limb_detected.jpg",
        RESULTS_FOLDER / "aligned_residual_limb_mask.png":
            RESULTS_FOLDER / f"{view}_aligned_residual_limb_mask.png",
        RESULTS_FOLDER / "photo_calibration.json":
            RESULTS_FOLDER / f"{view}_photo_calibration.json",
        REPORTS_FOLDER / "photo_geometry_report.txt":
            REPORTS_FOLDER / f"{view}_photo_geometry_report.txt",
    }

    for source, destination in result_files.items():
        save_file(source, destination)


def confirm_correct_photo(view: str) -> bool:
    expected_photo = (
        "photos/arm_side.jpg"
        if view == "side"
        else "photos/arm_top.jpg"
    )

    print()
    print(f"You are processing the {view.upper()} view.")
    print(f"Select this image in BOTH file-selection windows:")
    print(expected_photo)
    print()
    

    response = input("\nContinue? Enter Y or N: ").strip().lower()
    return response == "y"


def main() -> None:
    args = parse_args()
    view = args.view if args.view else choose_view()

    if not confirm_correct_photo(view):
        print("Workflow canceled.")
        return

    required_scripts = [
        "segment_limb.py",
        "calibrate_photo.py",
        "trim_limb.py",
        "measure_limb.py",
    ]

    for script in required_scripts:
        if not Path(script).exists():
            print(f"Could not find {script}.")
            return

    if not run_script("segment_limb.py"):
        print("Workflow stopped during segmentation.")
        return

    if not run_script("calibrate_photo.py"):
        print("Workflow stopped during calibration.")
        return

    print()
    print("=" * 48)
    print("SLEEVE BOUNDARY STEP")
    print("=" * 48)
    print(
        "Click where the sleeve ends so Prozi removes the "
        "upper arm and keeps only the residual limb."
    )
    print(
        "Side view uses a vertical boundary; "
        "top view uses a horizontal boundary."
    )

    trim_choice = "1\n" if view == "side" else "2\n"

    if not run_script("trim_limb.py", input_text=trim_choice):
        print("Workflow stopped during sleeve trimming.")
        return

    if not (RESULTS_FOLDER / "residual_limb_mask.png").exists():
        print()
        print("The sleeve-trimmed mask was not created.")
        print("Measurement was stopped to avoid using the full arm.")
        return

    if not run_script("measure_limb.py"):
        print("Workflow stopped during measurement.")
        return

    save_view_results(view)

    print()
    print("==============================")
    print(" VIEW WORKFLOW COMPLETED")
    print("==============================")
    print(f"Saved all results with the prefix: {view}_")

    top_report = REPORTS_FOLDER / "top_photo_geometry_report.txt"
    side_report = REPORTS_FOLDER / "side_photo_geometry_report.txt"

    if top_report.exists() and side_report.exists():
        print()
        print("Both DIFFERENT views are available.")
        print("Creating the combined two-view report...")

        if not Path("combine_views.py").exists():
            print("Could not find combine_views.py.")
            return

        run_script("combine_views.py")
    else:
        missing = "top" if not top_report.exists() else "side"
        print()
        print(f"The {missing} view is still missing.")
        print("Run the main workflow for that view next.")


if __name__ == "__main__":
    main()