from pathlib import Path
import json


MODELS_FOLDER = Path("models")
LIBRARY_PATH = Path("models/model_library.json")

SUPPORTED_CATEGORIES = {
    "upper_limb",
    "lower_limb",
}

FIT_KEYWORDS = (
    "gauntlet",
    "forearm",
    "socket",
    "arm",
    "cuff",
    "brace",
    "thermoform",
)

NON_FIT_KEYWORDS = (
    "finger",
    "thumb",
    "pin",
    "hinge",
    "palm",
    "tensioner",
    "whipple",
    "knuckle",
    "cap",
    "grip",
)


def ask_nonempty(prompt: str) -> str:
    while True:
        response = input(prompt).strip()

        if response:
            return response

        print("Please enter a value.")


def choose_category() -> str:
    while True:
        print()
        print("Choose the model category:")
        print("1 = Upper limb")
        print("2 = Lower limb")

        choice = input("Enter 1 or 2: ").strip()

        if choice == "1":
            return "upper_limb"

        if choice == "2":
            return "lower_limb"

        print("Please enter 1 or 2.")


def find_stl_files(
    category: str,
) -> list[Path]:
    category_folder = MODELS_FOLDER / category

    category_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    stl_files = list(
        category_folder.rglob("*.stl")
    )

    stl_files.extend(
        category_folder.rglob("*.STL")
    )

    unique_files = sorted(
        set(stl_files),
        key=lambda path: str(path).lower(),
    )

    return unique_files


def is_likely_fit_part(
    path: Path,
) -> bool:
    name = path.stem.lower()

    has_fit_keyword = any(
        keyword in name
        for keyword in FIT_KEYWORDS
    )

    has_non_fit_keyword = any(
        keyword in name
        for keyword in NON_FIT_KEYWORDS
    )

    return (
        has_fit_keyword
        and not has_non_fit_keyword
    )


def rank_stl_files(
    stl_files: list[Path],
) -> list[Path]:
    return sorted(
        stl_files,
        key=lambda path: (
            not is_likely_fit_part(path),
            str(path).lower(),
        ),
    )


def display_relative_path(
    path: Path,
) -> str:
    try:
        return str(
            path.relative_to(MODELS_FOLDER)
        )

    except ValueError:
        return str(path)


def choose_stl(
    stl_files: list[Path],
) -> Path | None:
    if not stl_files:
        print()
        print(
            "No STL files were found in this category."
        )
        return None

    ranked_files = rank_stl_files(
        stl_files
    )

    print()
    print("Available STL files:")
    print(
        "Parts marked [FIT PART] are more likely "
        "to affect recipient sizing."
    )

    for index, path in enumerate(
        ranked_files,
        start=1,
    ):
        label = (
            "[FIT PART]"
            if is_likely_fit_part(path)
            else "[OTHER]"
        )

        print(
            f"{index} = {label} "
            f"{display_relative_path(path)}"
        )

    while True:
        choice = input(
            "Choose an STL number, "
            "or enter C to cancel: "
        ).strip()

        if choice.lower() == "c":
            return None

        try:
            index = int(choice) - 1

            if 0 <= index < len(ranked_files):
                return ranked_files[index]

        except ValueError:
            pass

        print(
            "Please enter a valid model number "
            "or C to cancel."
        )


def infer_device_folder(
    stl_path: Path,
    category: str,
) -> str:
    category_folder = MODELS_FOLDER / category

    try:
        relative_path = stl_path.relative_to(
            category_folder
        )

    except ValueError:
        return stl_path.parent.name

    if len(relative_path.parts) > 1:
        return relative_path.parts[0]

    return stl_path.parent.name


def infer_part_type(
    stl_path: Path,
) -> str:
    filename = stl_path.stem.lower()

    if "gauntlet" in filename:
        return "gauntlet"

    if "forearm" in filename:
        return "forearm"

    if "socket" in filename:
        return "socket"

    if "cuff" in filename:
        return "cuff"

    if "thermoform" in filename:
        return "thermoform"

    if "arm" in filename:
        return "arm component"

    if "palm" in filename:
        return "palm"

    if "finger" in filename:
        return "finger"

    if "thumb" in filename:
        return "thumb"

    return "unspecified"


def load_library() -> list[dict]:
    if not LIBRARY_PATH.exists():
        return []

    try:
        with LIBRARY_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        if isinstance(data, list):
            return data

        print(
            "Warning: The model library was not "
            "stored as a list."
        )

    except json.JSONDecodeError:
        print(
            "Warning: Existing model library "
            "could not be read."
        )

    return []


def save_library(
    library: list[dict],
) -> None:
    LIBRARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with LIBRARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            library,
            file,
            indent=4,
        )


def register_model() -> None:
    category = choose_category()

    stl_files = find_stl_files(
        category
    )

    print()
    print(
        f"Found {len(stl_files)} STL file(s) "
        f"inside models/{category}/."
    )

    stl_path = choose_stl(
        stl_files
    )

    if stl_path is None:
        print()
        print("Model registration canceled.")
        return

    inferred_folder = infer_device_folder(
        stl_path,
        category,
    )

    inferred_part = infer_part_type(
        stl_path
    )

    print()
    print(
        f"Detected device folder: {inferred_folder}"
    )
    print(
        f"Detected part type: {inferred_part}"
    )

    default_name_prompt = (
        f"Enter the device name "
        f"[{inferred_folder}]: "
    )

    model_name_response = input(
        default_name_prompt
    ).strip()

    model_name = (
        model_name_response
        if model_name_response
        else inferred_folder
    )

    model_size = input(
        "Enter the model size or version "
        "[Original]: "
    ).strip()

    if not model_size:
        model_size = "Original"

    part_type_response = input(
        f"Enter the part type "
        f"[{inferred_part}]: "
    ).strip()

    part_type = (
        part_type_response
        if part_type_response
        else inferred_part
    )

    notes = input(
        "Enter optional notes: "
    ).strip()

    model_record = {
        "name": model_name,
        "size": model_size,
        "category": category,
        "device_folder": inferred_folder,
        "part_type": part_type,
        "stl_path": str(stl_path),
        "relative_stl_path":
            display_relative_path(stl_path),
        "likely_fit_part":
            is_likely_fit_part(stl_path),
        "notes": notes,
        "status": "experimental",
    }

    library = load_library()

    library = [
        model
        for model in library
        if model.get("stl_path")
        != str(stl_path)
    ]

    library.append(
        model_record
    )

    save_library(
        library
    )

    print()
    print("Model registered successfully.")
    print(f"Name: {model_name}")
    print(f"Size: {model_size}")
    print(f"Category: {category}")
    print(f"Device folder: {inferred_folder}")
    print(f"Part type: {part_type}")
    print(
        f"Likely fit part: "
        f"{model_record['likely_fit_part']}"
    )
    print(f"STL: {stl_path}")
    print(
        f"Library saved to: {LIBRARY_PATH}"
    )


def register_all_fit_parts() -> None:
    category = choose_category()

    stl_files = find_stl_files(
        category
    )

    fit_parts = [
        path
        for path in stl_files
        if is_likely_fit_part(path)
    ]

    if not fit_parts:
        print()
        print(
            "No likely fit-related STL files "
            "were found."
        )
        return

    library = load_library()

    existing_paths = {
        model.get("stl_path")
        for model in library
    }

    added_count = 0

    print()
    print(
        f"Found {len(fit_parts)} likely "
        "fit-related STL file(s)."
    )

    for stl_path in fit_parts:
        if str(stl_path) in existing_paths:
            continue

        device_folder = infer_device_folder(
            stl_path,
            category,
        )

        part_type = infer_part_type(
            stl_path
        )

        library.append(
            {
                "name": device_folder,
                "size": "Original",
                "category": category,
                "device_folder": device_folder,
                "part_type": part_type,
                "stl_path": str(stl_path),
                "relative_stl_path":
                    display_relative_path(stl_path),
                "likely_fit_part": True,
                "notes":
                    "Automatically registered "
                    "as a likely fit-related part.",
                "status": "experimental",
            }
        )

        existing_paths.add(
            str(stl_path)
        )

        added_count += 1

        print(
            f"Registered: "
            f"{display_relative_path(stl_path)}"
        )

    save_library(
        library
    )

    print()
    print(
        f"Automatically registered "
        f"{added_count} new fit-related model(s)."
    )
    print(
        f"Library saved to: {LIBRARY_PATH}"
    )


def display_library() -> None:
    library = load_library()

    if not library:
        print()
        print("The model library is empty.")
        return

    print()
    print("==============================")
    print(" PROZI MODEL LIBRARY")
    print("==============================")
    print(
        f"Registered models: {len(library)}"
    )

    for index, model in enumerate(
        library,
        start=1,
    ):
        print()
        print(f"Model {index}")
        print(
            f"Name: "
            f"{model.get('name', 'Unknown')}"
        )
        print(
            f"Size: "
            f"{model.get('size', 'Unknown')}"
        )
        print(
            f"Category: "
            f"{model.get('category', 'Unknown')}"
        )
        print(
            f"Device folder: "
            f"{model.get('device_folder', 'Unknown')}"
        )
        print(
            f"Part type: "
            f"{model.get('part_type', 'Unknown')}"
        )
        print(
            f"Likely fit part: "
            f"{model.get('likely_fit_part', False)}"
        )
        print(
            f"STL: "
            f"{model.get('relative_stl_path', model.get('stl_path'))}"
        )
        print(
            f"Status: "
            f"{model.get('status', 'Unknown')}"
        )

        notes = model.get("notes")

        if notes:
            print(f"Notes: {notes}")


def scan_library_summary() -> None:
    print()
    print("==============================")
    print(" MODEL FOLDER SCAN")
    print("==============================")

    for category in sorted(
        SUPPORTED_CATEGORIES
    ):
        stl_files = find_stl_files(
            category
        )

        fit_parts = [
            path
            for path in stl_files
            if is_likely_fit_part(path)
        ]

        print()
        print(
            f"{category}: "
            f"{len(stl_files)} total STL file(s)"
        )

        print(
            f"{category}: "
            f"{len(fit_parts)} likely "
            f"fit-related STL file(s)"
        )

        for path in fit_parts[:20]:
            print(
                f"  [FIT PART] "
                f"{display_relative_path(path)}"
            )

        if len(fit_parts) > 20:
            remaining = (
                len(fit_parts) - 20
            )

            print(
                f"  ...and {remaining} more"
            )


def main() -> None:
    while True:
        print()
        print("==============================")
        print(" PROZI MODEL LIBRARY")
        print("==============================")
        print("1 = Register one STL model")
        print(
            "2 = Automatically register "
            "likely fit-related parts"
        )
        print("3 = Display registered models")
        print("4 = Scan model folders")
        print("5 = Exit")

        choice = input(
            "Choose an option: "
        ).strip()

        if choice == "1":
            register_model()

        elif choice == "2":
            register_all_fit_parts()

        elif choice == "3":
            display_library()

        elif choice == "4":
            scan_library_summary()

        elif choice == "5":
            print("Model library closed.")
            break

        else:
            print(
                "Please enter 1, 2, 3, 4, or 5."
            )


if __name__ == "__main__":
    main()