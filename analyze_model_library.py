from pathlib import Path
import json

import numpy as np
import trimesh


LIBRARY_PATH = Path(
    "models/model_library.json"
)

ANALYSIS_PATH = Path(
    "models/model_analysis.json"
)


def load_library() -> list[dict]:
    if not LIBRARY_PATH.exists():
        raise FileNotFoundError(
            "Could not find models/model_library.json. "
            "Run model_library.py first."
        )

    with LIBRARY_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            "model_library.json must contain a list."
        )

    return data


def load_mesh(
    stl_path: Path,
) -> trimesh.Trimesh:
    loaded = trimesh.load(
        stl_path,
        force="mesh",
    )

    if isinstance(
        loaded,
        trimesh.Scene,
    ):
        if not loaded.geometry:
            raise ValueError(
                "The STL did not contain usable geometry."
            )

        mesh = trimesh.util.concatenate(
            tuple(
                loaded.geometry.values()
            )
        )

    else:
        mesh = loaded

    if mesh.is_empty:
        raise ValueError(
            "The STL mesh is empty."
        )

    return mesh


def analyze_mesh(
    mesh: trimesh.Trimesh,
) -> dict:
    bounds = np.asarray(
        mesh.bounds,
        dtype=float,
    )

    extents = np.asarray(
        mesh.extents,
        dtype=float,
    )

    sorted_extents = np.sort(
        extents
    )[::-1]

    length = float(
        sorted_extents[0]
    )

    width = float(
        sorted_extents[1]
    )

    thickness = float(
        sorted_extents[2]
    )

    surface_area = float(
        mesh.area
    )

    volume = None

    if mesh.is_watertight:
        volume = float(
            abs(mesh.volume)
        )

    return {
        "bounds_min": [
            float(value)
            for value in bounds[0]
        ],
        "bounds_max": [
            float(value)
            for value in bounds[1]
        ],
        "axis_extents": {
            "x": float(extents[0]),
            "y": float(extents[1]),
            "z": float(extents[2]),
        },
        "dimensions_mm": {
            "length": length,
            "width": width,
            "thickness": thickness,
        },
        "surface_area_mm_squared":
            surface_area,
        "volume_mm_cubed":
            volume,
        "vertex_count":
            int(len(mesh.vertices)),
        "face_count":
            int(len(mesh.faces)),
        "is_watertight":
            bool(mesh.is_watertight),
    }


def analyze_model(
    model: dict,
) -> dict:
    stl_path_value = model.get(
        "stl_path"
    )

    if not stl_path_value:
        raise ValueError(
            "Model record has no stl_path."
        )

    stl_path = Path(
        stl_path_value
    )

    if not stl_path.exists():
        raise FileNotFoundError(
            f"Could not find {stl_path}"
        )

    mesh = load_mesh(
        stl_path
    )

    geometry = analyze_mesh(
        mesh
    )

    return {
        "name":
            model.get("name", "Unknown"),
        "size":
            model.get("size", "Unknown"),
        "category":
            model.get("category", "Unknown"),
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
            str(stl_path),
        "relative_stl_path":
            model.get(
                "relative_stl_path",
                str(stl_path),
            ),
        "likely_fit_part":
            model.get(
                "likely_fit_part",
                False,
            ),
        "status":
            "analyzed",
        **geometry,
    }


def save_analysis(
    analyses: list[dict],
) -> None:
    ANALYSIS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with ANALYSIS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            analyses,
            file,
            indent=4,
        )


def print_analysis(
    analysis: dict,
) -> None:
    dimensions = analysis[
        "dimensions_mm"
    ]

    print()
    print(
        f"Device: "
        f"{analysis['name']}"
    )

    print(
        f"Part: "
        f"{analysis['part_type']}"
    )

    print(
        f"Length: "
        f"{dimensions['length']:.2f} mm"
    )

    print(
        f"Width: "
        f"{dimensions['width']:.2f} mm"
    )

    print(
        f"Thickness: "
        f"{dimensions['thickness']:.2f} mm"
    )

    print(
        f"Surface area: "
        f"{analysis['surface_area_mm_squared']:.2f} mm^2"
    )

    if analysis[
        "volume_mm_cubed"
    ] is None:
        print(
            "Volume: unavailable "
            "(mesh is not watertight)"
        )

    else:
        print(
            f"Volume: "
            f"{analysis['volume_mm_cubed']:.2f} mm^3"
        )

    print(
        f"Watertight: "
        f"{analysis['is_watertight']}"
    )


def main() -> None:
    try:
        library = load_library()

        if not library:
            print(
                "The model library is empty."
            )
            return

        analyses = []
        failures = []

        print()
        print("==============================")
        print(" PROZI MODEL ANALYSIS")
        print("==============================")
        print(
            f"Registered models: "
            f"{len(library)}"
        )

        for index, model in enumerate(
            library,
            start=1,
        ):
            name = model.get(
                "name",
                "Unknown",
            )

            print()
            print(
                f"Analyzing {index}/"
                f"{len(library)}: {name}"
            )

            try:
                analysis = analyze_model(
                    model
                )

                analyses.append(
                    analysis
                )

                print_analysis(
                    analysis
                )

            except (
                FileNotFoundError,
                ValueError,
                TypeError,
                trimesh.exceptions.MeshError,
            ) as error:
                failures.append(
                    {
                        "name": name,
                        "stl_path":
                            model.get(
                                "stl_path",
                                "",
                            ),
                        "error":
                            str(error),
                    }
                )

                print(
                    f"Skipped: {error}"
                )

        save_analysis(
            analyses
        )

        print()
        print("==============================")
        print(" ANALYSIS COMPLETED")
        print("==============================")

        print(
            f"Successfully analyzed: "
            f"{len(analyses)}"
        )

        print(
            f"Failed: "
            f"{len(failures)}"
        )

        print(
            f"Saved to: "
            f"{ANALYSIS_PATH}"
        )

        if failures:
            print()
            print("Failed models:")

            for failure in failures:
                print(
                    f"- {failure['name']}: "
                    f"{failure['error']}"
                )

    except (
        FileNotFoundError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(
            f"Model analysis failed: {error}"
        )


if __name__ == "__main__":
    main()