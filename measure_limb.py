from pathlib import Path
import json

import cv2
import numpy as np


MASK_PATH = Path(
    "results/residual_limb_mask.png"
)

CALIBRATION_PATH = Path(
    "results/photo_calibration.json"
)

REPORT_PATH = Path(
    "reports/photo_geometry_report.txt"
)

ALIGNED_MASK_PATH = Path(
    "results/aligned_residual_limb_mask.png"
)


def load_mask() -> np.ndarray:
    mask = cv2.imread(
        str(MASK_PATH),
        cv2.IMREAD_GRAYSCALE,
    )

    if mask is None:
        raise FileNotFoundError(
            "Could not find "
            "results/residual_limb_mask.png. "
            "Run trim_limb.py first."
        )

    _, binary_mask = cv2.threshold(
        mask,
        127,
        255,
        cv2.THRESH_BINARY,
    )

    return binary_mask


def load_calibration() -> dict:
    if not CALIBRATION_PATH.exists():
        raise FileNotFoundError(
            "Could not find "
            "results/photo_calibration.json. "
            "Run calibrate_photo.py first."
        )

    with CALIBRATION_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        calibration = json.load(file)

    if "mm_per_pixel" not in calibration:
        raise ValueError(
            "The calibration file does not contain "
            "mm_per_pixel."
        )

    mm_per_pixel = float(
        calibration["mm_per_pixel"]
    )

    if mm_per_pixel <= 0:
        raise ValueError(
            "The calibration scale must be "
            "greater than zero."
        )

    return calibration


def find_largest_contour(
    mask: np.ndarray,
) -> np.ndarray:
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    if not contours:
        raise ValueError(
            "No residual-limb contour was found."
        )

    largest_contour = max(
        contours,
        key=cv2.contourArea,
    )

    if cv2.contourArea(largest_contour) < 100:
        raise ValueError(
            "The residual-limb region is too small "
            "to measure."
        )

    return largest_contour


def create_clean_mask(
    original_mask: np.ndarray,
    contour: np.ndarray,
) -> np.ndarray:
    clean_mask = np.zeros_like(
        original_mask
    )

    cv2.drawContours(
        clean_mask,
        [contour],
        -1,
        255,
        cv2.FILLED,
    )

    return clean_mask


def calculate_centroid(
    contour: np.ndarray,
) -> tuple[int, int]:
    moments = cv2.moments(
        contour
    )

    if moments["m00"] == 0:
        return 0, 0

    center_x = int(
        moments["m10"] / moments["m00"]
    )

    center_y = int(
        moments["m01"] / moments["m00"]
    )

    return center_x, center_y


def calculate_principal_axis(
    contour: np.ndarray,
) -> tuple[float, np.ndarray]:
    points = contour.reshape(
        -1,
        2,
    ).astype(np.float64)

    mean, eigenvectors, eigenvalues = cv2.PCACompute2(
        points,
        mean=None,
    )

    del mean
    del eigenvalues

    major_axis = eigenvectors[0]

    angle_radians = np.arctan2(
        major_axis[1],
        major_axis[0],
    )

    angle_degrees = float(
        np.degrees(angle_radians)
    )

    return angle_degrees, major_axis


def rotate_mask_to_vertical(
    mask: np.ndarray,
    contour: np.ndarray,
) -> tuple[np.ndarray, float]:
    axis_angle, _ = calculate_principal_axis(
        contour
    )

    rotation_angle = 90.0 - axis_angle

    height, width = mask.shape[:2]

    center = (
        width / 2.0,
        height / 2.0,
    )

    rotation_matrix = cv2.getRotationMatrix2D(
        center,
        rotation_angle,
        1.0,
    )

    cosine = abs(
        rotation_matrix[0, 0]
    )

    sine = abs(
        rotation_matrix[0, 1]
    )

    new_width = int(
        height * sine
        + width * cosine
    )

    new_height = int(
        height * cosine
        + width * sine
    )

    rotation_matrix[0, 2] += (
        new_width / 2.0
        - center[0]
    )

    rotation_matrix[1, 2] += (
        new_height / 2.0
        - center[1]
    )

    rotated_mask = cv2.warpAffine(
        mask,
        rotation_matrix,
        (
            new_width,
            new_height,
        ),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    _, rotated_mask = cv2.threshold(
        rotated_mask,
        127,
        255,
        cv2.THRESH_BINARY,
    )

    rotated_mask = crop_to_foreground(
        rotated_mask
    )

    # Make the narrower distal end appear at the top.
    rotated_mask = orient_distal_end_up(
        rotated_mask
    )

    return rotated_mask, rotation_angle


def crop_to_foreground(
    mask: np.ndarray,
    padding: int = 10,
) -> np.ndarray:
    rows, columns = np.where(
        mask > 0
    )

    if len(rows) == 0:
        raise ValueError(
            "Rotation removed the residual-limb mask."
        )

    top = max(
        0,
        int(rows.min()) - padding,
    )

    bottom = min(
        mask.shape[0],
        int(rows.max()) + padding + 1,
    )

    left = max(
        0,
        int(columns.min()) - padding,
    )

    right = min(
        mask.shape[1],
        int(columns.max()) + padding + 1,
    )

    return mask[
        top:bottom,
        left:right,
    ]


def calculate_band_width(
    mask: np.ndarray,
    start_percentage: float,
    end_percentage: float,
) -> float:
    occupied_rows = np.where(
        np.any(mask > 0, axis=1)
    )[0]

    if len(occupied_rows) == 0:
        return 0.0

    top = int(
        occupied_rows.min()
    )

    bottom = int(
        occupied_rows.max()
    )

    limb_height = (
        bottom
        - top
        + 1
    )

    start_row = int(
        top
        + start_percentage
        * limb_height
    )

    end_row = int(
        top
        + end_percentage
        * limb_height
    )

    widths = []

    for row_index in range(
        max(top, start_row),
        min(bottom + 1, end_row),
    ):
        white_pixels = np.where(
            mask[row_index] > 0
        )[0]

        if len(white_pixels) >= 2:
            width = (
                white_pixels[-1]
                - white_pixels[0]
                + 1
            )

            widths.append(
                float(width)
            )

    if not widths:
        return 0.0

    return float(
        np.median(widths)
    )


def orient_distal_end_up(
    mask: np.ndarray,
) -> np.ndarray:
    top_width = calculate_band_width(
        mask,
        0.05,
        0.20,
    )

    bottom_width = calculate_band_width(
        mask,
        0.80,
        0.95,
    )

    # The residual-limb tip is usually narrower.
    # Flip vertically when the narrower end is at the bottom.
    if bottom_width < top_width:
        return cv2.flip(
            mask,
            0,
        )

    return mask


def calculate_row_widths(
    mask: np.ndarray,
) -> list[float]:
    widths = []

    for row in mask:
        white_pixels = np.where(
            row > 0
        )[0]

        if len(white_pixels) >= 2:
            width = float(
                white_pixels[-1]
                - white_pixels[0]
                + 1
            )

            widths.append(
                width
            )

    return widths


def calculate_width_at_percentage(
    mask: np.ndarray,
    percentage: float,
    search_radius: int = 8,
) -> float:
    occupied_rows = np.where(
        np.any(mask > 0, axis=1)
    )[0]

    if len(occupied_rows) == 0:
        return 0.0

    top = int(
        occupied_rows.min()
    )

    bottom = int(
        occupied_rows.max()
    )

    target_row = int(
        round(
            top
            + percentage
            * (bottom - top)
        )
    )

    widths = []

    for row_index in range(
        max(
            top,
            target_row - search_radius,
        ),
        min(
            bottom + 1,
            target_row + search_radius + 1,
        ),
    ):
        white_pixels = np.where(
            mask[row_index] > 0
        )[0]

        if len(white_pixels) >= 2:
            width = float(
                white_pixels[-1]
                - white_pixels[0]
                + 1
            )

            widths.append(
                width
            )

    if not widths:
        return 0.0

    return float(
        np.median(widths)
    )


def classify_taper(
    taper_ratio: float,
) -> str:
    if taper_ratio <= 0:
        return "Unable to classify"

    if taper_ratio < 0.70:
        return "Strongly tapered"

    if taper_ratio < 0.90:
        return "Moderately tapered"

    if taper_ratio <= 1.10:
        return "Nearly cylindrical"

    return "Wider toward the distal end"


def calculate_measurements(
    original_mask: np.ndarray,
    aligned_mask: np.ndarray,
    original_contour: np.ndarray,
    rotation_angle: float,
) -> dict:
    aligned_contour = find_largest_contour(
        aligned_mask
    )

    x, y, width, height = cv2.boundingRect(
        aligned_contour
    )

    del x
    del y

    occupied_rows = np.where(
        np.any(aligned_mask > 0, axis=1)
    )[0]

    if len(occupied_rows) == 0:
        raise ValueError(
            "No measurable rows were found."
        )

    estimated_length = float(
        occupied_rows.max()
        - occupied_rows.min()
        + 1
    )

    row_widths = calculate_row_widths(
        aligned_mask
    )

    if not row_widths:
        raise ValueError(
            "No measurable widths were found."
        )

    maximum_width = float(
        max(row_widths)
    )

    average_width = float(
        np.mean(row_widths)
    )

    median_width = float(
        np.median(row_widths)
    )

    distal_width = calculate_width_at_percentage(
        aligned_mask,
        0.10,
    )

    quarter_width = calculate_width_at_percentage(
        aligned_mask,
        0.25,
    )

    middle_width = calculate_width_at_percentage(
        aligned_mask,
        0.50,
    )

    three_quarter_width = (
        calculate_width_at_percentage(
            aligned_mask,
            0.75,
        )
    )

    proximal_width = calculate_width_at_percentage(
        aligned_mask,
        0.90,
    )

    if proximal_width > 0:
        taper_ratio = (
            distal_width
            / proximal_width
        )
    else:
        taper_ratio = 0.0

    center_x, center_y = calculate_centroid(
        original_contour
    )

    area_pixels = float(
        cv2.countNonZero(
            original_mask
        )
    )

    return {
        "bounding_width_px":
            float(width),

        "bounding_height_px":
            float(height),

        "estimated_length_px":
            estimated_length,

        "maximum_width_px":
            maximum_width,

        "average_width_px":
            average_width,

        "median_width_px":
            median_width,

        "distal_width_px":
            distal_width,

        "quarter_width_px":
            quarter_width,

        "middle_width_px":
            middle_width,

        "three_quarter_width_px":
            three_quarter_width,

        "proximal_width_px":
            proximal_width,

        "taper_ratio":
            taper_ratio,

        "taper_classification":
            classify_taper(
                taper_ratio
            ),

        "area_px_squared":
            area_pixels,

        "rotation_applied_degrees":
            rotation_angle,

        "centroid_x":
            center_x,

        "centroid_y":
            center_y,
    }


def convert_to_millimeters(
    measurements: dict,
    mm_per_pixel: float,
) -> dict:
    converted = dict(
        measurements
    )

    length_keys = [
        "bounding_width",
        "bounding_height",
        "estimated_length",
        "maximum_width",
        "average_width",
        "median_width",
        "distal_width",
        "quarter_width",
        "middle_width",
        "three_quarter_width",
        "proximal_width",
    ]

    for key in length_keys:
        converted[f"{key}_mm"] = (
            measurements[f"{key}_px"]
            * mm_per_pixel
        )

    converted["area_mm_squared"] = (
        measurements["area_px_squared"]
        * (mm_per_pixel ** 2)
    )

    converted["mm_per_pixel"] = (
        mm_per_pixel
    )

    return converted


def print_results(
    measurements: dict,
) -> None:
    print()
    print("==============================")
    print(" PROZI PHOTO ANALYSIS")
    print("==============================")

    print()
    print("Measurement Region")
    print("Residual limb only")
    print("Long-axis alignment applied")

    print()
    print("Residual-Limb Geometry")

    print(
        f"Estimated Length : "
        f"{measurements['estimated_length_px']:.1f} px | "
        f"{measurements['estimated_length_mm']:.1f} mm"
    )

    print(
        f"Maximum Width    : "
        f"{measurements['maximum_width_px']:.1f} px | "
        f"{measurements['maximum_width_mm']:.1f} mm"
    )

    print(
        f"Average Width    : "
        f"{measurements['average_width_px']:.1f} px | "
        f"{measurements['average_width_mm']:.1f} mm"
    )

    print(
        f"Median Width     : "
        f"{measurements['median_width_px']:.1f} px | "
        f"{measurements['median_width_mm']:.1f} mm"
    )

    print(
        f"Area             : "
        f"{measurements['area_px_squared']:.1f} px^2 | "
        f"{measurements['area_mm_squared']:.1f} mm^2"
    )

    print()
    print("Width Profile")

    print(
        f"Distal Width     : "
        f"{measurements['distal_width_px']:.1f} px | "
        f"{measurements['distal_width_mm']:.1f} mm"
    )

    print(
        f"25% Width        : "
        f"{measurements['quarter_width_px']:.1f} px | "
        f"{measurements['quarter_width_mm']:.1f} mm"
    )

    print(
        f"Middle Width     : "
        f"{measurements['middle_width_px']:.1f} px | "
        f"{measurements['middle_width_mm']:.1f} mm"
    )

    print(
        f"75% Width        : "
        f"{measurements['three_quarter_width_px']:.1f} px | "
        f"{measurements['three_quarter_width_mm']:.1f} mm"
    )

    print(
        f"Proximal Width   : "
        f"{measurements['proximal_width_px']:.1f} px | "
        f"{measurements['proximal_width_mm']:.1f} mm"
    )

    print(
        f"Taper Ratio      : "
        f"{measurements['taper_ratio']:.3f}"
    )

    print(
        f"Shape            : "
        f"{measurements['taper_classification']}"
    )

    print()
    print(
        f"Internal Rotation: "
        f"{measurements['rotation_applied_degrees']:.2f} degrees"
    )

    print(
        f"Original Centroid: "
        f"({measurements['centroid_x']}, "
        f"{measurements['centroid_y']})"
    )

    print(
        f"Image Scale      : "
        f"{measurements['mm_per_pixel']:.6f} mm/px"
    )


def build_report(
    measurements: dict,
) -> str:
    return (
        "==============================\n"
        " PROZI PHOTO ANALYSIS\n"
        "==============================\n"
        "\n"
        "Measurement Region\n"
        "------------------\n"
        "Residual limb only\n"
        "The mask was aligned with its principal "
        "long axis before measurement.\n"
        "\n"
        "Residual-Limb Geometry\n"
        "----------------------\n"
        f"Estimated length: "
        f"{measurements['estimated_length_px']:.1f} px | "
        f"{measurements['estimated_length_mm']:.1f} mm\n"
        f"Maximum width: "
        f"{measurements['maximum_width_px']:.1f} px | "
        f"{measurements['maximum_width_mm']:.1f} mm\n"
        f"Average width: "
        f"{measurements['average_width_px']:.1f} px | "
        f"{measurements['average_width_mm']:.1f} mm\n"
        f"Median width: "
        f"{measurements['median_width_px']:.1f} px | "
        f"{measurements['median_width_mm']:.1f} mm\n"
        f"Area: "
        f"{measurements['area_px_squared']:.1f} px^2 | "
        f"{measurements['area_mm_squared']:.1f} mm^2\n"
        "\n"
        "Width Profile\n"
        "-------------\n"
        f"Distal width: "
        f"{measurements['distal_width_mm']:.1f} mm\n"
        f"25% width: "
        f"{measurements['quarter_width_mm']:.1f} mm\n"
        f"Middle width: "
        f"{measurements['middle_width_mm']:.1f} mm\n"
        f"75% width: "
        f"{measurements['three_quarter_width_mm']:.1f} mm\n"
        f"Proximal width: "
        f"{measurements['proximal_width_mm']:.1f} mm\n"
        f"Taper ratio: "
        f"{measurements['taper_ratio']:.3f}\n"
        f"Shape classification: "
        f"{measurements['taper_classification']}\n"
        "\n"
        "Processing Information\n"
        "----------------------\n"
        f"Internal rotation applied: "
        f"{measurements['rotation_applied_degrees']:.2f} degrees\n"
        f"Image scale: "
        f"{measurements['mm_per_pixel']:.6f} mm/px\n"
        "\n"
        "Important Note\n"
        "--------------\n"
        "These measurements are experimental "
        "computer-vision estimates. They should not "
        "be used independently for clinical or "
        "prosthetic-fitting decisions.\n"
    )


def save_results(
    measurements: dict,
    aligned_mask: np.ndarray,
) -> None:
    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ALIGNED_MASK_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            build_report(
                measurements
            )
        )

    cv2.imwrite(
        str(ALIGNED_MASK_PATH),
        aligned_mask,
    )

    print()
    print(
        f"Report saved to: {REPORT_PATH}"
    )

    print(
        f"Aligned mask saved to: {ALIGNED_MASK_PATH}"
    )


def main() -> None:
    try:
        original_mask = load_mask()

        calibration = load_calibration()

        mm_per_pixel = float(
            calibration["mm_per_pixel"]
        )

        original_contour = find_largest_contour(
            original_mask
        )

        clean_mask = create_clean_mask(
            original_mask,
            original_contour,
        )

        aligned_mask, rotation_angle = (
            rotate_mask_to_vertical(
                clean_mask,
                original_contour,
            )
        )

        measurements = calculate_measurements(
            clean_mask,
            aligned_mask,
            original_contour,
            rotation_angle,
        )

        measurements = convert_to_millimeters(
            measurements,
            mm_per_pixel,
        )

        print_results(
            measurements
        )

        save_results(
            measurements,
            aligned_mask,
        )

    except (
        FileNotFoundError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        print(
            f"Measurement failed: {error}"
        )


if __name__ == "__main__":
    main()