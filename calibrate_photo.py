from pathlib import Path
import json
from tkinter import Tk, filedialog

import cv2
import numpy as np


CALIBRATION_PATH = Path("results/photo_calibration.json")

clicked_points: list[tuple[int, int]] = []
mouse_position: tuple[int, int] = (0, 0)


def choose_photo() -> Path | None:
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    selected_file = filedialog.askopenfilename(
        title="Choose the photo containing the ruler",
        initialdir="photos",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png"),
            ("All files", "*.*"),
        ],
    )

    root.destroy()

    if not selected_file:
        return None

    return Path(selected_file)


def resize_for_display(
    image: np.ndarray,
    maximum_width: int = 1100,
    maximum_height: int = 800,
) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]

    scale = min(
        maximum_width / width,
        maximum_height / height,
        1.0,
    )

    if scale == 1.0:
        return image.copy(), scale

    resized = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )

    return resized, scale


def draw_crosshair(
    image: np.ndarray,
    point: tuple[int, int],
    size: int = 12,
) -> None:
    x, y = point

    cv2.line(
        image,
        (x - size, y),
        (x + size, y),
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.line(
        image,
        (x, y - size),
        (x, y + size),
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.circle(
        image,
        point,
        2,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )


def create_zoom_panel(
    source_image: np.ndarray,
    center: tuple[int, int],
    panel_size: int = 220,
    crop_radius: int = 16,
) -> np.ndarray:
    x, y = center
    height, width = source_image.shape[:2]

    x1 = max(0, x - crop_radius)
    y1 = max(0, y - crop_radius)
    x2 = min(width, x + crop_radius + 1)
    y2 = min(height, y + crop_radius + 1)

    crop = source_image[y1:y2, x1:x2]

    if crop.size == 0:
        return np.zeros(
            (panel_size, panel_size, 3),
            dtype=np.uint8,
        )

    zoom = cv2.resize(
        crop,
        (panel_size, panel_size),
        interpolation=cv2.INTER_NEAREST,
    )

    center_x = panel_size // 2
    center_y = panel_size // 2

    cv2.line(
        zoom,
        (center_x, 0),
        (center_x, panel_size - 1),
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.line(
        zoom,
        (0, center_y),
        (panel_size - 1, center_y),
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.circle(
        zoom,
        (center_x, center_y),
        3,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.rectangle(
        zoom,
        (0, 0),
        (panel_size - 1, panel_size - 1),
        (255, 255, 255),
        1,
    )

    return zoom


def show_confirmation(
    display_image: np.ndarray,
    points: list[tuple[int, int]],
) -> bool:
    panel_size = 240
    margin = 20

    confirmation = np.zeros(
        (panel_size + 110, panel_size * 2 + margin * 3, 3),
        dtype=np.uint8,
    )

    first_zoom = create_zoom_panel(
        display_image,
        points[0],
        panel_size=panel_size,
        crop_radius=14,
    )

    second_zoom = create_zoom_panel(
        display_image,
        points[1],
        panel_size=panel_size,
        crop_radius=14,
    )

    confirmation[
        70:70 + panel_size,
        margin:margin + panel_size,
    ] = first_zoom

    confirmation[
        70:70 + panel_size,
        margin * 2 + panel_size:margin * 2 + panel_size * 2,
    ] = second_zoom

    cv2.putText(
        confirmation,
        "Point 1",
        (margin, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        confirmation,
        "Point 2",
        (margin * 2 + panel_size, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        confirmation,
        "Press Y to accept or N to select again",
        (margin, panel_size + 345 - panel_size),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    window_name = "Confirm Ruler Points"
    cv2.namedWindow(window_name)

    while True:
        cv2.imshow(window_name, confirmation)
        key = cv2.waitKey(20) & 0xFF

        if key in (ord("y"), ord("Y"), 10, 13):
            cv2.destroyWindow(window_name)
            return True

        if key in (ord("n"), ord("N"), 27):
            cv2.destroyWindow(window_name)
            return False


def select_ruler_points(
    image: np.ndarray,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    global clicked_points
    global mouse_position

    clicked_points = []
    mouse_position = (0, 0)

    display_image, display_scale = resize_for_display(image)
    window_name = "Select Two Ruler Marks"

    def mouse_callback(
        event: int,
        x: int,
        y: int,
        flags: int,
        parameter: object,
    ) -> None:
        del flags
        del parameter

        global mouse_position
        mouse_position = (x, y)

        if (
            event == cv2.EVENT_LBUTTONDOWN
            and len(clicked_points) < 2
        ):
            clicked_points.append((x, y))

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print()
    print("Move the cursor over the ruler to use the live zoom.")
    print("Click two ruler marks with a known distance between them.")
    print("The exact selected pixel is the center of the crosshair.")
    print("Press R to restart.")
    print("Press Enter after selecting both points.")
    print("Press Escape to cancel.")

    while True:
        preview = display_image.copy()

        draw_crosshair(
            preview,
            mouse_position,
            size=10,
        )

        for point in clicked_points:
            cv2.circle(
                preview,
                point,
                3,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            draw_crosshair(
                preview,
                point,
                size=7,
            )

        if len(clicked_points) == 2:
            cv2.line(
                preview,
                clicked_points[0],
                clicked_points[1],
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            pixel_distance = np.linalg.norm(
                np.array(clicked_points[1], dtype=float)
                - np.array(clicked_points[0], dtype=float)
            )

            cv2.putText(
                preview,
                f"Displayed distance: {pixel_distance:.1f} px",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

        zoom_panel = create_zoom_panel(
            display_image,
            mouse_position,
            panel_size=220,
            crop_radius=16,
        )

        preview_height, preview_width = preview.shape[:2]
        panel_height, panel_width = zoom_panel.shape[:2]

        panel_x = max(0, preview_width - panel_width - 10)
        panel_y = 10

        if panel_y + panel_height <= preview_height:
            preview[
                panel_y:panel_y + panel_height,
                panel_x:panel_x + panel_width,
            ] = zoom_panel

            cv2.putText(
                preview,
                "LIVE ZOOM",
                (panel_x + 8, panel_y + panel_height - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )

        cv2.imshow(window_name, preview)

        key = cv2.waitKey(20) & 0xFF

        if key in (ord("r"), ord("R")):
            clicked_points = []

        elif key in (10, 13) and len(clicked_points) == 2:
            accepted = show_confirmation(
                display_image,
                clicked_points,
            )

            if accepted:
                break

            clicked_points = []

        elif key == 27:
            cv2.destroyAllWindows()
            return None

    cv2.destroyAllWindows()

    first_point = (
        int(round(clicked_points[0][0] / display_scale)),
        int(round(clicked_points[0][1] / display_scale)),
    )

    second_point = (
        int(round(clicked_points[1][0] / display_scale)),
        int(round(clicked_points[1][1] / display_scale)),
    )

    return first_point, second_point


def request_known_distance() -> float:
    presets = {
        "1": 50.0,
        "2": 90.0,
        "3": 100.0,
        "4": 150.0,
    }

    while True:
        print()
        print("Known ruler distance:")
        print("1 = 50 mm")
        print("2 = 90 mm")
        print("3 = 100 mm")
        print("4 = 150 mm")
        print("5 = Custom")

        choice = input("Choose 1-5: ").strip()

        if choice in presets:
            return presets[choice]

        if choice == "5":
            response = input(
                "Enter the real distance in millimeters: "
            )

            try:
                distance_mm = float(response)

                if distance_mm <= 0:
                    print("Distance must be greater than zero.")
                    continue

                return distance_mm

            except ValueError:
                print("Please enter a valid number.")
                continue

        print("Please enter a number from 1 to 5.")


def calculate_calibration(
    first_point: tuple[int, int],
    second_point: tuple[int, int],
    known_distance_mm: float,
) -> dict:
    first = np.array(first_point, dtype=float)
    second = np.array(second_point, dtype=float)

    pixel_distance = float(
        np.linalg.norm(second - first)
    )

    if pixel_distance <= 0:
        raise ValueError(
            "The two ruler points must be different."
        )

    pixels_per_mm = pixel_distance / known_distance_mm
    mm_per_pixel = known_distance_mm / pixel_distance

    return {
        "first_point": list(first_point),
        "second_point": list(second_point),
        "pixel_distance": pixel_distance,
        "known_distance_mm": known_distance_mm,
        "pixels_per_mm": pixels_per_mm,
        "mm_per_pixel": mm_per_pixel,
    }


def save_calibration(
    calibration: dict,
    photo_path: Path,
) -> None:
    CALIBRATION_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    calibration["photo"] = str(photo_path)

    with CALIBRATION_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            calibration,
            file,
            indent=4,
        )


def main() -> None:
    print("Starting Prozi Photo Calibration...")

    photo_path = choose_photo()

    if photo_path is None:
        print("No photo was selected.")
        return

    image = cv2.imread(str(photo_path))

    if image is None:
        print("The selected photo could not be opened.")
        return

    points = select_ruler_points(image)

    if points is None:
        print("Calibration was canceled.")
        return

    known_distance_mm = request_known_distance()

    try:
        calibration = calculate_calibration(
            points[0],
            points[1],
            known_distance_mm,
        )
    except ValueError as error:
        print(f"Calibration failed: {error}")
        return

    save_calibration(
        calibration,
        photo_path,
    )

    print()
    print("==============================")
    print(" PROZI PHOTO CALIBRATION")
    print("==============================")
    print(f"Pixel distance : {calibration['pixel_distance']:.2f} px")
    print(f"Real distance  : {known_distance_mm:.2f} mm")
    print(f"Pixels per mm  : {calibration['pixels_per_mm']:.4f}")
    print(f"Millimeters/px : {calibration['mm_per_pixel']:.4f}")
    print()
    print(f"Calibration saved to: {CALIBRATION_PATH}")


if __name__ == "__main__":
    main()