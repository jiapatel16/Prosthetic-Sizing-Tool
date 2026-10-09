from pathlib import Path

import cv2
import numpy as np


SIDE_PHOTO_PATH = Path("photos/arm_side.jpg")
TOP_PHOTO_PATH = Path("photos/arm_top.jpg")

FULL_MASK_PATH = Path("results/limb_mask.png")

RESIDUAL_MASK_PATH = Path(
    "results/residual_limb_mask.png"
)

PREVIEW_PATH = Path(
    "results/residual_limb_detected.jpg"
)


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


def choose_view() -> str:
    while True:
        print()
        print("Choose the photo view:")
        print("1 = Side view")
        print("2 = Top view")

        response = input(
            "Enter 1 or 2: "
        ).strip()

        if response == "1":
            return "side"

        if response == "2":
            return "top"

        print("Please enter 1 or 2.")


def get_photo_path(
    view: str,
) -> Path:
    if view == "side":
        return SIDE_PHOTO_PATH

    return TOP_PHOTO_PATH


def load_files(
    photo_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(
        str(photo_path)
    )

    mask = cv2.imread(
        str(FULL_MASK_PATH),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not open {photo_path}"
        )

    if mask is None:
        raise FileNotFoundError(
            "Could not find results/limb_mask.png. "
            "Run segment_limb.py first."
        )

    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(
            "The photo and mask dimensions do not match. "
            "Make sure you segmented the same photo."
        )

    _, binary_mask = cv2.threshold(
        mask,
        127,
        255,
        cv2.THRESH_BINARY,
    )

    return image, binary_mask


def select_vertical_boundary(
    image: np.ndarray,
) -> int | None:
    clicked_x = None

    display_image, scale = resize_for_display(
        image
    )

    window_name = "Select Skin-Sleeve Boundary"

    def mouse_callback(
        event: int,
        x: int,
        y: int,
        flags: int,
        parameter: object,
    ) -> None:
        nonlocal clicked_x

        del y
        del flags
        del parameter

        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_x = x

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL,
    )

    cv2.setMouseCallback(
        window_name,
        mouse_callback,
    )

    print()
    print("SIDE VIEW")
    print("Click where the skin meets the sleeve.")
    print("A vertical boundary line will appear.")
    print("Everything left of the line will be removed.")
    print("Press Enter to confirm.")
    print("Press R to choose again.")
    print("Press Escape to cancel.")

    while True:
        preview = display_image.copy()

        if clicked_x is not None:
            cv2.line(
                preview,
                (clicked_x, 0),
                (
                    clicked_x,
                    preview.shape[0],
                ),
                (0, 255, 0),
                3,
            )

            cv2.putText(
                preview,
                "Skin-sleeve boundary",
                (
                    min(
                        clicked_x + 10,
                        preview.shape[1] - 260,
                    ),
                    35,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        cv2.imshow(
            window_name,
            preview,
        )

        key = cv2.waitKey(20) & 0xFF

        if key in (10, 13):
            if clicked_x is not None:
                break

        elif key in (ord("r"), ord("R")):
            clicked_x = None

        elif key == 27:
            cv2.destroyAllWindows()
            return None

    cv2.destroyAllWindows()

    return int(
        clicked_x / scale
    )


def select_horizontal_boundary(
    image: np.ndarray,
) -> int | None:
    clicked_y = None

    display_image, scale = resize_for_display(
        image
    )

    window_name = "Select Skin-Sleeve Boundary"

    def mouse_callback(
        event: int,
        x: int,
        y: int,
        flags: int,
        parameter: object,
    ) -> None:
        nonlocal clicked_y

        del x
        del flags
        del parameter

        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_y = y

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL,
    )

    cv2.setMouseCallback(
        window_name,
        mouse_callback,
    )

    print()
    print("TOP VIEW")
    print("Click where the skin meets the sleeve.")
    print("A horizontal boundary line will appear.")
    print("Everything below the line will be removed.")
    print("Press Enter to confirm.")
    print("Press R to choose again.")
    print("Press Escape to cancel.")

    while True:
        preview = display_image.copy()

        if clicked_y is not None:
            cv2.line(
                preview,
                (0, clicked_y),
                (
                    preview.shape[1],
                    clicked_y,
                ),
                (0, 255, 0),
                3,
            )

            cv2.putText(
                preview,
                "Skin-sleeve boundary",
                (
                    20,
                    max(
                        35,
                        clicked_y - 15,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        cv2.imshow(
            window_name,
            preview,
        )

        key = cv2.waitKey(20) & 0xFF

        if key in (10, 13):
            if clicked_y is not None:
                break

        elif key in (ord("r"), ord("R")):
            clicked_y = None

        elif key == 27:
            cv2.destroyAllWindows()
            return None

    cv2.destroyAllWindows()

    return int(
        clicked_y / scale
    )


def create_side_view_mask(
    full_mask: np.ndarray,
    boundary_x: int,
) -> np.ndarray:
    residual_mask = full_mask.copy()

    residual_mask[
        :,
        :boundary_x
    ] = 0

    return keep_largest_contour(
        residual_mask
    )


def create_top_view_mask(
    full_mask: np.ndarray,
    boundary_y: int,
) -> np.ndarray:
    residual_mask = full_mask.copy()

    residual_mask[
        boundary_y:,
        :
    ] = 0

    return keep_largest_contour(
        residual_mask
    )


def keep_largest_contour(
    mask: np.ndarray,
) -> np.ndarray:
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        raise ValueError(
            "No residual limb remained after trimming."
        )

    largest_contour = max(
        contours,
        key=cv2.contourArea,
    )

    if cv2.contourArea(
        largest_contour
    ) < 100:
        raise ValueError(
            "The remaining residual-limb area "
            "is too small."
        )

    final_mask = np.zeros_like(
        mask
    )

    cv2.drawContours(
        final_mask,
        [largest_contour],
        -1,
        255,
        cv2.FILLED,
    )

    return final_mask


def create_preview(
    image: np.ndarray,
    residual_mask: np.ndarray,
) -> np.ndarray:
    preview = image.copy()

    contours, _ = cv2.findContours(
        residual_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return preview

    largest_contour = max(
        contours,
        key=cv2.contourArea,
    )

    cv2.drawContours(
        preview,
        [largest_contour],
        -1,
        (0, 255, 0),
        5,
    )

    x, y, width, height = cv2.boundingRect(
        largest_contour
    )

    cv2.rectangle(
        preview,
        (x, y),
        (
            x + width,
            y + height,
        ),
        (255, 0, 0),
        3,
    )

    cv2.putText(
        preview,
        "Residual limb measurement region",
        (
            x,
            max(
                30,
                y - 15,
            ),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
    )

    return preview


def save_results(
    residual_mask: np.ndarray,
    preview: np.ndarray,
) -> None:
    RESIDUAL_MASK_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(RESIDUAL_MASK_PATH),
        residual_mask,
    )

    cv2.imwrite(
        str(PREVIEW_PATH),
        preview,
    )


def main() -> None:
    try:
        view = choose_view()

        photo_path = get_photo_path(
            view
        )

        image, full_mask = load_files(
            photo_path
        )

        if view == "side":
            boundary_x = select_vertical_boundary(
                image
            )

            if boundary_x is None:
                print("Trimming was canceled.")
                return

            residual_mask = create_side_view_mask(
                full_mask,
                boundary_x,
            )

        else:
            boundary_y = select_horizontal_boundary(
                image
            )

            if boundary_y is None:
                print("Trimming was canceled.")
                return

            residual_mask = create_top_view_mask(
                full_mask,
                boundary_y,
            )

        preview = create_preview(
            image,
            residual_mask,
        )

        save_results(
            residual_mask,
            preview,
        )

        print()
        print("Residual-limb region created.")
        print(
            f"View used: {view}"
        )
        print(
            f"Mask saved to: {RESIDUAL_MASK_PATH}"
        )
        print(
            f"Preview saved to: {PREVIEW_PATH}"
        )

        display_preview, _ = resize_for_display(
            preview
        )

        cv2.imshow(
            "Residual Limb Result",
            display_preview,
        )

        print(
            "Press any key in the result window to close."
        )

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except (
        FileNotFoundError,
        ValueError,
    ) as error:
        print(
            f"Trimming failed: {error}"
        )


if __name__ == "__main__":
    main()