from pathlib import Path
from tkinter import Tk, filedialog

import cv2
import numpy as np


RESULTS_FOLDER = Path("results")
MASK_PATH = RESULTS_FOLDER / "limb_mask.png"
OUTLINE_PATH = RESULTS_FOLDER / "detected_limb.jpg"
SEGMENTED_PATH = RESULTS_FOLDER / "segmented_limb.png"

ROI_WINDOW = "Select Limb Region"
CORRECTION_WINDOW = "Correct Segmentation"


def choose_photo() -> Path | None:
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    selected_file = filedialog.askopenfilename(
        title="Choose a residual-limb photo",
        initialdir="photos",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
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


def select_limb_region(
    image: np.ndarray,
) -> tuple[int, int, int, int] | None:
    display_image, scale = resize_for_display(image)

    print()
    print("Draw a box around the complete arm and residual limb.")
    print("Include a small amount of background.")
    print("Leave the ruler outside the box if possible.")
    print("Press Enter or Space when finished.")
    print("Press C to cancel.")

    selected_box = cv2.selectROI(
        ROI_WINDOW,
        display_image,
        showCrosshair=True,
        fromCenter=False,
    )

    cv2.destroyAllWindows()

    x, y, width, height = selected_box

    if width == 0 or height == 0:
        return None

    return (
        int(x / scale),
        int(y / scale),
        int(width / scale),
        int(height / scale),
    )


def create_grabcut_mask(
    image: np.ndarray,
    region: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.zeros(
        image.shape[:2],
        dtype=np.uint8,
    )

    background_model = np.zeros(
        (1, 65),
        dtype=np.float64,
    )

    foreground_model = np.zeros(
        (1, 65),
        dtype=np.float64,
    )

    cv2.grabCut(
        image,
        mask,
        region,
        background_model,
        foreground_model,
        8,
        cv2.GC_INIT_WITH_RECT,
    )

    return mask, background_model, foreground_model


def convert_grabcut_mask(
    grabcut_mask: np.ndarray,
) -> np.ndarray:
    return np.where(
        (grabcut_mask == cv2.GC_FGD)
        | (grabcut_mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)


def refine_mask_interactively(
    image: np.ndarray,
    grabcut_mask: np.ndarray,
    background_model: np.ndarray,
    foreground_model: np.ndarray,
) -> np.ndarray:
    display_image, scale = resize_for_display(image)

    display_mask = cv2.resize(
        grabcut_mask,
        (
            display_image.shape[1],
            display_image.shape[0],
        ),
        interpolation=cv2.INTER_NEAREST,
    )

    drawing_mode = "foreground"
    brush_radius = 10
    is_drawing = False

    def draw_on_mask(
        event: int,
        x: int,
        y: int,
        flags: int,
        parameter: object,
    ) -> None:
        nonlocal is_drawing

        del flags
        del parameter

        if event == cv2.EVENT_LBUTTONDOWN:
            is_drawing = True

        elif event == cv2.EVENT_LBUTTONUP:
            is_drawing = False

        elif event == cv2.EVENT_MOUSEMOVE and is_drawing:
            value = (
                cv2.GC_FGD
                if drawing_mode == "foreground"
                else cv2.GC_BGD
            )

            cv2.circle(
                display_mask,
                (x, y),
                brush_radius,
                value,
                -1,
            )

    cv2.namedWindow(
        CORRECTION_WINDOW,
        cv2.WINDOW_NORMAL,
    )

    cv2.setMouseCallback(
        CORRECTION_WINDOW,
        draw_on_mask,
    )

    print()
    print("Segmentation correction controls:")
    print("Press F to paint definite limb.")
    print("Press B to paint definite background.")
    print("Drag with the left mouse button to paint.")
    print("Press Enter to rerun segmentation.")
    print("Press S to save and continue.")
    print("Press Escape to cancel.")

    while True:
        binary_display_mask = np.where(
            (display_mask == cv2.GC_FGD)
            | (display_mask == cv2.GC_PR_FGD),
            255,
            0,
        ).astype(np.uint8)

        overlay = display_image.copy()

        foreground_overlay = np.zeros_like(display_image)
        foreground_overlay[:, :] = (0, 255, 0)

        background_overlay = np.zeros_like(display_image)
        background_overlay[:, :] = (0, 0, 255)

        foreground_pixels = (
            (display_mask == cv2.GC_FGD)
            | (display_mask == cv2.GC_PR_FGD)
        )

        definite_background_pixels = (
            display_mask == cv2.GC_BGD
        )

        overlay[foreground_pixels] = cv2.addWeighted(
            display_image[foreground_pixels],
            0.55,
            foreground_overlay[foreground_pixels],
            0.45,
            0,
        )

        overlay[definite_background_pixels] = cv2.addWeighted(
            display_image[definite_background_pixels],
            0.75,
            background_overlay[definite_background_pixels],
            0.25,
            0,
        )

        mode_text = (
            "Mode: LIMB"
            if drawing_mode == "foreground"
            else "Mode: BACKGROUND"
        )

        cv2.putText(
            overlay,
            mode_text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            overlay,
            "F = limb | B = background | Enter = refine | S = save",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        cv2.imshow(
            CORRECTION_WINDOW,
            overlay,
        )

        key = cv2.waitKey(20) & 0xFF

        if key in (ord("f"), ord("F")):
            drawing_mode = "foreground"

        elif key in (ord("b"), ord("B")):
            drawing_mode = "background"

        elif key in (10, 13):
            full_size_mask = cv2.resize(
                display_mask,
                (
                    image.shape[1],
                    image.shape[0],
                ),
                interpolation=cv2.INTER_NEAREST,
            )

            cv2.grabCut(
                image,
                full_size_mask,
                None,
                background_model,
                foreground_model,
                5,
                cv2.GC_INIT_WITH_MASK,
            )

            display_mask = cv2.resize(
                full_size_mask,
                (
                    display_image.shape[1],
                    display_image.shape[0],
                ),
                interpolation=cv2.INTER_NEAREST,
            )

        elif key in (ord("s"), ord("S")):
            break

        elif key == 27:
            cv2.destroyAllWindows()
            raise RuntimeError(
                "Segmentation correction was canceled."
            )

    cv2.destroyAllWindows()

    corrected_full_size_mask = cv2.resize(
        display_mask,
        (
            image.shape[1],
            image.shape[0],
        ),
        interpolation=cv2.INTER_NEAREST,
    )

    return convert_grabcut_mask(
        corrected_full_size_mask
    )


def clean_mask(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7),
    )

    cleaned = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )

    return cleaned


def keep_largest_contour(
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        raise ValueError(
            "No foreground object was detected."
        )

    largest_contour = max(
        contours,
        key=cv2.contourArea,
    )

    if cv2.contourArea(largest_contour) < 1000:
        raise ValueError(
            "The detected limb area is too small."
        )

    final_mask = np.zeros_like(mask)

    cv2.drawContours(
        final_mask,
        [largest_contour],
        -1,
        255,
        thickness=cv2.FILLED,
    )

    return final_mask, largest_contour


def create_outline_image(
    image: np.ndarray,
    contour: np.ndarray,
) -> np.ndarray:
    outlined = image.copy()

    cv2.drawContours(
        outlined,
        [contour],
        -1,
        (0, 255, 0),
        thickness=5,
    )

    x, y, width, height = cv2.boundingRect(contour)

    cv2.rectangle(
        outlined,
        (x, y),
        (x + width, y + height),
        (255, 0, 0),
        thickness=3,
    )

    cv2.putText(
        outlined,
        "Detected limb",
        (x, max(30, y - 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    return outlined


def create_segmented_image(
    image: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    segmented = cv2.bitwise_and(
        image,
        image,
        mask=mask,
    )

    blue, green, red = cv2.split(segmented)

    return cv2.merge(
        [blue, green, red, mask]
    )


def save_results(
    mask: np.ndarray,
    outline_image: np.ndarray,
    segmented_image: np.ndarray,
) -> None:
    RESULTS_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(MASK_PATH),
        mask,
    )

    cv2.imwrite(
        str(OUTLINE_PATH),
        outline_image,
    )

    cv2.imwrite(
        str(SEGMENTED_PATH),
        segmented_image,
    )


def show_result(
    outline_image: np.ndarray,
) -> None:
    display_image, _ = resize_for_display(
        outline_image
    )

    cv2.imshow(
        "Prozi Detection Result",
        display_image,
    )

    print()
    print("Press any key in the result window to close it.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> None:
    print("Starting Prozi Limb Segmentation...")

    photo_path = choose_photo()

    if photo_path is None:
        print("No photo was selected.")
        return

    print(f"Selected photo: {photo_path.name}")

    image = cv2.imread(
        str(photo_path)
    )

    if image is None:
        print("The selected photo could not be opened.")
        return

    region = select_limb_region(image)

    if region is None:
        print("No limb region was selected.")
        return

    try:
        (
            grabcut_mask,
            background_model,
            foreground_model,
        ) = create_grabcut_mask(
            image,
            region,
        )

        corrected_mask = refine_mask_interactively(
            image,
            grabcut_mask,
            background_model,
            foreground_model,
        )

        cleaned_mask = clean_mask(
            corrected_mask
        )

        (
            final_mask,
            largest_contour,
        ) = keep_largest_contour(
            cleaned_mask
        )

    except (
        ValueError,
        RuntimeError,
    ) as error:
        print(f"Segmentation failed: {error}")
        return

    outline_image = create_outline_image(
        image,
        largest_contour,
    )

    segmented_image = create_segmented_image(
        image,
        final_mask,
    )

    save_results(
        final_mask,
        outline_image,
        segmented_image,
    )

    print()
    print("Segmentation completed.")
    print(f"Mask saved to: {MASK_PATH}")
    print(f"Outline saved to: {OUTLINE_PATH}")
    print(f"Segmented image saved to: {SEGMENTED_PATH}")

    show_result(
        outline_image
    )


if __name__ == "__main__":
    main()