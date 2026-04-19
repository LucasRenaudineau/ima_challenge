import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import color, filters
from skimage.io import imread

# This part is for the tunable parameters for all tbis file

IMAGE_PATH  = "./IMA205-challenge/train/train_00004.png"
CROP_X      = 104
CROP_Y      = 104
CROP_SIZE   = 159

HUE_MIN     = 0.68
HUE_MAX     = 0.92
SAT_MIN     = 0.08
VAL_MIN     = 0.10
VAL_MAX     = 0.97

OTSU_CHANNEL = "saturation"

OUTPUT_PATH  = "./outputs/purple_segmentation.png"

# Image loading and pre-processing

def load_image(image_path: str) -> np.ndarray:
    img = imread(image_path)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return img

def crop_image(img: np.ndarray, crop_x: int, crop_y: int, crop_size: int) -> np.ndarray:
    return img[crop_y : crop_y + crop_size, crop_x : crop_x + crop_size].copy()

def normalize_to_float(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return img.astype(np.float32)

def load_and_prepare_image(
    image_path: str, crop_x: int, crop_y: int, crop_size: int
) -> np.ndarray:
    img_raw   = load_image(image_path)
    img_crop  = crop_image(img_raw, crop_x, crop_y, crop_size)
    img_float = normalize_to_float(img_crop)
    return img_float

# This converts a float image into its H, S, V version (all between 0 and 1).

def compute_hsv_channels(img_float: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    img_hsv = color.rgb2hsv(img_float)
    H = img_hsv[:, :, 0]
    S = img_hsv[:, :, 1]
    V = img_hsv[:, :, 2]
    return H, S, V

# Purple mask

def compute_purple_mask(
    H: np.ndarray,
    S: np.ndarray,
    V: np.ndarray,
    hue_min: float = HUE_MIN,
    hue_max: float = HUE_MAX,
    sat_min: float = SAT_MIN,
    val_min: float = VAL_MIN,
    val_max: float = VAL_MAX,
) -> np.ndarray:
    """Boolean mask that selects purple pixels by hue, saturation and value gates."""
    return (
        (H >= hue_min) & (H <= hue_max)
        & (S >= sat_min)
        & (V >= val_min) & (V <= val_max)
    )

# The next functions implement the OTSU method for segmentation

def select_otsu_channel(
    S: np.ndarray,
    V: np.ndarray,
    otsu_channel: str,
) -> np.ndarray:
    return S # I also tried returning V instead of S but the result is better with S

def compute_otsu_threshold(channel: np.ndarray, mask: np.ndarray) -> float:
    """Computes the Otsu threshold from the masked pixels of *channel*."""
    pixel_values = channel[mask]
    if len(pixel_values) == 0:
        raise RuntimeError(
            "No purple pixels found."
        )
    return float(filters.threshold_otsu(pixel_values))

def compute_nucleus_cytoplasm_masks(
    purple_mask: np.ndarray,
    channel: np.ndarray,
    otsu_thresh: float,
) -> tuple[np.ndarray, np.ndarray]: # this splits the mask into the 2 masks depending of if the pixel saturation is greater or lower than the threshold
    nucleus_mask   = purple_mask & (channel <  otsu_thresh)
    cytoplasm_mask = purple_mask & (channel >= otsu_thresh)
    return nucleus_mask, cytoplasm_mask

def segment_purple_cells(
    img_float: np.ndarray,
    hue_min: float = HUE_MIN,
    hue_max: float = HUE_MAX,
    sat_min: float = SAT_MIN,
    val_min: float = VAL_MIN,
    val_max: float = VAL_MAX,
    otsu_channel: str = OTSU_CHANNEL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    # Full segmentation pipeline on a prepared float image.
    # returns : H, S, V and nucleus_mask and cytoplasm_mask and otsu_thresh
    H, S, V       = compute_hsv_channels(img_float)
    purple_mask   = compute_purple_mask(H, S, V, hue_min, hue_max, sat_min, val_min, val_max)
    channel       = select_otsu_channel(S, V, otsu_channel)
    otsu_thresh   = compute_otsu_threshold(channel, purple_mask)
    nucleus_mask, cytoplasm_mask = compute_nucleus_cytoplasm_masks(
        purple_mask, channel, otsu_thresh
    )
    return H, S, V, nucleus_mask, cytoplasm_mask, otsu_thresh

# This part is only for visualisation, tests, and tuning by hand the hyperparameters

def build_overlay(
    img_float: np.ndarray,
    nucleus_mask: np.ndarray,
    cytoplasm_mask: np.ndarray,
) -> np.ndarray:
    overlay = img_float.copy()
    overlay[nucleus_mask]   = [0.40, 0.00, 0.60]  # dark purple
    overlay[cytoplasm_mask] = [0.80, 0.50, 1.00]  # light purple
    return overlay


def build_figure(
    img_float: np.ndarray,
    H: np.ndarray,
    S: np.ndarray,
    V: np.ndarray,
    nucleus_mask: np.ndarray,
    cytoplasm_mask: np.ndarray,
    otsu_thresh: float,
    crop_size: int,
) -> plt.Figure:
    purple_mask = nucleus_mask | cytoplasm_mask   # re-derive for display
    hue_display = plt.cm.hsv(H)[:, :, :3]
    overlay = build_overlay(img_float, nucleus_mask, cytoplasm_mask)

    nucleus_coords = list(zip(*np.where(nucleus_mask)))
    cytoplasm_coords = list(zip(*np.where(cytoplasm_mask)))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        "Purple Cell Segmentation  |  Hover over any panel to see Hue value",
        fontsize=13,
    )

    ax_orig, ax_hue, ax_purple = axes[0]
    ax_nucleus, ax_cyto, ax_overlay = axes[1]

    ax_orig.imshow(img_float); ax_orig.set_title("Original (cropped)")
    ax_hue.imshow(hue_display); ax_hue.set_title("Hue map (HSV colormap)")
    ax_purple.imshow(purple_mask, cmap="gray");
    ax_purple.set_title("Purple mask (hue+sat+val gate)")
    ax_nucleus.imshow(nucleus_mask, cmap="Purples")
    ax_nucleus.set_title(f"Nucleus (channel < {otsu_thresh:.3f})")
    ax_cyto.imshow(cytoplasm_mask, cmap="RdPu")
    ax_cyto.set_title(f"Cytoplasm (channel >= {otsu_thresh:.3f})")
    ax_overlay.imshow(overlay); ax_overlay.set_title("Overlay: dark=nucleus, light=cytoplasm")

    patch_n = mpatches.Patch(color=(0.40, 0.00, 0.60), label=f"Nucleus ({len(nucleus_coords)} px)")
    patch_c = mpatches.Patch(color=(0.80, 0.50, 1.00), label=f"Cytoplasm ({len(cytoplasm_coords)} px)")
    ax_overlay.legend(handles=[patch_n, patch_c], loc="lower right", fontsize=8)

    for ax in axes.flat:
        ax.axis("off")

    _attach_hover_callback(fig, axes, H, S, V, nucleus_mask, cytoplasm_mask, crop_size)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


def _attach_hover_callback(
    fig: plt.Figure,
    axes: np.ndarray,
    H: np.ndarray,
    S: np.ndarray,
    V: np.ndarray,
    nucleus_mask: np.ndarray,
    cytoplasm_mask: np.ndarray,
    crop_size: int,
) -> None:
    hue_text = fig.text(
        0.5, 0.01,
        "Move mouse over image -> Hue value will appear here",
        ha="center", fontsize=11, color="navy",
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray"),
    )

    def on_mouse_move(event):
        for ax in axes.flat:
            if event.inaxes == ax:
                x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
                if 0 <= y < crop_size and 0 <= x < crop_size:
                    region = (
                        "NUCLEUS" if nucleus_mask[y, x]
                        else ("CYTOPLASM" if cytoplasm_mask[y, x] else "background")
                    )
                    hue_text.set_text(
                        f"Pixel ({x}, {y}) | Hue={H[y,x]:.3f}  "
                        f"Sat={S[y,x]:.3f} Val={V[y,x]:.3f} -> {region}"
                    )
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)


def save_figure(fig: plt.Figure, output_path: str) -> None:
    # Saves fig to disk, creating parent directories as needed
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {output_path}")

# Only starts when this file is chosen

if __name__ == "__main__":
    img_float = load_and_prepare_image(IMAGE_PATH, CROP_X, CROP_Y, CROP_SIZE)

    H, S, V, nucleus_mask, cytoplasm_mask, otsu_thresh = segment_purple_cells(
        img_float,
        hue_min=HUE_MIN, hue_max=HUE_MAX,
        sat_min=SAT_MIN, val_min=VAL_MIN, val_max=VAL_MAX,
        otsu_channel=OTSU_CHANNEL,
    )

    print(f"Otsu threshold on '{OTSU_CHANNEL}': {otsu_thresh:.4f}")
    print(f"Nucleus pixels : {nucleus_mask.sum()}")
    print(f"Cytoplasm pixels : {cytoplasm_mask.sum()}")

    nucleus_coords   = list(zip(*np.where(nucleus_mask)))
    cytoplasm_coords = list(zip(*np.where(cytoplasm_mask)))

    fig = build_figure(
        img_float, H, S, V,
        nucleus_mask, cytoplasm_mask,
        otsu_thresh, CROP_SIZE,
    )

    save_figure(fig, OUTPUT_PATH)
    plt.show()

    print("\n=== Results ===")
    print(f"nucleus_coords[0:5] = {nucleus_coords[:5]}")
    print(f"cytoplasm_coords[0:5] = {cytoplasm_coords[:5]}")
