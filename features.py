import os
import glob
import numpy as np
import pandas as pd
from skimage import filters
from skimage.measure import regionprops_table
from skimage.morphology import remove_small_objects, remove_small_holes

# Reuse image loading and segmentation from otsu.py
from otsu import (
    load_and_prepare_image,
    compute_hsv_channels,
    compute_purple_mask,
    select_otsu_channel,
    compute_otsu_threshold,
    compute_nucleus_cytoplasm_masks,
)

# Hyperparameters that I had to tune

CROP_PARAMS  = (104, 104, 159) # (crop_x, crop_y, crop_size)
THRESHOLDS   = (0.68, 0.92, 0.08, 0.10, 0.97) # (hue_min, hue_max, sat_min, val_min, val_max)
TRAIN_FOLDER = "./IMA205-challenge/train/"
OTSU_CHANNEL = "saturation" # channel passed to Otsu (value or saturation, but saturation is better)
MIN_OBJECT_SIZE = 50 # px — remove_small_objects / remove_small_holes
REGION_PROPS = ("area", "perimeter", "eccentricity", "solidity", "extent")

# Reuse of otsu for our case

def segment_image(
    img_float: np.ndarray,
    hue_min: float,
    hue_max: float,
    sat_min: float,
    val_min: float,
    val_max: float,
    otsu_channel: str = OTSU_CHANNEL,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    H, S, V     = compute_hsv_channels(img_float)
    purple_mask = compute_purple_mask(H, S, V, hue_min, hue_max, sat_min, val_min, val_max)
    channel     = select_otsu_channel(S, V, otsu_channel)

    purple_pixels = channel[purple_mask]
    if len(purple_pixels) == 0:
        return None, None

    otsu_thresh = compute_otsu_threshold(channel, purple_mask)
    nucleus_mask, cytoplasm_mask = compute_nucleus_cytoplasm_masks(
        purple_mask, channel, otsu_thresh
    )
    return nucleus_mask, cytoplasm_mask

# Mask cleaning (thin post-processing of otsu to remove the holes, needed for some features)

def clean_mask(mask: np.ndarray, min_size: int = MIN_OBJECT_SIZE) -> np.ndarray:
    cleaned = remove_small_objects(mask, max_size=min_size)
    cleaned = remove_small_holes(cleaned, max_size=min_size)
    return cleaned.astype(int)

# Extracting regions

def extract_region_features(
    mask: np.ndarray,
    intensity_image: np.ndarray,
    prefix: str,
    props: tuple = REGION_PROPS,
) -> pd.DataFrame:
    # Runs regionprops_table on *mask* and returns a DataFrame with all column names prefixed by prefix
    table = regionprops_table(
        mask,
        intensity_image=intensity_image,
        properties=props + ("mean_intensity",),
    )
    return pd.DataFrame(table).add_prefix(prefix)


def compute_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    # Appends nucleus-to-cytoplasm ratio features to df and returns it (modifies df in the same time). Expects columns : nuc_area, cyto_area, nuc_mean_intensity, cyto_mean_intensity.
    df["ratio_NC"] = df["nuc_area"] / (df["cyto_area"] + 1e-5)
    df["ratio_intensity"] = df["nuc_mean_intensity"] / (df["cyto_mean_intensity"] + 1e-5)
    return df

# Extraction of features from an image (workflow)

def extract_features_from_image(
    image_path: str,
    crop_params: tuple,
    thresholds: tuple,
    otsu_channel: str = OTSU_CHANNEL,
    min_size: int = MIN_OBJECT_SIZE,
) -> pd.DataFrame:
    # Full per-image pipeline: load -> crop -> segment -> clean -> extract features -> ratios

    crop_x, crop_y, crop_size = crop_params
    hue_min, hue_max, sat_min, val_min, val_max = thresholds

    # load & prepare (reuses otsu.py)
    img_float = load_and_prepare_image(image_path, crop_x, crop_y, crop_size)

    # segmentation (reuses otsu.py)
    nucleus_mask, cytoplasm_mask = segment_image(
        img_float, hue_min, hue_max, sat_min, val_min, val_max, otsu_channel
    )
    if nucleus_mask is None:
        return pd.DataFrame()

    # HSV for intensity image
    _, _, V = compute_hsv_channels(img_float)

    # cleaning
    nuc_clean  = clean_mask(nucleus_mask,   min_size)
    cyto_clean = clean_mask(cytoplasm_mask, min_size)

    # feature extraction
    nuc_feat  = extract_region_features(nuc_clean,  V, prefix="nuc_")
    cyto_feat = extract_region_features(cyto_clean, V, prefix="cyto_")

    df_cell = pd.concat([nuc_feat, cyto_feat], axis=1)

    if not df_cell.empty:
        df_cell = compute_ratio_features(df_cell)
        df_cell["image_id"] = os.path.basename(image_path)

    return df_cell

# Builds dataset

def collect_image_paths(image_folder: str, extension: str = "*.png") -> list[str]:
    # Returns all image paths matching extension inside image_folder
    return glob.glob(os.path.join(image_folder, extension))


def build_feature_dataset(
    image_folder: str,
    crop_params: tuple,
    thresholds: tuple,
    otsu_channel: str = OTSU_CHANNEL,
    log_every: int = 100,
) -> pd.DataFrame:
    # Iterates over every image in image_folder and concatenates per-image feature DataFrames into a single dataset.
    images     = collect_image_paths(image_folder)
    all_frames = []

    print(f"Starting extraction on {len(images)} images...")
    for idx, image_path in enumerate(images):
        if idx > 0 and idx % log_every == 0:
            print(f"  Progress: {idx} / {len(images)}")

        df_cell = extract_features_from_image(
            image_path, crop_params, thresholds, otsu_channel
        )
        if not df_cell.empty:
            all_frames.append(df_cell)

    df_final = pd.concat(all_frames, ignore_index=True)
    print(f"Extraction done! {len(df_final)} rows across {len(all_frames)} images.")
    return df_final


def save_features(df: pd.DataFrame, output_csv: str = "outputs/features.csv") -> None:
    # Saves the feature DataFrame to a CSV file
    df.to_csv(output_csv, index=False)
    print(f"Features saved to {output_csv}")

if __name__ == "__main__":
    dataset_features = build_feature_dataset(TRAIN_FOLDER, CROP_PARAMS, THRESHOLDS)
    save_features(dataset_features, "outputs/features.csv")
