
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_CSV = BASE_DIR / "samples" / "samples.csv"

FEATURES = {
    "DEM": "DEM.tif",
    "Slope": "Slope.tif",
    "RoadDist": "RoadDist.tif",
    "NDVI": "NDVI.tif",
    "PopDensity": "PopDensity.tif",
    "Nightlight": "Nightlight.tif",
    "WaterDist": "WaterDist.tif",
}

MASKS = {
    "positive": "EcologicalSources.tif",
    "negative": "HighResistance.tif",
}

N_SAMPLES = 10000
RANDOM_SEED = 42


def check_consistency(paths):
    reference = None

    for path in paths:
        with rasterio.open(path) as src:
            metadata = (src.shape, src.crs, src.transform)

            if reference is None:
                reference = metadata
            elif metadata != reference:
                raise ValueError(
                    f"Raster alignment mismatch: {path.name}"
                )


def main():
    paths = [
        DATA_DIR / filename
        for filename in FEATURES.values()
    ] + [
        DATA_DIR / filename
        for filename in MASKS.values()
    ]

    check_consistency(paths)

    mask_arrays = {}

    for name, filename in MASKS.items():
        with rasterio.open(DATA_DIR / filename) as src:
            mask_arrays[name] = src.read(1) == 1

    if np.any(
        mask_arrays["positive"] & mask_arrays["negative"]
    ):
        raise ValueError("Positive and negative masks overlap.")

    feature_arrays = []
    valid_features = None

    for filename in FEATURES.values():
        with rasterio.open(DATA_DIR / filename) as src:
            values = src.read(1, masked=True)
            valid = ~np.ma.getmaskarray(values)
            valid &= np.isfinite(values.data)

            feature_arrays.append(
                values.filled(np.nan).astype(float)
            )

            if valid_features is None:
                valid_features = valid
            else:
                valid_features &= valid

    np.random.seed(RANDOM_SEED)
    records = []

    for name, label in [
        ("positive", 0),
        ("negative", 1),
    ]:
        valid_pixels = np.argwhere(
            mask_arrays[name] & valid_features
        )

        if len(valid_pixels) < N_SAMPLES:
            raise ValueError(
                f"Insufficient valid pixels in {name} mask: "
                f"{len(valid_pixels)}"
            )

        selected = valid_pixels[
            np.random.choice(
                len(valid_pixels),
                N_SAMPLES,
                replace=False
            )
        ]

        rows, cols = selected[:, 0], selected[:, 1]

        values = np.column_stack([
            array[rows, cols]
            for array in feature_arrays
        ])

        labels = np.full(
            (N_SAMPLES, 1),
            label,
            dtype=int
        )

        records.append(
            np.column_stack([values, labels])
        )

    columns = list(FEATURES.keys()) + ["Label"]

    samples = pd.DataFrame(
        np.vstack(records),
        columns=columns
    )

    samples["Label"] = samples["Label"].astype(int)

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    samples.to_csv(OUTPUT_CSV, index=False)

    print(f"Samples saved: {OUTPUT_CSV}")
    print(f"Total samples: {len(samples)}")


if __name__ == "__main__":
    main()
