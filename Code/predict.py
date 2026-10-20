
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "results" / "xgb_model.pkl"
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_PATH = OUTPUT_DIR / "ResistanceProbability.tif"

FEATURES = {
    "DEM": "DEM.tif",
    "Slope": "Slope.tif",
    "RoadDist": "RoadDist.tif",
    "NDVI": "NDVI.tif",
    "PopDensity": "PopDensity.tif",
    "Nightlight": "Nightlight.tif",
    "WaterDist": "WaterDist.tif"
}

NODATA = -9999.0
BLOCK_SIZE = 512


def load_model():
    model_data = joblib.load(MODEL_PATH)

    model = model_data["model"]
    scaler = model_data["scaler"]
    feature_cols = list(model_data["feature_cols"])

    if feature_cols != list(FEATURES.keys()):
        raise ValueError("Feature order does not match training.")

    if scaler.n_features_in_ != len(FEATURES):
        raise ValueError("Scaler feature count mismatch.")

    return model, scaler, feature_cols


def check_raster_alignment(datasets):
    reference = datasets[0]

    for src in datasets[1:]:
        if (
            src.shape != reference.shape
            or src.crs != reference.crs
            or not src.transform.almost_equals(
                reference.transform
            )
        ):
            raise ValueError(
                f"Raster alignment mismatch: {src.name}"
            )

    return reference.profile.copy()


def predict_window(
    datasets, window, model, scaler, feature_cols
):
    arrays = []
    valid_mask = None

    for src in datasets:
        data = src.read(
            1, window=window, masked=True
        )

        values = np.asarray(
            data.filled(np.nan), dtype=np.float64
        )

        valid = np.isfinite(values)

        if valid_mask is None:
            valid_mask = valid
        else:
            valid_mask &= valid

        arrays.append(values)

    height, width = arrays[0].shape

    output = np.full(
        (height, width),
        NODATA,
        dtype=np.float32
    )

    if not np.any(valid_mask):
        return output

    X = np.column_stack([
        array[valid_mask]
        for array in arrays
    ])

    X = pd.DataFrame(
        X,
        columns=feature_cols
    )

    X_scaled = pd.DataFrame(
        scaler.transform(X),
        columns=feature_cols
    )

    probabilities = model.predict_proba(
        X_scaled
    )[:, 1]

    output[valid_mask] = probabilities.astype(
        np.float32
    )

    return output


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model, scaler, feature_cols = load_model()

    paths = [
        DATA_DIR / filename
        for filename in FEATURES.values()
    ]

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    from contextlib import ExitStack

    with ExitStack() as stack:
        datasets = [
            stack.enter_context(rasterio.open(path))
            for path in paths
        ]

        profile = check_raster_alignment(datasets)

        profile.update(
            driver="GTiff",
            dtype="float32",
            count=1,
            nodata=NODATA,
            compress="deflate"
        )

        height = datasets[0].height
        width = datasets[0].width

        with rasterio.open(
            OUTPUT_PATH, "w", **profile
        ) as dst:

            for row in range(0, height, BLOCK_SIZE):
                for col in range(0, width, BLOCK_SIZE):

                    window = Window(
                        col,
                        row,
                        min(BLOCK_SIZE, width - col),
                        min(BLOCK_SIZE, height - row)
                    )

                    prediction = predict_window(
                        datasets,
                        window,
                        model,
                        scaler,
                        feature_cols
                    )

                    dst.write(
                        prediction,
                        1,
                        window=window
                    )

    print(f"Resistance probability saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
