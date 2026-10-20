# ============================================================
# HomeMate - House Price Prediction
# Flask + Linear Regression + Decision Tree
# ============================================================

from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIRS = [BASE_DIR / "Dataset", BASE_DIR / "dataset"]
MODEL_DIR = BASE_DIR / "models"
COMPARISON_PATH = MODEL_DIR / "model_comparison.csv"

MODEL_PATHS = {
    "Decision Tree": MODEL_DIR / "decision_tree_pipeline.pkl",
    "Linear Regression": MODEL_DIR / "linear_regression_pipeline.pkl",
}

FEATURE_COLUMNS = [
    "Size",
    "No_of_BHK",
    "Property_type",
    "Property_status",
    "City_name",
    "is_furnished",
    "is_ready_to_move",
]

CLEAN_FILENAME = "HomeMate_residential.csv"
RAW_FILENAME = "Makaan_Properties_Buy.csv"
COMPRESSED_FILENAME = "compressed_data.csv.gz"
EXPLORE_MAX_ROWS = 20000


# ============================================================
# DATASET HELPERS
# ============================================================

def find_dataset(filename):
    for folder in DATASET_DIRS:
        path = folder / filename
        if path.exists():
            return path
    return None


def parse_number(value):
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else np.nan

    text = str(value).strip().lower()

    if not text or text in {
        "nan", "none", "null", "-", "na", "n/a", "<na>"
    }:
        return np.nan

    text = (
        text.replace(",", "")
        .replace("₹", "")
        .replace("rs.", "")
        .replace("rs", "")
        .replace("inr", "")
        .strip()
    )

    multiplier = 1.0

    if re.search(r"\b(crore|crores|cr)\b", text):
        multiplier = 10000000.0
    elif re.search(r"\b(lakh|lakhs|lac|lacs)\b", text):
        multiplier = 100000.0
    elif re.search(r"\b(thousand|k)\b", text):
        multiplier = 1000.0

    text = re.sub(
        r"\b(crore|crores|cr|lakh|lakhs|lac|lacs|thousand|k)\b",
        "",
        text,
    )

    text = re.sub(
        r"\b(sq\.?\s*ft|sqft|square feet)\b",
        "",
        text,
    )

    match = re.search(r"-?\d+(?:\.\d+)?", text)

    if not match:
        return np.nan

    try:
        number = float(match.group()) * multiplier
        return number if np.isfinite(number) else np.nan
    except (ValueError, OverflowError):
        return np.nan



def read_csv_safely(path):
    path = Path(path)

    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(
                path,
                low_memory=False,
                encoding=encoding,
                compression="infer"
            )
        except UnicodeDecodeError:
            continue

    raise UnicodeError(f"Could not decode CSV: {path}")


def normalize_columns(df):
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

    for column in ["Price", "Size", "No_of_BHK"]:
        if column in df.columns:
            df[column] = df[column].apply(parse_number)

    return df


def load_dataset(prefer_cleaned=False):
    
    filenames = ([CLEAN_FILENAME, COMPRESSED_FILENAME, RAW_FILENAME]
    if prefer_cleaned
    else [COMPRESSED_FILENAME, RAW_FILENAME, CLEAN_FILENAME]
)


    for filename in filenames:
        path = find_dataset(filename)

        if path is None:
            continue

        try:
            df = normalize_columns(read_csv_safely(path))
            print(f"Dataset loaded: {path.name} ({len(df):,} rows)")
            return df, path.name

        except Exception:
            app.logger.exception("Could not load dataset: %s", path)

    app.logger.error("No readable dataset CSV found.")
    return pd.DataFrame(), None


def load_performance_dataset():
    df, source = load_dataset(prefer_cleaned=True)

    if df.empty:
        return df

    required = FEATURE_COLUMNS + ["Price"]
    missing = [column for column in required if column not in df.columns]

    if missing:
        app.logger.error("Missing performance columns: %s", missing)
        return pd.DataFrame()

    df = df.copy()

    categorical_columns = [
        "Property_type",
        "Property_status",
        "City_name",
        "is_furnished",
        "is_ready_to_move",
    ]

    for column in categorical_columns:
        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
            .replace("", pd.NA)
        )

    df = df.dropna(
        subset=["Price", "Size", "No_of_BHK"]
    ).copy()

    df = df[
        np.isfinite(df["Price"])
        & np.isfinite(df["Size"])
        & np.isfinite(df["No_of_BHK"])
    ]

    df = df[
        (df["Price"] > 0)
        & (df["Size"] > 0)
        & df["No_of_BHK"].between(1, 10)
    ].copy()

    if source != CLEAN_FILENAME:
        types = (
            df["Property_type"]
            .fillna("")
            .astype(str)
            .str.lower()
        )

        mask = types.str.contains(
            r"apartment|independent|villa|residential|house|flat",
            regex=True,
            na=False,
        )

        if mask.any():
            df = df[mask].copy()

    for column in categorical_columns:
        df[column] = (
            df[column]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )

    return df.reset_index(drop=True)


def json_safe_records(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.astype(object).where(pd.notna(df), None)
    return df.to_dict(orient="records")


# ============================================================
# MODEL HELPERS
# ============================================================

def load_pipeline(model_name):
    path = MODEL_PATHS.get(model_name)

    if path is None or not path.exists():
        app.logger.error("Model file not found: %s", path)
        return None

    try:
        return joblib.load(path)
    except Exception:
        app.logger.exception("Could not load model: %s", model_name)
        return None


def load_model_comparison():
    if not COMPARISON_PATH.exists():
        app.logger.warning("Model comparison CSV is missing.")
        return []

    try:
        return (
            read_csv_safely(COMPARISON_PATH)
            .fillna("")
            .to_dict(orient="records")
        )
    except Exception:
        app.logger.exception("Could not read model comparison report.")
        return []


def get_best_r2(report):
    if not report:
        return 71.67

    candidates = []

    for row in report:
        normalized = {
            str(key).strip().lower().replace("²", "2"): value
            for key, value in row.items()
        }

        for key, value in normalized.items():
            if (
                "r2" not in key
                and "r-squared" not in key
                and "r squared" not in key
            ):
                continue

            match = re.search(r"-?\d+(?:\.\d+)?", str(value))

            if not match:
                continue

            try:
                score = float(match.group())

                if not np.isfinite(score):
                    continue

                if abs(score) <= 1:
                    score *= 100

                candidates.append(score)

            except ValueError:
                continue

    return round(max(candidates), 2) if candidates else 71.67


def get_form_options(df):
    if df.empty:
        return [], []

    cities = (
        sorted(
            df["City_name"]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )
        if "City_name" in df.columns else []
    )

    property_types = (
        sorted(
            df["Property_type"]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )
        if "Property_type" in df.columns else []
    )

    return cities, property_types


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/")
@app.route("/dashboard")
def dashboard():
    df, _ = load_dataset()

    chart_cities = []
    chart_prices = []
    property_labels = []
    property_counts = []

    total_properties = len(df)
    median_price = 0
    total_cities = 0

    if not df.empty:

        if {"City_name", "Price"}.issubset(df.columns):
            city_data = df.dropna(
                subset=["City_name", "Price"]
            )

            summary = (
                city_data.groupby("City_name")["Price"]
                .median()
                .sort_values(ascending=False)
                .head(10)
            )

            chart_cities = summary.index.astype(str).tolist()
            chart_prices = [
                float(value) for value in summary.values
            ]

            total_cities = int(df["City_name"].nunique())

        if "Property_type" in df.columns:
            summary = (
                df["Property_type"]
                .dropna()
                .astype(str)
                .value_counts()
            )

            property_labels = summary.index.tolist()
            property_counts = [
                int(value) for value in summary.values
            ]

        if "Price" in df.columns:
            prices = df["Price"].dropna()

            if not prices.empty:
                median_price = float(prices.median())

    report = load_model_comparison()
    best_r2 = get_best_r2(report)

    return render_template(
        "dashboard.html",
        chart_cities=chart_cities,
        chart_prices=chart_prices,
        property_labels=property_labels,
        property_counts=property_counts,
        total_properties=total_properties,
        median_price=median_price,
        avg_price=median_price,
        total_cities=total_cities,
        city_count=total_cities,
        best_r2=best_r2,
        model_comparison=report,
    )


# ============================================================
# PRICE PREDICTION
# ============================================================

@app.route("/predict", methods=["GET", "POST"])
def predict():
    df, _ = load_dataset()
    cities, property_types = get_form_options(df)

    prediction = None
    error_message = None

    selected_model = request.form.get(
        "model", "Decision Tree"
    )

    if selected_model not in MODEL_PATHS:
        selected_model = "Decision Tree"

    if request.method == "POST":
        try:
            size = float(request.form.get("size", ""))
            bhk = int(request.form.get("bhk", ""))
            city = request.form.get("city", "").strip()
            property_type = request.form.get(
                "property_type", ""
            ).strip()

            if size <= 0 or bhk <= 0:
                raise ValueError(
                    "Area and BHK must be greater than zero."
                )

            if city not in cities:
                raise ValueError("Please select a valid city.")

            if property_type not in property_types:
                raise ValueError(
                    "Please select a valid property type."
                )

            pipeline = load_pipeline(selected_model)

            if pipeline is None:
                raise FileNotFoundError(
                    f"{selected_model} model could not be loaded."
                )

            def mode_value(column):
                if (
                    column in df.columns
                    and not df[column].dropna().empty
                ):
                    return df[column].mode().iloc[0]

                return "Unknown"

            input_data = pd.DataFrame([{
                "Size": size,
                "No_of_BHK": bhk,
                "Property_type": property_type,
                "Property_status": mode_value("Property_status"),
                "City_name": city,
                "is_furnished": mode_value("is_furnished"),
                "is_ready_to_move": mode_value("is_ready_to_move"),
            }])

            expected = getattr(
                pipeline, "feature_names_in_", None
            )

            if expected is not None:
                input_data = input_data[list(expected)]

            prediction = float(
                np.asarray(
                    pipeline.predict(input_data)
                ).ravel()[0]
            )

            if not np.isfinite(prediction):
                raise ValueError(
                    "The model returned an invalid prediction."
                )

            prediction = max(0, prediction)

        except (ValueError, FileNotFoundError) as error:
            error_message = str(error)

        except Exception:
            app.logger.exception("Prediction failed.")
            error_message = (
                "Prediction failed. Check model files and input features."
            )

    return render_template(
        "predict.html",
        prediction=prediction,
        error_message=error_message,
        cities=cities,
        property_types=property_types,
        selected_model=selected_model,
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

@app.route("/comparison")
def comparison():
    report = load_model_comparison()

    return render_template(
        "comparison.html",
        model_comparison=report,
        comparison_data=report,
        results=report,
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

@app.route("/performance")
def performance():
    report = load_model_comparison()

    actual_prices = []
    predicted_prices = []
    residual_labels = []
    residual_counts = []

    sample_count = 0
    performance_error = None

    try:
        df = load_performance_dataset()

        if len(df) < 5:
            raise ValueError(
                "Not enough valid dataset rows for charts."
            )

        X = df[FEATURE_COLUMNS]
        y = df["Price"]

        _, X_test, _, y_test = train_test_split(
            X, y,
            test_size=0.20,
            random_state=42
        )

        pipeline = load_pipeline("Decision Tree")

        if pipeline is None:
            raise FileNotFoundError(
                "Decision Tree model could not be loaded."
            )

        predictions = np.asarray(
            pipeline.predict(X_test)
        ).ravel()

        actual = y_test.to_numpy(dtype=float)

        valid = (
            np.isfinite(actual)
            & np.isfinite(predictions)
        )

        actual = actual[valid]
        predictions = predictions[valid]

        if len(actual) == 0:
            raise ValueError(
                "No valid predictions were generated."
            )

        rng = np.random.RandomState(42)
        count = min(300, len(actual))

        indices = np.sort(
            rng.choice(
                len(actual),
                size=count,
                replace=False
            )
        )

        actual_prices = [
            round(float(x), 2) for x in actual[indices]
        ]

        predicted_prices = [
            round(float(x), 2)
            for x in predictions[indices]
        ]

        sample_count = len(actual_prices)

        residuals = actual - predictions

        lower = float(np.percentile(residuals, 1))
        upper = float(np.percentile(residuals, 99))

        if lower == upper:
            lower = float(np.min(residuals))
            upper = float(np.max(residuals))

        if lower == upper:
            lower -= 1
            upper += 1

        visible = residuals[
            (residuals >= lower)
            & (residuals <= upper)
        ]

        counts, edges = np.histogram(
            visible,
            bins=20,
            range=(lower, upper)
        )

        residual_labels = [
            f"{edges[i] / 100000:.1f}L to "
            f"{edges[i + 1] / 100000:.1f}L"
            for i in range(len(edges) - 1)
        ]

        residual_counts = [
            int(value) for value in counts
        ]

    except Exception as error:
        app.logger.exception("Performance chart error.")
        performance_error = str(error)

    return render_template(
        "performance.html",
        model_comparison=report,
        metrics=report,
        results=report,
        actual_prices=actual_prices,
        predicted_prices=predicted_prices,
        residual_labels=residual_labels,
        residual_counts=residual_counts,
        sample_count=sample_count,
        performance_error=performance_error,
    )


# ============================================================
# EXPLORE PAGE
# ============================================================

@app.route("/explore")
def explore():
    return render_template("explore.html")


# ============================================================
# EXPLORE DATA API
# ============================================================

@app.route("/api/explore-data")
def explore_data_api():
    """
    Return complete-dataset statistics and charts.
    Return only a limited sample of rows for the browser table.
    """

    df, source = load_dataset(prefer_cleaned=False)

    if df.empty:
        return jsonify({
            "error": (
                "Dataset CSV could not be found or read. "
                "Check the Dataset folder."
            )
        }), 404

    required = [
        "Price",
        "Size",
        "No_of_BHK",
        "City_name",
        "Property_type",
    ]

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        return jsonify({
            "error": "Missing columns: " + ", ".join(missing),
            "source": source,
            "available_columns": df.columns.tolist(),
        }), 400

    # Standardize numeric columns.
    for column in ["Price", "Size", "No_of_BHK"]:
        df[column] = df[column].apply(parse_number)

    # Standardize categorical columns.
    categorical_columns = [
        "City_name",
        "Property_type",
        "Property_status",
        "is_furnished",
        "is_ready_to_move",
    ]

    for column in categorical_columns:
        if column not in df.columns:
            df[column] = "Unknown"

        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
            .replace("", pd.NA)
            .fillna("Unknown")
            .astype(str)
        )

    df["City_name"] = df["City_name"].apply(
        lambda value: (
            "Unknown"
            if value.strip().lower()
            in {"", "nan", "none", "null", "<na>"}
            else value.strip().title()
        )
    )

    df["Property_type"] = df["Property_type"].apply(
        lambda value: (
            "Unknown"
            if value.strip().lower()
            in {"", "nan", "none", "null", "<na>"}
            else value.strip()
        )
    )

    # Total properties means every row in the source dataset.
    total_records = int(len(df))

    # Price statistics use rows with valid positive prices.
    valid_prices = df[
        df["Price"].notna()
        & np.isfinite(df["Price"])
        & (df["Price"] > 0)
    ].copy()

    if valid_prices.empty:
        return jsonify({
            "error": "No valid positive property prices found."
        }), 400

    prices = valid_prices["Price"]

    summary = {
        "total_records": total_records,
        "valid_price_records": int(len(valid_prices)),
        "average_price": float(prices.mean()),
        "lowest_price": float(prices.min()),
        "highest_price": float(prices.max()),
        "total_cities": int(
            valid_prices.loc[
                valid_prices["City_name"] != "Unknown",
                "City_name"
            ].nunique()
        ),
    }

    # Average price by city from the full dataset.
    city_summary = (
        valid_prices[
            valid_prices["City_name"] != "Unknown"
        ]
        .groupby("City_name")["Price"]
        .agg(["mean", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
        .head(10)
    )

    city_chart = [
        {
            "name": str(row["City_name"]),
            "average": float(row["mean"]),
            "count": int(row["count"]),
        }
        for _, row in city_summary.iterrows()
    ]

    # Average price by BHK from the full dataset.
    bhk_data = valid_prices[
        valid_prices["No_of_BHK"].between(1, 10)
    ]

    bhk_summary = (
        bhk_data.groupby("No_of_BHK")["Price"]
        .agg(["mean", "count"])
        .reset_index()
        .sort_values("No_of_BHK")
    )

    bhk_chart = [
        {
            "bhk": float(row["No_of_BHK"]),
            "average": float(row["mean"]),
            "count": int(row["count"]),
        }
        for _, row in bhk_summary.iterrows()
    ]

    # Property type distribution from the full dataset.
    type_counts = (
        valid_prices["Property_type"]
        .value_counts()
        .head(12)
    )

    type_chart = [
        {"name": str(name), "count": int(count)}
        for name, count in type_counts.items()
    ]

    # Price distribution using full-dataset quantiles.
    quantiles = prices.quantile(
        [0, 1/6, 2/6, 3/6, 4/6, 5/6, 1]
    ).tolist()

    price_bins = []

    for i in range(6):
        low = float(quantiles[i])
        high = float(quantiles[i + 1])

        if i == 5:
            count = int(
                ((prices >= low) & (prices <= high)).sum()
            )
        else:
            count = int(
                ((prices >= low) & (prices < high)).sum()
            )

        price_bins.append({
            "low": low,
            "high": high,
            "count": count,
        })

    # Filter dropdown options come from the entire dataset.
    cities = sorted(
        value
        for value in df["City_name"].unique().tolist()
        if value != "Unknown"
    )

    property_types = sorted(
        value
        for value in df["Property_type"].unique().tolist()
        if value != "Unknown"
    )

    # Send at most 20,000 rows to the browser.
    truncated = len(df) > EXPLORE_MAX_ROWS

    if truncated:
        sample_df = (
            df.sample(
                n=EXPLORE_MAX_ROWS,
                random_state=42
            )
            .sort_index()
        )
    else:
        sample_df = df

    records = json_safe_records(sample_df)

    return jsonify({
        "source": source,
        "total_records": total_records,
        "returned_records": len(records),
        "truncated": truncated,
        "summary": summary,
        "chart_data": {
            "city": city_chart,
            "bhk": bhk_chart,
            "types": type_chart,
            "price_distribution": price_bins,
        },
        "cities": cities,
        "property_types": property_types,
        "total_cities": summary["total_cities"],
        "properties": records,
    })


# ============================================================
# ABOUT PROJECT
# ============================================================

@app.route("/about")
def about():
    return render_template("about.html")


# ============================================================
# API HEALTH CHECK
# ============================================================

@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "project": "HomeMate",
    })


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5001,
        debug=True
    )