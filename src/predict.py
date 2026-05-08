from __future__ import annotations

import argparse
import json
import joblib
import pandas as pd

def main(model_path: str, input_json: str) -> None:
    """
    input_json should be a JSON string like:
    {"MedInc": 8.3, "HouseAge": 41, "AveRooms": 6.98, "AveBedrms": 1.02,
     "Population": 322, "AveOccup": 2.56, "Latitude": 37.88, "Longitude": -122.23}
    """
    pipeline = joblib.load(model_path)
    row = json.loads(input_json)
    X = pd.DataFrame([row])
    pred = pipeline.predict(X)[0]
    print(f"Predicted price (target units): {pred:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_json", type=str, required=True)
    args = parser.parse_args()

    main(args.model_path, args.input_json)