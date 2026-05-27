import argparse
import csv
import json
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def run_info():
    student_path = Path("/app/STUDENT.json")

    with student_path.open("r", encoding="utf-8") as f:
        student_info = json.load(f)

    print(json.dumps(student_info, indent=2))


def find_images(input_dir):
    image_paths = []

    for path in input_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.append(path)

    return sorted(image_paths)


def run_predict():
    input_dir = Path("/data/input")
    output_dir = Path("/data/output")
    output_csv = output_dir / "predictions.csv"

    model_path = Path("/app/models/best.onnx")

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    from detector import CatDetector

    detector = CatDetector(
        onnx_path=str(model_path),
        imgsz=640,
        conf=0.25,
        class_names=("cat",)
    )

    image_paths = find_images(input_dir)

    fieldnames = [
        "image_path",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "confidence",
        "class",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for image_path in image_paths:
            relative_path = image_path.relative_to(input_dir).as_posix()

            detections = detector.predict(str(image_path))

            if len(detections) == 0:
                writer.writerow({
                    "image_path": relative_path,
                    "xmin": "",
                    "ymin": "",
                    "xmax": "",
                    "ymax": "",
                    "confidence": "",
                    "class": "",
                })
                continue

            for det in detections:
                writer.writerow({
                    "image_path": relative_path,
                    "xmin": det["xmin"],
                    "ymin": det["ymin"],
                    "xmax": det["xmax"],
                    "ymax": det["ymax"],
                    "confidence": det["confidence"],
                    "class": det["class"],
                })

    print(f"Predictions written to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Cat detector container CLI")
    parser.add_argument(
        "command",
        choices=["info", "predict"],
        help="Command to run: info or predict"
    )

    args = parser.parse_args()

    if args.command == "info":
        run_info()
    elif args.command == "predict":
        run_predict()


if __name__ == "__main__":
    main()
