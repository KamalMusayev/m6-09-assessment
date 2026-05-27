import numpy as np
import onnxruntime as ort
from PIL import Image


class CatDetector:
    def __init__(self, onnx_path, imgsz=640, conf=0.25, class_names=("cat",)):
        self.onnx_path = onnx_path
        self.imgsz = imgsz
        self.conf = conf
        self.class_names = class_names

        self.session = ort.InferenceSession(
            self.onnx_path,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_shape = self.session.get_outputs()[0].shape

    def _letterbox(self, image):
        orig_w, orig_h = image.size

        scale = min(self.imgsz / orig_w, self.imgsz / orig_h)

        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))

        resized = image.resize((new_w, new_h), Image.BILINEAR)

        padded = Image.new("RGB", (self.imgsz, self.imgsz), (114, 114, 114))

        pad_x = (self.imgsz - new_w) // 2
        pad_y = (self.imgsz - new_h) // 2

        padded.paste(resized, (pad_x, pad_y))

        return padded, scale, pad_x, pad_y

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size

        padded, scale, pad_x, pad_y = self._letterbox(image)

        x = np.asarray(padded, dtype=np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, axis=0)

        outputs = self.session.run(None, {self.input_name: x})
        detections = outputs[0]

        if detections.ndim == 3:
            detections = detections[0]

        results = []

        for det in detections:
            x1, y1, x2, y2, score, cls_id = det[:6]

            score = float(score)

            if score < self.conf:
                continue

            x1 = (float(x1) - pad_x) / scale
            y1 = (float(y1) - pad_y) / scale
            x2 = (float(x2) - pad_x) / scale
            y2 = (float(y2) - pad_y) / scale

            x1 = max(0.0, min(float(orig_w), x1))
            y1 = max(0.0, min(float(orig_h), y1))
            x2 = max(0.0, min(float(orig_w), x2))
            y2 = max(0.0, min(float(orig_h), y2))

            cls_id = int(cls_id)
            class_name = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)

            results.append({
                "xmin": float(x1),
                "ymin": float(y1),
                "xmax": float(x2),
                "ymax": float(y2),
                "confidence": score,
                "class": class_name,
            })

        return results
