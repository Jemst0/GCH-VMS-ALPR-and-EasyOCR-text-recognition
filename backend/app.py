from __future__ import annotations

import io
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - handled at runtime
    YOLO = None

easyocr = None

try:
    import numpy as np
except Exception:  # pragma: no cover - handled at runtime
    np = None
try:
    import torch
except Exception:  # pragma: no cover - handled at runtime
    torch = None

try:
    import backend.lpr_infer as lpr_infer
except Exception:  # pragma: no cover - handled at runtime
    lpr_infer = None

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "best.pt"

app = FastAPI(title="ALPR Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_model_error: str | None = None
_ocr_reader = None
_ocr_error: str | None = None
_lpr_model = None
_lpr_device = None
_lpr_args = None
_lpr_error: str | None = None


def _is_plate_label(label: str | None) -> bool:
    if not label:
        return False

    normalized = label.lower().replace("_", "-")
    return "plate" in normalized or "license" in normalized or "licence" in normalized


def _prepare_plate_crop(plate_crop: Image.Image) -> Image.Image:
    enlarged_width = max(plate_crop.width * 2, 1)
    enlarged_height = max(plate_crop.height * 2, 1)
    resampling = getattr(Image, "Resampling", Image)
    enlarged = plate_crop.resize((enlarged_width, enlarged_height), resampling.LANCZOS)
    grayscale = ImageOps.grayscale(enlarged)
    return ImageOps.autocontrast(grayscale)


def _read_plate_text(plate_crop: Image.Image) -> tuple[str | None, str | None]:
    """Return (text, source) where source is 'lprnet' or None.

    Uses LPRNet only. Returns (None, None) on failure.
    """
    processed_crop = _prepare_plate_crop(plate_crop)

    try:
        lpr = load_lpr_model()
        if lpr is not None and torch is not None and np is not None:
            model, device, args = lpr
            img_arr = np.array(processed_crop)
            # ensure 3 channels (LPRNet expects RGB-like input)
            if img_arr.ndim == 2:
                img_arr = np.stack([img_arr, img_arr, img_arr], axis=2)
            im = lpr_infer.numpy2tensor(img_arr, args.img_size).unsqueeze(0).to(device)
            with torch.no_grad():
                logit = model(im).detach().cpu()
            pred, _ = lpr_infer.decode(logit, args.chars)
            text = pred[0] if pred else None
            if text:
                return text, "lprnet"
    except Exception:
        return None, None

    return None, None


def load_lpr_model():
    """Load and cache the LPRNet model using `lpr_infer` module.
    Returns tuple (model, device, args) or None on failure.
    """
    global _lpr_model, _lpr_device, _lpr_args, _lpr_error

    if _lpr_model is not None or _lpr_error is not None:
        return (_lpr_model, _lpr_device, _lpr_args) if _lpr_model is not None else None

    if lpr_infer is None:
        _lpr_error = "lpr_infer module not available"
        return None

    try:
        args = lpr_infer.build_args()
        model, device = lpr_infer.load_model(args)
        _lpr_model = model
        _lpr_device = device
        _lpr_args = args
        return (_lpr_model, _lpr_device, _lpr_args)
    except Exception as exc:  # pragma: no cover - runtime guard
        _lpr_error = str(exc)
        return None


def load_model():
    global _model, _model_error

    if _model is not None or _model_error is not None:
        return _model

    if YOLO is None:
        _model_error = "ultralytics is not installed"
        return None

    try:
        _model = YOLO(str(MODEL_PATH))
    except Exception as exc:  # pragma: no cover - runtime guard
        _model_error = str(exc)

    return _model


def load_ocr_reader():
    global _ocr_reader, _ocr_error

    if _ocr_reader is not None or _ocr_error is not None:
        return _ocr_reader

    if easyocr is None:
        _ocr_error = "easyocr is not installed"
        return None

    if np is None:
        _ocr_error = "numpy is not installed"
        return None

    try:
        _ocr_reader = easyocr.Reader(["en"], gpu=False)
    except Exception as exc:  # pragma: no cover - runtime guard
        _ocr_error = str(exc)

    return _ocr_reader


@app.get("/health")
def health():
    model = load_model()
    lpr = load_lpr_model()
    return {
        "ok": True,
        "model_path": str(MODEL_PATH),
        "model_loaded": model is not None,
        "ocr_enabled": False,
        "lpr_loaded": lpr is not None,
        "error": _model_error,
        "ocr_error": None,
        "lpr_error": _lpr_error,
    }


@app.post("/detect")
async def detect(image: UploadFile = File(...)):
    model = load_model()
    if model is None:
        raise HTTPException(status_code=503, detail=_model_error or "model could not be loaded")

    # Using LPRNet for OCR; EasyOCR removed

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="empty image upload")

    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    predictions = model.predict(source=pil_image, verbose=False)

    detections = []
    for result in predictions:
        names = getattr(result, "names", {}) or {}
        for box in getattr(result, "boxes", []):
            xyxy = box.xyxy[0].tolist()
            confidence = float(box.conf[0]) if getattr(box, "conf", None) is not None else 0.0
            class_id = int(box.cls[0]) if getattr(box, "cls", None) is not None else 0
            x1, y1, x2, y2 = xyxy

            left = max(0, int(round(x1)))
            top = max(0, int(round(y1)))
            right = min(pil_image.width, int(round(x2)))
            bottom = min(pil_image.height, int(round(y2)))
            label = names.get(class_id, f"class_{class_id}")

            if right <= left or bottom <= top or not _is_plate_label(label):
                plate_text = None
                ocr_source = None
            else:
                plate_crop = pil_image.crop((left, top, right, bottom))
                # _read_plate_text uses LPRNet only; return tuple (text, source)
                result = _read_plate_text(plate_crop)
                if isinstance(result, tuple):
                    plate_text, ocr_source = result
                else:
                    plate_text = result
                    # if text returned but no source provided, mark as unknown
                    ocr_source = "lprnet" if _lpr_model is not None else None

            detections.append(
                {
                    "label": label,
                    "confidence": confidence,
                    "class_id": class_id,
                    "text": plate_text,
                    "ocr_source": ocr_source,
                    "box": [
                        x1 / pil_image.width,
                        y1 / pil_image.height,
                        (x2 - x1) / pil_image.width,
                        (y2 - y1) / pil_image.height,
                    ],
                }
            )

    detections.sort(key=lambda item: item["confidence"], reverse=True)
    return {
        "image": {"width": pil_image.width, "height": pil_image.height},
        "count": len(detections),
        "plate_text": next((item["text"] for item in detections if item.get("text")), None),
        "detections": detections,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)