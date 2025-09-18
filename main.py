"""
FastAPI app for Ultralyics YOLO prediction
File: main.py

Features:
- /models/load : load a model into cache
- /models : list loaded models
- /predict/image : POST image -> returns PNG with bounding boxes
- /predict/json : POST image -> returns JSON with boxes + image_base64
- /analyze : POST image -> returns counts for kelapa_muda / kelapa_tua and confidence (for your React frontend)

Usage:
1) Install dependencies:
   pip install fastapi uvicorn[standard] ultralytics pillow numpy python-multipart aiofiles

2) Run server:
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Notes:
- This code assumes your YOLO model's class names include 'kelapa_muda' and 'kelapa_tua'.
- You can specify model by name/path with the `model` form field. If not provided, DEFAULT_MODEL is used.
- CORS is enabled for common dev origins; adjust in production.

"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from typing import Optional, List, Dict
from PIL import Image, ImageDraw, ImageFont
import io
import numpy as np
import base64
import threading

app = FastAPI(title="Ultralytics YOLO - FastAPI")

# CORS - adjust origins as needed
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple model cache + lock
_models_cache: Dict[str, YOLO] = {}
_models_lock = threading.Lock()
DEFAULT_MODEL = "best.pt"  # change to your default model path/id


def load_model(model_name: str = DEFAULT_MODEL) -> YOLO:
    """Load model into cache and return it. Thread-safe."""
    with _models_lock:
        if model_name in _models_cache:
            return _models_cache[model_name]
        try:
            model = YOLO(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{model_name}': {e}")
        _models_cache[model_name] = model
        return model


def _pil_from_numpy(arr: np.ndarray) -> Image.Image:
    """Convert numpy RGB array to PIL Image."""
    if arr.dtype != np.uint8:
        arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr)


def draw_boxes_on_pil(image_pil: Image.Image, boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, names: Dict[int, str]):
    """Draw bounding boxes on PIL image in-place.

    boxes: (N,4) array in xyxy format
    scores: (N,) array
    classes: (N,) array
    names: dict mapping class id to name
    """
    draw = ImageDraw.Draw(image_pil)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=14)
    except Exception:
        font = ImageFont.load_default()

    for (x1, y1, x2, y2), score, cls in zip(boxes, scores, classes):
        label_name = names.get(int(cls), str(int(cls)))
        label = f"{label_name} {float(score):.2f}"
        # box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        # label background
        text_w, text_h = draw.textsize(label, font=font)
        text_bg = [x1, max(0, y1 - text_h - 6), x1 + text_w + 6, y1]
        draw.rectangle(text_bg, fill="red")
        draw.text((x1 + 3, y1 - text_h - 3), label, fill="white", font=font)


@app.post("/models/load")
async def api_load_model(model: str = Form(...)):
    """Load a model into cache (provide model path or hub id)."""
    try:
        load_model(model)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "loaded", "model": model}


@app.get("/models")
async def api_models():
    return {"loaded_models": list(_models_cache.keys())}


@app.post("/predict/image", responses={200: {"content": {"image/png": {}}}})
async def predict_return_image(
    file: UploadFile = File(...),
    model: str = Form(DEFAULT_MODEL),
    conf: Optional[float] = Form(0.25),
    iou: Optional[float] = Form(0.45),
    max_det: Optional[int] = Form(100),
):
    """Return PNG image (with bounding boxes) for a single uploaded image."""
    try:
        mdl = load_model(model)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    content = await file.read()
    try:
        pil = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    try:
        results = mdl.predict(source=np.array(pil), conf=conf, iou=iou, max_det=max_det, verbose=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Predict error: {e}")

    res = results[0]

    # Try using res.plot() first
    pil_out = None
    try:
        plotted = res.plot()  # numpy array RGB
        if isinstance(plotted, np.ndarray):
            pil_out = _pil_from_numpy(plotted)
    except Exception:
        pil_out = None

    if pil_out is None:
        pil_out = pil.copy()
        if hasattr(res, "boxes") and res.boxes is not None:
            try:
                xy = res.boxes.xyxy.cpu().numpy() if hasattr(res.boxes.xyxy, "cpu") else np.array(res.boxes.xyxy)
                confs = res.boxes.conf.cpu().numpy() if hasattr(res.boxes.conf, "cpu") else np.array(res.boxes.conf)
                clss = res.boxes.cls.cpu().numpy() if hasattr(res.boxes.cls, "cpu") else np.array(res.boxes.cls)
                draw_boxes_on_pil(pil_out, xy, confs, clss, getattr(mdl, "names", {}) or {})
            except Exception:
                # ignore drawing errors
                pass

    buf = io.BytesIO()
    pil_out.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.post("/predict/json")
async def predict_return_json(
    file: UploadFile = File(...),
    model: str = Form(DEFAULT_MODEL),
    conf: Optional[float] = Form(0.25),
    iou: Optional[float] = Form(0.45),
    max_det: Optional[int] = Form(100),
):
    """Return JSON with boxes and image_base64 (PNG)"""
    try:
        mdl = load_model(model)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    content = await file.read()
    try:
        pil = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    try:
        results = mdl.predict(source=np.array(pil), conf=conf, iou=iou, max_det=max_det, verbose=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Predict error: {e}")

    res = results[0]
    names = getattr(mdl, "names", {}) or {}

    boxes_out = []
    if hasattr(res, "boxes") and res.boxes is not None:
        try:
            xy = res.boxes.xyxy.cpu().numpy() if hasattr(res.boxes.xyxy, "cpu") else np.array(res.boxes.xyxy)
            confs = res.boxes.conf.cpu().numpy() if hasattr(res.boxes.conf, "cpu") else np.array(res.boxes.conf)
            clss = res.boxes.cls.cpu().numpy() if hasattr(res.boxes.cls, "cpu") else np.array(res.boxes.cls)
            for (x1, y1, x2, y2), s, c in zip(xy, confs, clss):
                boxes_out.append({
                    "xyxy": [float(x1), float(y1), float(x2), float(y2)],
                    "score": float(s),
                    "class": int(c),
                    "label": str(names.get(int(c), str(int(c))))
                })
        except Exception:
            boxes_out = []

    # make image with boxes to return as base64
    pil_out = None
    try:
        plotted = res.plot()
        if isinstance(plotted, np.ndarray):
            pil_out = _pil_from_numpy(plotted)
    except Exception:
        pil_out = pil.copy()
        if boxes_out:
            try:
                draw_boxes_on_pil(pil_out,
                                  np.array([b['xyxy'] for b in boxes_out], dtype=float) if boxes_out else np.array([]),
                                  np.array([b['score'] for b in boxes_out], dtype=float) if boxes_out else np.array([]),
                                  np.array([b['class'] for b in boxes_out], dtype=int) if boxes_out else np.array([]),
                                  names)
            except Exception:
                pass

    buf = io.BytesIO()
    pil_out.save(buf, format="PNG")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    data_uri = f"data:image/png;base64,{img_b64}"

    return JSONResponse({"boxes": boxes_out, "image_base64": data_uri})


@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    model: str = Form(DEFAULT_MODEL),
    conf: Optional[float] = Form(0.25),
    iou: Optional[float] = Form(0.45),
    max_det: Optional[int] = Form(100),
):
    """Return coconut counts and confidence for your frontend.

    Response shape:
    {
      "young_coconuts": int,
      "mature_coconuts": int,
      "confidence": float
    }
    """
    try:
        mdl = load_model(model)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    content = await file.read()
    try:
        pil = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    try:
        results = mdl.predict(source=np.array(pil), conf=conf, iou=iou, max_det=max_det, verbose=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Predict error: {e}")

    res = results[0]
    names = getattr(mdl, "names", {}) or {}

    young = 0
    mature = 0
    scores: List[float] = []

    if hasattr(res, "boxes") and res.boxes is not None:
        try:
            confs = res.boxes.conf.cpu().numpy() if hasattr(res.boxes.conf, "cpu") else np.array(res.boxes.conf)
            clss = res.boxes.cls.cpu().numpy() if hasattr(res.boxes.cls, "cpu") else np.array(res.boxes.cls)
            for s, c in zip(confs, clss):
                scores.append(float(s))
                cls_name = names.get(int(c), str(int(c))).lower()
                if cls_name in ("kelapa_muda", "kelapa muda", "young_coconut", "youngcoconut"):
                    young += 1
                elif cls_name in ("kelapa_tua", "kelapa tua", "mature_coconut", "maturecoconut"):
                    mature += 1
                else:
                    # unknown class - ignore or handle as needed
                    pass
        except Exception:
            # if extraction fails, return zeros
            young = 0
            mature = 0
            scores = []

    confidence = round(max(scores) * 100, 2) if scores else 0.0

    return JSONResponse({
        "young_coconuts": young,
        "mature_coconuts": mature,
        "confidence": confidence,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
