from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.middleware.cors import CORSMiddleware
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.inference import model
import time
import logging
import tempfile
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    contents = await file.read()
    tmp.write(contents)
    tmp.close()
    t0 = time.time()
    results = model(tmp.name)
    out = []
    for r in results:
        if hasattr(r, 'boxes') and r.boxes is not None:
            for box in r.boxes:
                out.append({
                    "xyxy": np.array(box.xyxy).tolist(),
                    "xywh": np.array(box.xywh).tolist(),
                    "conf": float(box.conf[0]) if hasattr(box, 'conf') else None,
                    "class": int(box.cls[0]) if hasattr(box, 'cls') else None
                })
    Path(tmp.name).unlink(missing_ok=True)
    latency = time.time() - t0
    logging.info(f"predict latency={latency:.3f}s size={len(contents)} bytes results={len(out)}")
    return JSONResponse({"detections": out, "latency_sec": latency})

@app.get("/metrics")
def metrics():
    # Minimal metrics endpoint (Prometheus-like could be added later)
    return PlainTextResponse("service_up 1\n")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
