from contextlib import asynccontextmanager

from fastapi import FastAPI

from routers.predictions import router as predictions_router
from services.model_loader import BASE_DIR, load_model_bundle
from services.pose_pipeline import PosePipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 시작 시 한 번만 실행
    app.state.bundle = load_model_bundle()
    app.state.pose = PosePipeline(app.state.bundle.project_config, BASE_DIR)
    yield


app = FastAPI(title="Vision Action AI Server", lifespan=lifespan)
app.include_router(predictions_router)


@app.get("/health")
def health():
    bundle = getattr(app.state, "bundle", None)
    if bundle is None:
        return {"status": "ok", "model_loaded": False}
    return {
        "status": "ok",
        "model_loaded": True,
        "classes": bundle.training_info["classes"],
        "feature_count": bundle.training_info["feature_count"],
        "sequence_steps": bundle.training_info["sequence_steps"],
    }