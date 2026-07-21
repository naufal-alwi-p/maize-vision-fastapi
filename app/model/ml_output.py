from pydantic import BaseModel

class InferenceResult(BaseModel):
    original_image: bytes
    grad_cam_image: bytes
    scores: list[float]
    binary_scores: list[float]
