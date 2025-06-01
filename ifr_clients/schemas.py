from typing import Optional, List

from pydantic import BaseModel, ConfigDict


class BaseModelMod(BaseModel):
    model_config = ConfigDict(extra='ignore')


class Took(BaseModelMod):
    total_ms: Optional[float]


class FacesResponse(BaseModelMod):
    num_det: Optional[int]
    prob: Optional[float]
    size: Optional[int]
    bbox: Optional[List[int]]
    landmarks: Optional[List[List]]
    norm: Optional[float]
    vec: Optional[List]
    facedata: Optional[str]


class ImageResponse(BaseModelMod):
    status: str
    took_ms: Optional[float]
    faces: Optional[list[FacesResponse]]


class RecognitionResponse(BaseModelMod):
    took: Took
    data: List[ImageResponse]
