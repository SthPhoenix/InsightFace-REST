import os
import logging
import time
from typing import Optional, List

import pydantic
from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
from starlette.staticfiles import StaticFiles
from starlette.responses import StreamingResponse, RedirectResponse
from fastapi.responses import UJSONResponse
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)

from modules.processing import Processing
from env_parser import EnvConfigs

__version__ = "0.6.1.0"

dir_path = os.path.dirname(os.path.realpath(__file__))

# Read runtime settings from environment variables
configs = EnvConfigs()

logging.basicConfig(
    level=configs.log_level,
    format='%(asctime)s %(levelname)s - %(message)s',
    datefmt='[%H:%M:%S]',
)

processing = Processing(det_name=configs.models.det_name, rec_name=configs.models.rec_name,
                        ga_name=configs.models.ga_name,
                        device=configs.models.device,
                        max_size=configs.defaults.max_size,
                        max_rec_batch_size=configs.models.rec_batch_size,
                        backend_name=configs.models.backend_name,
                        force_fp16=configs.models.fp16,
                        triton_uri=configs.models.triton_uri)

app = FastAPI(
    title="InsightFace-REST",
    description="FastAPI wrapper for InsightFace API.",
    version=__version__,
    docs_url=None,
    redoc_url=None
)

example_img = 'test_images/Stallone.jpg'


class Images(BaseModel):
    data: Optional[List[str]] = pydantic.Field(default=None, example=None, description='List of base64 encoded images')
    urls: Optional[List[str]] = pydantic.Field(default=None,
                                               example=[example_img],
                                               description='List of images urls')


class BodyExtract(BaseModel):
    images: Images
    max_size: Optional[List[int]] = pydantic.Field(default=configs.defaults.max_size,
                                                   example=configs.defaults.max_size,
                                                   description='Resize all images to this proportions')

    threshold: Optional[float] = pydantic.Field(default=configs.defaults.threshold,
                                                example=configs.defaults.threshold,
                                                description='Detector threshold')

    embed_only: Optional[bool] = pydantic.Field(default=False,
                                                example=False,
                                                description='Treat input images as face crops and omit detection step')

    return_face_data: Optional[bool] = pydantic.Field(default=configs.defaults.return_face_data,
                                                      example=configs.defaults.return_face_data,
                                                      description='Return face crops encoded in base64')

    return_landmarks: Optional[bool] = pydantic.Field(default=configs.defaults.return_landmarks,
                                                      example=configs.defaults.return_landmarks,
                                                      description='Return face landmarks')

    extract_embedding: Optional[bool] = pydantic.Field(default=configs.defaults.extract_embedding,
                                                       example=configs.defaults.extract_embedding,
                                                       description='Extract face embeddings (otherwise only detect \
                                                       faces)')

    extract_ga: Optional[bool] = pydantic.Field(default=configs.defaults.extract_ga,
                                                example=configs.defaults.extract_ga,
                                                description='Extract gender/age')

    limit_faces: Optional[int] = pydantic.Field(default=0,
                                                example=0,
                                                description='Maximum number of faces to be processed')

    min_face_size: Optional[int] = pydantic.Field(default=0,
                                                  example=0,
                                                  description='Ignore faces smaller than this size')

    verbose_timings: Optional[bool] = pydantic.Field(default=False,
                                                     example=True,
                                                     description='Return all timings.')

    api_ver: Optional[str] = pydantic.Field(default=configs.defaults.api_ver,
                                            example='2',
                                            description='Output data serialization format.')


class BodyDraw(BaseModel):
    images: Images

    threshold: Optional[float] = pydantic.Field(default=configs.defaults.threshold,
                                                example=configs.defaults.threshold,
                                                description='Detector threshold')

    draw_landmarks: Optional[bool] = pydantic.Field(default=True,
                                                    example=True,
                                                    description='Return face landmarks')

    draw_scores: Optional[bool] = pydantic.Field(default=True,
                                                 example=True,
                                                 description='Draw detection scores')

    draw_sizes: Optional[bool] = pydantic.Field(default=True,
                                                example=True,
                                                description='Draw face sizes')

    limit_faces: Optional[int] = pydantic.Field(default=0,
                                                example=0,
                                                description='Maximum number of faces to be processed')

    min_face_size: Optional[int] = pydantic.Field(default=0,
                                                  example=0,
                                                  description='Ignore faces smaller than this size')


@app.post('/extract', tags=['Detection & recognition'])
async def extract(data: BodyExtract):
    """
    Face extraction/embeddings endpoint accept json with
    parameters in following format:

       - **images**: dict containing either links or data lists. (*required*)
       - **threshold**: Detection threshold. Default: 0.6 (*optional*)
       - **embed_only**: Treat input images as face crops (112x112 crops required), omit detection step. Default: False (*optional*)
       - **return_face_data**: Return face crops encoded in base64. Default: False (*optional*)
       - **return_landmarks**: Return face landmarks. Default: False (*optional*)
       - **extract_embedding**: Extract face embeddings (otherwise only detect faces). Default: True (*optional*)
       - **extract_ga**: Extract gender/age. Default: False (*optional*)
       - **limit_faces**: Maximum number of faces to be processed.  0 for unlimited number. Default: 0 (*optional*)
       - **verbose_timings**: Return all timings. Default: False (*optional*)
       - **api_ver**: Output data serialization format. Currently only version "1" is supported (*optional*)
       \f

       :return:
       List[List[dict]]
    """
    images = jsonable_encoder(data.images)
    output = await processing.extract(images, max_size=data.max_size, return_face_data=data.return_face_data,
                                      embed_only=data.embed_only, extract_embedding=data.extract_embedding,
                                      threshold=data.threshold, extract_ga=data.extract_ga,
                                      limit_faces=data.limit_faces, return_landmarks=data.return_landmarks,
                                      verbose_timings=data.verbose_timings, api_ver=data.api_ver)
    return UJSONResponse(output)


@app.post('/draw_detections', tags=['Detection & recognition'])
async def draw(data: BodyDraw):
    """
    Return image with drawn faces for testing purposes.

       - **images**: dict containing either links or data lists. (*required*)
       - **threshold**: Detection threshold. Default: 0.6 (*optional*)
       - **draw_landmarks**: Draw faces landmarks Default: True (*optional*)
       - **draw_scores**: Draw detection scores Default: True (*optional*)
       - **draw_sizes**: Draw face sizes Default: True (*optional*)
       - **limit_faces**: Maximum number of faces to be processed.  0 for unlimited number. Default: 0 (*optional*)
       \f
    """

    images = jsonable_encoder(data.images)
    output = await processing.draw(images, threshold=data.threshold,
                                   draw_landmarks=data.draw_landmarks, draw_scores=data.draw_scores,
                                   limit_faces=data.limit_faces, draw_sizes=data.draw_sizes)
    output.seek(0)
    return StreamingResponse(output, media_type="image/png")


@app.post('/multipart/draw_detections', tags=['Detection & recognition'])
async def draw_upl(file: bytes = File(...), threshold: float = Form(0.6), draw_landmarks: bool = Form(True),
                   draw_scores: bool = Form(True), draw_sizes: bool = Form(True), limit_faces: int = Form(0)):
    """
    Return image with drawn faces for testing purposes.

       - **file**: Image file (*required*)
       - **threshold**: Detection threshold. Default: 0.6 (*optional*)
       - **draw_landmarks**: Draw faces landmarks Default: True (*optional*)
       - **draw_scores**: Draw detection scores Default: True (*optional*)
       - **draw_sizes**: Draw face sizes Default: True (*optional*)
       - **limit_faces**: Maximum number of faces to be processed.  0 for unlimited number. Default: 0 (*optional*)
       \f
    """

    output = await processing.draw(file, threshold=threshold,
                                   draw_landmarks=draw_landmarks, draw_scores=draw_scores, draw_sizes=draw_sizes,
                                   limit_faces=limit_faces,
                                   multipart=True)
    output.seek(0)
    return StreamingResponse(output, media_type='image/jpg')


@app.get('/info', tags=['Utility'])
def info():
    """
    Enslist container configuration.

    """

    about = dict(
        version=__version__,
        tensorrt_version=os.getenv('TRT_VERSION', os.getenv('TENSORRT_VERSION')),
        log_level=configs.log_level,
        models=vars(configs.models),
        defaults=vars(configs.defaults),
    )
    about['models'].pop('ga_ignore', None)
    about['models'].pop('rec_ignore', None)
    about['models'].pop('device', None)
    return about


@app.get('/', include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url="/docs")


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
        swagger_favicon_url='/static/favicon.png'
    )


@app.get(app.swagger_ui_oauth2_redirect_url, include_in_schema=False)
async def swagger_ui_redirect():
    return get_swagger_ui_oauth2_redirect_html()


@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=app.title + " - ReDoc",
        redoc_js_url="/static/redoc.standalone.js",
    )
