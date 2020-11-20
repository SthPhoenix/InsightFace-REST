import os
import logging
from typing import Optional, List

import pydantic
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
from starlette.staticfiles import StaticFiles
from starlette.responses import StreamingResponse, RedirectResponse
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)

from modules.processing import Processing
from env_parser import EnvConfigs

__version__ = "0.5.8"

# TODO Refactor reading variables
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
                        max_size=configs.defaults.max_size, select_largest=configs.mtcnn.select_largest,
                        keep_all=configs.mtcnn.keep_all,
                        min_face_size=configs.mtcnn.min_face_size, mtcnn_factor=configs.mtcnn.mtcnn_factor,
                        backend_name=configs.models.backend_name, force_fp16=configs.models.fp16)

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

    return_face_data: Optional[bool] = pydantic.Field(default=configs.defaults.return_face_data,
                                                      example=configs.defaults.return_face_data,
                                                      description='Return face crops encoded in base64')
    extract_embedding: Optional[bool] = pydantic.Field(default=configs.defaults.extract_embedding,
                                                       example=configs.defaults.extract_embedding,
                                                       description='Extract face embeddings (otherwise only detect \
                                                       faces)')
    extract_ga: Optional[bool] = pydantic.Field(default=configs.defaults.extract_ga,
                                                example=configs.defaults.extract_ga,
                                                description='Extract gender/age')
    api_ver: Optional[str] = pydantic.Field(default=configs.defaults.api_ver,
                                            example=configs.defaults.api_ver,
                                            description='Output data serialization format. Currently only version "1" \
                                            is supported')


@app.post('/extract', tags=['Detection & recognition'])
async def extract(data: BodyExtract):
    """
    Face extraction/embeddings endpoint accept json with
    parameters in following format:

       - **images**: dict containing either links or data lists. (*required*)
       - **max_size**: Resize all images to this proportions. Default: [640,480] (*optional*)
       - **threshold**: Detection threshold. Default: 0.6 (*optional*)
       - **return_face_data**: Return face crops encoded in base64. Default: False (*optional*)
       - **extract_embedding**: Extract face embeddings (otherwise only detect faces). Default: True (*optional*)
       - **extract_ga**: Extract gender/age. Default: False (*optional*)
       - **api_ver**: Output data serialization format. Currently only version "1" is supported (*optional*)
       \f

       :return:
       List[List[dict]]
    """

    images = jsonable_encoder(data.images)
    output = await processing.embed(images, max_size=data.max_size, return_face_data=data.return_face_data,
                                    extract_embedding=data.extract_embedding, threshold=data.threshold,
                                    extract_ga=data.extract_ga,
                                    api_ver=data.api_ver)

    return output


@app.post('/draw_detections', tags=['Detection & recognition'])
async def draw(data: BodyExtract):
    """
    Return image with drawn faces for testing purposes, accepts data in same format as extract endpoint,
    but processes only first image.

       - **images**: dict containing either links or data lists. (*required*)
       - **max_size**: Resize all images to this proportions. Default: [640,480] (*optional*)
       - **threshold**: Detection threshold. Default: 0.6 (*optional*)
       - **return_face_data**: Return face crops encoded in base64. Default: False (*optional*)
       - **extract_embedding**: Extract face embeddings (otherwise only detect faces). Default: True (*optional*)
       - **extract_ga**: Extract gender/age. Default: False (*optional*)
       - **api_ver**: Output data serialization format. Currently only version "1" is supported (*optional*)
       \f

       :return:
       List[List[dict]]

    """

    images = jsonable_encoder(data.images)
    output = await processing.draw(images, max_size=data.max_size, threshold=data.threshold,
                                   return_face_data=data.return_face_data,
                                   extract_embedding=data.extract_embedding, extract_ga=data.extract_ga)
    output.seek(0)
    return StreamingResponse(output, media_type="image/png")


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
