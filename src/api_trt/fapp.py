import json
import os
from json import JSONEncoder
import ujson

from typing import Optional, Set, List, Dict

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
from starlette.responses import StreamingResponse, RedirectResponse
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)

from starlette.staticfiles import StaticFiles
import pydantic

from modules.processing import Processing
from modules.utils.helpers import parse_size, tobool

import logging
logging.basicConfig(
    level='DEBUG',
    format='%(asctime)s %(levelname)s - %(message)s',
    datefmt='[%H:%M:%S]',
)



__version__ = "0.5"

# TODO Refactor reading variables
dir_path = os.path.dirname(os.path.realpath(__file__))

port = os.getenv('PORT', 18080)

device = os.getenv("DEVICE", 'cuda')

backend_name = os.getenv('INFERENCE_BACKEND','trt')
rec_name = os.getenv("REC_NAME", "arcface_r100_v1")
det_name = os.getenv("DET_NAME", "retinaface_mnet025_v2")
ga_name = os.getenv("GA_NAME", "genderage_v1")

ga_ignore = tobool(os.getenv('GA_IGNORE', False))
rec_ignore = tobool(os.getenv('REC_IGNORE', False))

if rec_ignore:
    rec_name = None
if ga_ignore:
    ga_name = None

# Global parameters
max_size = parse_size(os.getenv('MAX_SIZE'))
return_face_data = tobool(os.getenv('DEF_RETURN_FACE_DATA', False))
extract_embedding = tobool(os.getenv('DEF_EXTRACT_EMBEDDING', True))
extract_ga = tobool(os.getenv('DEF_EXTRACT_GA', False))
api_ver = os.getenv('DEF_API_VER', "1")


# MTCNN parameters
select_largest = tobool(os.getenv("SELECT_LARGEST", True))
keep_all = tobool(os.getenv("KEEP_ALL", True))
min_face_size = int(os.getenv("MIN_FACE_SIZE", 20))
mtcnn_factor = float(os.getenv("MTCNN_FACTOR", 0.709))


processing = Processing(det_name=det_name, rec_name=rec_name, ga_name=ga_name, device=device,
                        max_size=max_size,select_largest=select_largest, keep_all=keep_all, min_face_size=min_face_size,
                        mtcnn_factor=mtcnn_factor, backend_name=backend_name)

app = FastAPI(
    title="InsightFace-REST",
    description="FastAPI wrapper for InsightFace API.",
    version=__version__,
    docs_url=None,
    redoc_url=None
)

example_img = 'test_images/TH.png'


class Images(BaseModel):
    data: Optional[List[str]] = pydantic.Field(default=None, example=None, description='List of base64 encoded images')
    urls: Optional[List[str]] = pydantic.Field(default=None,
                                               example=[example_img],
                                               description='List of images urls')


class BodyExtract(BaseModel):
    images: Images
    max_size: Optional[List[int]] = pydantic.Field(default=max_size,
                                                   example=max_size,
                                                   description='Resize all images to this proportions')

    threshold: Optional[float] = pydantic.Field(default=0.6,
                                                example=0.6,
                                                description='Detector threshold')

    return_face_data: Optional[bool] = pydantic.Field(default=return_face_data,
                                                      example=return_face_data,
                                                      description='Return face crops encoded in base64')
    extract_embedding: Optional[bool] = pydantic.Field(default=extract_embedding,
                                                       example=extract_embedding,
                                                       description='Extract face embeddings (otherwise only detect \
                                                       faces)')
    extract_ga: Optional[bool] = pydantic.Field(default=extract_ga,
                                                example=extract_ga,
                                                description='Extract gender/age')
    api_ver: Optional[str] = pydantic.Field(default=api_ver,
                                            example=api_ver,
                                            description='Output data serialization format. Currently only version "1" \
                                            is supported')


@app.get('/')
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





@app.post('/extract')
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
    output = processing.embed(images, max_size=data.max_size, return_face_data=data.return_face_data,
                              extract_embedding=data.extract_embedding, threshold=data.threshold, extract_ga=data.extract_ga,
                              api_ver=data.api_ver)

    return output


@app.post('/draw_detections')
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
    output = processing.draw(images, max_size=data.max_size, threshold=data.threshold,
                             return_face_data=data.return_face_data,
                             extract_embedding=data.extract_embedding, extract_ga=data.extract_ga)
    output.seek(0)
    return StreamingResponse(output, media_type="image/png")
