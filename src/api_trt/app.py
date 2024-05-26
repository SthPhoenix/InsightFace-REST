import logging
import os
from typing import Optional, List

import aiohttp
import msgpack
from aiohttp import ClientTimeout, TCPConnector
from fastapi import File, Form, Header, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import UJSONResponse
from fastapi_offline import FastAPIOffline
from starlette.responses import StreamingResponse, RedirectResponse, PlainTextResponse

from api_trt.logger import logger
from api_trt.modules.processing import Processing
from api_trt.schemas import BodyDraw, BodyExtract
from api_trt.settings import Settings

__version__ = os.getenv('IFR_VERSION','0.9.0.0')

dir_path = os.path.dirname(os.path.realpath(__file__))

# Read runtime settings from environment variables
settings = Settings()

logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s %(levelname)s - %(message)s',
    datefmt='[%H:%M:%S]',
)



processing = None

app = FastAPIOffline(
    title="InsightFace-REST",
    description="FastAPI wrapper for InsightFace API.",
    version=__version__,
)


@app.on_event('startup')
async def startup():
    """
    Perform any necessary setup when the application starts up.
    This includes initializing the `processing` object aiohttp.ClientSession.

    Raises:
        Exception: If an error occurs during processing initialization.
    """
    logger.info(f"Starting processing module...")
    global processing
    try:
        timeout = ClientTimeout(total=60., )
        dl_client = aiohttp.ClientSession(timeout=timeout, connector=TCPConnector(verify_ssl=False))
        processing = Processing(det_name=settings.models.det_name, rec_name=settings.models.rec_name,
                                ga_name=settings.models.ga_name,
                                mask_detector=settings.models.mask_detector,
                                max_size=settings.models.max_size,
                                max_rec_batch_size=settings.models.rec_batch_size,
                                max_det_batch_size=settings.models.det_batch_size,
                                backend_name=settings.models.inference_backend,
                                force_fp16=settings.models.force_fp16,
                                triton_uri=settings.models.triton_uri,
                                root_dir='/models',
                                dl_client=dl_client
                                )
        logger.info(f"Processing module ready!")
    except Exception as e:
        logger.error(e)
        exit(1)


@app.post('/extract', tags=['Detection & recognition'])
async def extract(data: BodyExtract, accept: Optional[List[str]] = Header(None)):
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
       - **msgpack**: Serialize output to msgpack format for transfer. Default: False (*optional*)
       \f

       :return:
       List[List[dict]]
    """
    try:
        images = jsonable_encoder(data.images)
        output = await processing.extract(images, max_size=data.max_size, return_face_data=data.return_face_data,
                                          embed_only=data.embed_only, extract_embedding=data.extract_embedding,
                                          threshold=data.threshold, extract_ga=data.extract_ga,
                                          limit_faces=data.limit_faces, min_face_size=data.min_face_size,
                                          return_landmarks=data.return_landmarks,
                                          detect_masks=data.detect_masks,
                                          verbose_timings=data.verbose_timings)

        if data.msgpack or 'application/x-msgpack' in accept:
            return PlainTextResponse(msgpack.dumps(output, use_single_float=True), media_type='application/x-msgpack')
        else:
            return UJSONResponse(output)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    try:
        images = jsonable_encoder(data.images)
        output = await processing.draw(images, threshold=data.threshold,
                                       draw_landmarks=data.draw_landmarks, draw_scores=data.draw_scores,
                                       limit_faces=data.limit_faces, min_face_size=data.min_face_size,
                                       draw_sizes=data.draw_sizes,
                                       detect_masks=data.detect_masks)
        output.seek(0)
        return StreamingResponse(output, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/multipart/draw_detections', tags=['Detection & recognition'])
async def draw_upl(file: bytes = File(...), threshold: float = Form(0.6), draw_landmarks: bool = Form(True),
                   draw_scores: bool = Form(True), draw_sizes: bool = Form(True), limit_faces: int = Form(0), use_rotation: bool = Form(False)):
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
    try:
        output = await processing.draw(file, threshold=threshold,
                                       draw_landmarks=draw_landmarks, draw_scores=draw_scores, draw_sizes=draw_sizes,
                                       limit_faces=limit_faces,
                                       multipart=True)
        output.seek(0)
        return StreamingResponse(output, media_type='image/jpg')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/info', tags=['Utility'])
def info():
    """
    Enlist container configuration.

    """
    try:
        about = dict(
            version=__version__,
            tensorrt_version=os.getenv('TRT_VERSION', os.getenv('TENSORRT_VERSION')),
            log_level=settings.log_level,
            models=settings.models.dict(),
            defaults=settings.defaults.dict(),
        )
        about['models'].pop('ga_ignore', None)
        about['models'].pop('rec_ignore', None)
        about['models'].pop('mask_ignore', None)
        about['models'].pop('device', None)
        return about
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/', include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url="/docs")
