import os

from fastapi import APIRouter, HTTPException
from starlette.responses import RedirectResponse

from if_rest.core.processing import ProcessingDep
from if_rest.schemas import Images
from if_rest.settings import Settings

settings = Settings()
router = APIRouter()

__version__ = os.getenv('IFR_VERSION', '0.9.5.0')


@router.get('/info', tags=['Utility'])
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


@router.get('/health', tags=['Utility'])
async def check_health(processing: ProcessingDep):
    """
    Execute recognition request with default parameters to verify recognition is actually working

    """

    data = Images(urls=["test_images/Stallone.jpg"])

    try:
        res = await processing.extract(images=data)
        faces = res.get('data', [{}])[0].get('faces', [])
        assert len(faces) >= 1
        return {'status': 'ok'}
    except Exception:
        raise HTTPException(500, detail='self check failed')


@router.get('/', include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url="/docs")
