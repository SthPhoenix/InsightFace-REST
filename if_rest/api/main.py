import logging
import os
import ssl
from contextlib import asynccontextmanager

import aiohttp
from aiohttp import ClientTimeout, TCPConnector
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_offline import FastAPIOffline

from if_rest.api.routes.v1 import v1_router
from if_rest.core.processing import get_processing
from if_rest.logger import logger
from if_rest.settings import Settings

__version__ = os.getenv('IFR_VERSION', '0.9.5.0')

dir_path = os.path.dirname(os.path.realpath(__file__))

# Read runtime settings from environment variables
settings = Settings()

logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s %(levelname)s - %(message)s',
    datefmt='[%H:%M:%S]',
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Perform any necessary setup when the application starts up.
    This includes initializing the `processing` object aiohttp.ClientSession.

    Raises:
        Exception: If an error occurs during processing initialization.
    """

    logger.info(f"Starting processing module...")
    try:
        timeout = ClientTimeout(total=60.)
        if settings.defaults.sslv3_hack:
            ssl_context = ssl._create_unverified_context()
            ssl_context.set_ciphers('DEFAULT')
            dl_client = aiohttp.ClientSession(timeout=timeout, connector=TCPConnector(ssl=ssl_context))
        else:
            dl_client = aiohttp.ClientSession(timeout=timeout, connector=TCPConnector(ssl=False))
        processing = await get_processing()
        await processing.start(dl_client=dl_client)
        logger.info(f"Processing module ready!")
    except Exception as e:
        logger.error(e)
        exit(1)
    yield


def get_app() -> FastAPI:
    application = FastAPIOffline(
        title="InsightFace-REST",
        description="Face recognition REST API",
        version=__version__,
        lifespan=lifespan
    )

    application.add_middleware(
        CORSMiddleware,  # noqa
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*']
    )
    application.include_router(v1_router)

    return application


app = get_app()
