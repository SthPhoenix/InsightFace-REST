from fastapi import APIRouter

from if_rest.api.routes.v1.recognition import router as rec_router
from if_rest.api.routes.v1.service import router as service_router

v1_router = APIRouter()

v1_router.include_router(rec_router)
v1_router.include_router(service_router)
