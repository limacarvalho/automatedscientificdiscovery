from fastapi import APIRouter
from . import knockoff



# app = FastAPI(
#     title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json"
# )

api_router = APIRouter()
api_router.include_router(knockoff.router, prefix="/knockoff", tags=["knockoff"])

