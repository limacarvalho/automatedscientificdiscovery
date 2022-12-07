from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

# from api.api_v1.endpoints.api import api_router
from app.api.api_v1 import api
from app.core import settings


app = FastAPI(
    title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

app.include_router(api.api_router, prefix=settings.API_V1_STR)
