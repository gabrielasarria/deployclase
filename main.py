from fastapi import FastAPI
from routers import routerDiabetes

app = FastAPI()

app.include_router(routerDiabetes.router)