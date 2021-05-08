from fastapi import FastAPI, Depends
from api.forecast import views
# mlflow
from fastapi.middleware.wsgi import WSGIMiddleware
from ml_flow.main import mlapp

# app = FastAPI(dependencies=[Depends(auth.validUserPass)])

app = FastAPI()

app.include_router(views.router)

@app.post("/")
async def root():
    return {"message": "Hello bigger"}

# mlflow

app.mount("/mlflow", WSGIMiddleware(mlapp))