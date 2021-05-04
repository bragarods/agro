from fastapi import FastAPI, Depends
from api.forecast import views

# app = FastAPI(dependencies=[Depends(auth.validUserPass)])

app = FastAPI()

app.include_router(views.router)

@app.post("/")
async def root():
    return {"message": "Hello bigger"}