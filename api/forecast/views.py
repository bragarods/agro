from fastapi import APIRouter, HTTPException, BackgroundTasks, Header
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from rainfall_forecast.data import inmetData
from datetime import datetime
import json

inmet = inmetData()

router = APIRouter(prefix='/forecast', tags=['forecast'])

@router.post("/predictions")
async def predictions(cd_estacao: str = Header(None)):
    pred = inmet.prophetPredict(cd_estacao=cd_estacao)
    pred_filt = pred[['ds', 'yhat']].to_dict(orient='records')
    resp = jsonable_encoder(pred_filt)
    return JSONResponse(content=resp) 