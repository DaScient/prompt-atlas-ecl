from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path
ROOT=Path(__file__).resolve().parent
app=FastAPI()
app.mount('/static', StaticFiles(directory=ROOT/'static'), name='static')
@app.get('/')
def index():
    return HTMLResponse((ROOT/'static'/'index.html').read_text())
