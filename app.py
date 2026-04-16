from fastapi import FastAPI, Request
from pydantic import BaseModel
from implement_LLM import generate_response, load_chunks
from contextlib import asynccontextmanager

MAX_PROMPT_LENGTH=2000
DEFAULT_TEMPERATURE=0.3
DEFAULT_MAX_NEW_TOKENS=512
DEFAULT_DISTANCE_THRESHOLD=20
TOP_K=5
@asynccontextmanager
async def lifespan(app: FastAPI):
    faiss_index,all_chunks=load_chunks()
    app.state.faiss_index = faiss_index
    app.state.all_chunks=all_chunks
    yield
app = FastAPI(
    title='RAG System on domain knowledge base',
    description='Domain base include 3 anime wiki fandom: Naruto, HunterxHunter, SAO',
    lifespan=lifespan
    )


class QueryRequest(BaseModel):
    query: str
    top_k: int=TOP_K
    temperature:float=DEFAULT_TEMPERATURE
    max_tokens:int = DEFAULT_MAX_NEW_TOKENS
    threshold:float=DEFAULT_DISTANCE_THRESHOLD

@app.post('/generate')
def generate(request: QueryRequest,R:Request):
    query=request.query
    top_k=request.top_k
    temperature=request.temperature
    max_tokens=request.max_tokens
    threshold=request.threshold
    text=generate_response(query=query,
                           top_k=top_k,
                           faiss_index=R.app.state.faiss_index,
                           all_chunks=R.app.state.all_chunks,
                           temperature=temperature,
                           max_tokens=max_tokens,
                           threshold=threshold)
    return {'answer':text}