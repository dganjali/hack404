from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn

from api.processes import router as processes_router
from api.scraping import router as scraping_router
from api.graph import router as graph_router

app = FastAPI(
    title="ProcessUnravel Pro API",
    description="AI-powered government process mapping and decision trees",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(processes_router, prefix="/api/processes", tags=["processes"])
app.include_router(scraping_router, prefix="/api/scraping", tags=["scraping"])
app.include_router(graph_router, prefix="/api/graph", tags=["graph"])

@app.get("/")
async def root():
    return {
        "message": "ProcessUnravel Pro API",
        "version": "1.0.0",
        "endpoints": {
            "processes": "/api/processes",
            "scraping": "/api/scraping", 
            "graph": "/api/graph"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ProcessUnravel Pro"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 