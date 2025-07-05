from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
from enum import Enum
from datetime import datetime

class StepType(str, Enum):
    ACTION = "ACTION"
    DOCUMENT = "DOCUMENT"
    FEE = "FEE"
    WAIT = "WAIT"
    DECISION = "DECISION"
    INFO = "INFO"

class ProcessStep(BaseModel):
    id: str
    type: StepType
    title: str
    description: Optional[str] = None
    url: Optional[str] = None
    cost: Optional[str] = None
    duration: Optional[str] = None
    required_documents: List[str] = Field(default_factory=list)
    conditions: List[str] = Field(default_factory=list)
    depends_on: List[str] = Field(default_factory=list)
    outputs: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class UserContext(BaseModel):
    province: str
    city: str
    business_type: str
    industry: str
    is_incorporated: bool = False
    additional_info: Dict[str, Any] = Field(default_factory=dict)

class ProcessGraph(BaseModel):
    id: str
    title: str
    description: str
    steps: List[ProcessStep]
    source_urls: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class ScrapingRequest(BaseModel):
    urls: List[str]
    process_name: str
    description: Optional[str] = None

class ScrapingResponse(BaseModel):
    process_id: str
    steps: List[ProcessStep]
    raw_content: Dict[str, Any]
    processing_time: float

class GraphNode(BaseModel):
    id: str
    position: Dict[str, float]  # x, y coordinates
    data: ProcessStep
    type: str = "default"

class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    label: Optional[str] = None

class ProcessGraphVisualization(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    metadata: Dict[str, Any] = Field(default_factory=dict)

class PersonalizedProcessRequest(BaseModel):
    user_context: UserContext
    process_id: str

class PersonalizedProcessResponse(BaseModel):
    filtered_steps: List[ProcessStep]
    visualization: ProcessGraphVisualization
    estimated_time: str
    estimated_cost: str
    required_documents: List[str] 