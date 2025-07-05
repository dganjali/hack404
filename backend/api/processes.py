from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Dict, Any
import uuid
from datetime import datetime

from models import (
    ProcessGraph, ProcessStep, UserContext, PersonalizedProcessRequest,
    PersonalizedProcessResponse, ScrapingRequest, ScrapingResponse
)
from agents.page_chunker import PageChunkerAgent
from agents.step_classifier import StepClassifierAgent
from agents.dependency_agent import DependencyAgent
from scrapers.web_scraper import scrape_government_sites
from graph.graph_engine import ProcessGraphEngine

router = APIRouter()

# In-memory storage for demo (replace with database in production)
processes_db = {}
sample_processes = {}

@router.get("/")
async def list_processes():
    """List all available processes"""
    return {
        "processes": list(processes_db.keys()),
        "sample_processes": list(sample_processes.keys()),
        "total": len(processes_db) + len(sample_processes)
    }

@router.get("/{process_id}")
async def get_process(process_id: str):
    """Get a specific process by ID"""
    if process_id in processes_db:
        return processes_db[process_id]
    elif process_id in sample_processes:
        return sample_processes[process_id]
    else:
        raise HTTPException(status_code=404, detail="Process not found")

@router.post("/scrape")
async def scrape_and_create_process(request: ScrapingRequest, background_tasks: BackgroundTasks):
    """Scrape government websites and create a new process"""
    try:
        # Scrape websites
        scraped_data = await scrape_government_sites(request.urls)
        
        # Process content with AI agents
        all_content = ""
        for data in scraped_data:
            if data['success']:
                all_content += f"\n\n{data['content']}"
        
        # Use AI agents to process content
        chunker = PageChunkerAgent()
        classifier = StepClassifierAgent()
        dependency_agent = DependencyAgent()
        
        # Chunk the content
        chunks = chunker.process(all_content)
        
        # Classify each chunk
        steps = []
        for chunk in chunks:
            step_data = classifier.process(chunk)
            if step_data:
                # Convert to ProcessStep
                step = ProcessStep(
                    id=step_data['id'],
                    type=step_data['type'],
                    title=step_data['title'],
                    description=step_data.get('description', ''),
                    cost=step_data.get('cost'),
                    duration=step_data.get('duration'),
                    required_documents=step_data.get('required_documents', []),
                    conditions=step_data.get('conditions', []),
                    depends_on=step_data.get('depends_on', []),
                    outputs=step_data.get('outputs', []),
                    metadata=step_data.get('metadata', {})
                )
                steps.append(step)
        
        # Analyze dependencies
        dependencies = dependency_agent.process([step.dict() for step in steps])
        steps = dependency_agent.update_step_dependencies(steps, dependencies)
        
        # Create process
        process_id = str(uuid.uuid4())
        process = ProcessGraph(
            id=process_id,
            title=request.process_name,
            description=request.description or "Process created from scraped content",
            steps=steps,
            source_urls=request.urls
        )
        
        # Store process
        processes_db[process_id] = process.dict()
        
        return ScrapingResponse(
            process_id=process_id,
            steps=steps,
            raw_content=scraped_data,
            processing_time=0.0  # TODO: Add timing
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing content: {str(e)}")

@router.post("/personalize")
async def personalize_process(request: PersonalizedProcessRequest):
    """Create a personalized process for a specific user"""
    try:
        # Get the process
        process_id = request.process_id
        if process_id in processes_db:
            process_data = processes_db[process_id]
        elif process_id in sample_processes:
            process_data = sample_processes[process_id]
        else:
            raise HTTPException(status_code=404, detail="Process not found")
        
        # Convert to ProcessGraph
        steps = [ProcessStep(**step) for step in process_data['steps']]
        process = ProcessGraph(**process_data)
        
        # Create graph engine
        graph_engine = ProcessGraphEngine()
        graph_engine.build_graph_from_steps(steps)
        
        # Filter for user context
        user_context_dict = request.user_context.dict()
        filtered_graph = graph_engine.filter_graph_for_user(user_context_dict)
        
        # Get filtered steps
        filtered_steps = []
        for node_id in filtered_graph.nodes():
            step = graph_engine.steps.get(node_id)
            if step:
                filtered_steps.append(step)
        
        # Create visualization
        visualization = graph_engine.create_visualization(filtered_graph)
        
        # Get estimates
        estimated_time = graph_engine.estimate_completion_time(user_context_dict)
        estimated_cost = graph_engine.estimate_total_cost(user_context_dict)
        required_documents = graph_engine.get_required_documents(user_context_dict)
        
        return PersonalizedProcessResponse(
            filtered_steps=filtered_steps,
            visualization=visualization,
            estimated_time=estimated_time,
            estimated_cost=estimated_cost,
            required_documents=required_documents
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error personalizing process: {str(e)}")

@router.post("/sample/food-truck")
async def create_sample_food_truck_process():
    """Create a sample food truck process for demonstration"""
    sample_steps = [
        ProcessStep(
            id="step_1",
            type="DECISION",
            title="Choose business structure",
            description="Decide whether to operate as sole proprietor, partnership, or corporation",
            conditions=["Province == 'Ontario'"],
            depends_on=[],
            outputs=["Business structure decision"]
        ),
        ProcessStep(
            id="step_2",
            type="ACTION",
            title="Register business name",
            description="Register your business name with the province",
            cost="$60 CAD",
            duration="1-2 weeks",
            required_documents=["Proof of name availability"],
            conditions=["Province == 'Ontario'"],
            depends_on=["step_1"],
            outputs=["Business name registration"]
        ),
        ProcessStep(
            id="step_3",
            type="ACTION",
            title="Apply for Business Number (BIN)",
            description="Get a Business Number from the Canada Revenue Agency",
            cost="$0",
            duration="1-2 weeks",
            required_documents=["Business name registration"],
            depends_on=["step_2"],
            outputs=["Business Number"]
        ),
        ProcessStep(
            id="step_4",
            type="DOCUMENT",
            title="Get food handler certification",
            description="Complete food safety training and certification",
            cost="$50-100 CAD",
            duration="1 day",
            required_documents=["Food handler certificate"],
            conditions=["Industry == 'food'"],
            depends_on=["step_1"],
            outputs=["Food handler certificate"]
        ),
        ProcessStep(
            id="step_5",
            type="ACTION",
            title="Apply for mobile vending license",
            description="Apply for mobile vending license with the city",
            cost="$500 CAD",
            duration="4-6 weeks",
            required_documents=["Business Number", "Food handler certificate"],
            conditions=["City == 'Toronto'", "Industry == 'food'"],
            depends_on=["step_3", "step_4"],
            outputs=["Mobile vending license"]
        ),
        ProcessStep(
            id="step_6",
            type="ACTION",
            title="Health inspection",
            description="Schedule and complete health inspection",
            cost="$200 CAD",
            duration="1-2 weeks",
            required_documents=["Mobile vending license"],
            conditions=["Industry == 'food'"],
            depends_on=["step_5"],
            outputs=["Health inspection certificate"]
        ),
        ProcessStep(
            id="step_7",
            type="ACTION",
            title="Fire department approval",
            description="Get fire safety approval for your food truck",
            cost="$150 CAD",
            duration="1-2 weeks",
            required_documents=["Health inspection certificate"],
            conditions=["Industry == 'food'"],
            depends_on=["step_6"],
            outputs=["Fire safety approval"]
        ),
        ProcessStep(
            id="step_8",
            type="ACTION",
            title="Get operating permit",
            description="Final step to get your operating permit",
            cost="$100 CAD",
            duration="1 week",
            required_documents=["Fire safety approval"],
            conditions=["Industry == 'food'"],
            depends_on=["step_7"],
            outputs=["Operating permit"]
        )
    ]
    
    sample_process = ProcessGraph(
        id="food-truck-sample",
        title="Starting a Food Truck in Toronto",
        description="Complete process for starting a food truck business in Toronto, Ontario",
        steps=sample_steps,
        source_urls=[
            "https://www.toronto.ca/business-economy/starting-a-business/",
            "https://www.ontario.ca/page/start-business"
        ]
    )
    
    sample_processes["food-truck-sample"] = sample_process.dict()
    
    return {
        "message": "Sample food truck process created",
        "process_id": "food-truck-sample",
        "steps_count": len(sample_steps)
    }

@router.delete("/{process_id}")
async def delete_process(process_id: str):
    """Delete a process"""
    if process_id in processes_db:
        del processes_db[process_id]
        return {"message": "Process deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Process not found") 