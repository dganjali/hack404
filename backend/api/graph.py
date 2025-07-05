from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from models import ProcessStep, ProcessGraphVisualization
from graph.graph_engine import ProcessGraphEngine

router = APIRouter()

@router.post("/analyze")
async def analyze_graph(steps: List[ProcessStep]):
    """Analyze a process graph and return insights"""
    try:
        # Create graph engine
        graph_engine = ProcessGraphEngine()
        graph = graph_engine.build_graph_from_steps(steps)
        
        # Get graph analysis
        analysis = {
            'total_steps': len(steps),
            'total_dependencies': len(graph.edges()),
            'topological_order': graph_engine.get_topological_order(),
            'start_nodes': [n for n in graph.nodes() if graph.in_degree(n) == 0],
            'end_nodes': [n for n in graph.nodes() if graph.out_degree(n) == 0],
            'cycles_detected': len(list(graph_engine.graph.nodes())) != len(list(graph.nodes())) if hasattr(graph_engine, 'graph') else False
        }
        
        return analysis
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing graph: {str(e)}")

@router.post("/visualize")
async def create_visualization(steps: List[ProcessStep]):
    """Create visualization data for a process"""
    try:
        # Create graph engine
        graph_engine = ProcessGraphEngine()
        graph_engine.build_graph_from_steps(steps)
        
        # Create visualization
        visualization = graph_engine.create_visualization()
        
        return visualization
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating visualization: {str(e)}")

@router.post("/path")
async def find_path(steps: List[ProcessStep], start_step: str, end_step: str):
    """Find optimal path between two steps"""
    try:
        # Create graph engine
        graph_engine = ProcessGraphEngine()
        graph_engine.build_graph_from_steps(steps)
        
        # Find path
        path = graph_engine.get_optimal_path(start_step, end_step)
        
        if not path:
            raise HTTPException(status_code=404, detail="No path found between steps")
        
        # Get step details for the path
        path_steps = []
        for step_id in path:
            step = graph_engine.steps.get(step_id)
            if step:
                path_steps.append(step)
        
        return {
            'path': path,
            'steps': path_steps,
            'total_steps': len(path)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding path: {str(e)}")

@router.post("/estimate")
async def estimate_process(steps: List[ProcessStep], user_context: Dict[str, Any]):
    """Estimate time and cost for a process"""
    try:
        # Create graph engine
        graph_engine = ProcessGraphEngine()
        graph_engine.build_graph_from_steps(steps)
        
        # Filter for user context
        filtered_graph = graph_engine.filter_graph_for_user(user_context)
        
        # Get estimates
        estimated_time = graph_engine.estimate_completion_time(user_context)
        estimated_cost = graph_engine.estimate_total_cost(user_context)
        required_documents = graph_engine.get_required_documents(user_context)
        
        return {
            'estimated_time': estimated_time,
            'estimated_cost': estimated_cost,
            'required_documents': required_documents,
            'filtered_steps_count': len(filtered_graph.nodes()),
            'total_steps_count': len(steps)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error estimating process: {str(e)}")

@router.get("/status")
async def get_graph_status():
    """Get graph service status"""
    return {
        "status": "healthy",
        "services": {
            "graph_analysis": "available",
            "visualization": "available",
            "path_finding": "available",
            "estimation": "available"
        },
        "supported_operations": [
            "Graph analysis",
            "Visualization creation",
            "Path finding",
            "Time and cost estimation"
        ]
    } 