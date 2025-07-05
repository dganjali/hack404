import networkx as nx
from typing import List, Dict, Any, Optional, Tuple
from models import ProcessStep, ProcessGraphVisualization, GraphNode, GraphEdge
import json
import uuid

class ProcessGraphEngine:
    """Graph engine for building and managing process decision trees"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.steps = {}
        self.user_context = {}
    
    def build_graph_from_steps(self, steps: List[ProcessStep]) -> nx.DiGraph:
        """Build a directed graph from process steps"""
        self.graph.clear()
        self.steps = {step.id: step for step in steps}
        
        # Add nodes
        for step in steps:
            self.graph.add_node(step.id, data=step)
        
        # Add edges based on dependencies
        for step in steps:
            for dep_id in step.depends_on:
                if dep_id in self.steps:
                    self.graph.add_edge(dep_id, step.id)
        
        # Validate graph (remove cycles, ensure DAG)
        self._validate_and_fix_graph()
        
        return self.graph
    
    def _validate_and_fix_graph(self):
        """Validate graph and fix issues like cycles"""
        # Check for cycles
        try:
            cycles = list(nx.simple_cycles(self.graph))
            if cycles:
                print(f"Found cycles in graph: {cycles}")
                # Remove cycles by removing some edges
                for cycle in cycles:
                    if len(cycle) > 1:
                        # Remove the last edge in the cycle
                        self.graph.remove_edge(cycle[-1], cycle[0])
        except nx.NetworkXNoCycle:
            pass  # No cycles found
    
    def get_topological_order(self) -> List[str]:
        """Get steps in topological order"""
        try:
            return list(nx.topological_sort(self.graph))
        except nx.NetworkXError:
            # If there are cycles, return nodes in order of addition
            return list(self.graph.nodes())
    
    def get_step_dependencies(self, step_id: str) -> List[str]:
        """Get all dependencies for a step"""
        if step_id not in self.graph:
            return []
        
        # Get all ancestors (dependencies)
        ancestors = nx.ancestors(self.graph, step_id)
        return list(ancestors)
    
    def get_step_dependents(self, step_id: str) -> List[str]:
        """Get all steps that depend on this step"""
        if step_id not in self.graph:
            return []
        
        # Get all descendants (dependents)
        descendants = nx.descendants(self.graph, step_id)
        return list(descendants)
    
    def filter_graph_for_user(self, user_context: Dict[str, Any]) -> nx.DiGraph:
        """Filter graph based on user context"""
        self.user_context = user_context
        
        # Create a copy of the graph
        filtered_graph = self.graph.copy()
        
        # Remove nodes that don't apply to this user
        nodes_to_remove = []
        for node_id in filtered_graph.nodes():
            step = self.steps.get(node_id)
            if step and not self._step_applies_to_user(step, user_context):
                nodes_to_remove.append(node_id)
        
        # Remove nodes and their edges
        for node_id in nodes_to_remove:
            filtered_graph.remove_node(node_id)
        
        return filtered_graph
    
    def _step_applies_to_user(self, step: ProcessStep, user_context: Dict[str, Any]) -> bool:
        """Check if a step applies to the user based on conditions"""
        if not step.conditions:
            return True
        
        for condition in step.conditions:
            if not self._evaluate_condition(condition, user_context):
                return False
        
        return True
    
    def _evaluate_condition(self, condition: str, user_context: Dict[str, Any]) -> bool:
        """Evaluate a condition against user context"""
        condition_lower = condition.lower()
        
        # Simple condition evaluation
        if 'ontario' in condition_lower and user_context.get('province', '').lower() != 'ontario':
            return False
        if 'toronto' in condition_lower and user_context.get('city', '').lower() != 'toronto':
            return False
        if 'food' in condition_lower and user_context.get('industry', '').lower() != 'food':
            return False
        if 'incorporated' in condition_lower and not user_context.get('is_incorporated', False):
            return False
        
        return True
    
    def create_visualization(self, graph: Optional[nx.DiGraph] = None) -> ProcessGraphVisualization:
        """Create visualization data for the graph"""
        if graph is None:
            graph = self.graph
        
        # Use hierarchical layout
        pos = nx.spring_layout(graph, k=3, iterations=50)
        
        # Create nodes
        nodes = []
        for node_id, position in pos.items():
            step = self.steps.get(node_id)
            if step:
                nodes.append(GraphNode(
                    id=node_id,
                    position={'x': float(position[0] * 300), 'y': float(position[1] * 200)},
                    data=step,
                    type=self._get_node_type(step)
                ))
        
        # Create edges
        edges = []
        for source, target in graph.edges():
            edges.append(GraphEdge(
                id=f"{source}-{target}",
                source=source,
                target=target
            ))
        
        return ProcessGraphVisualization(
            nodes=nodes,
            edges=edges,
            metadata={
                'total_steps': len(nodes),
                'total_dependencies': len(edges),
                'user_context': self.user_context
            }
        )
    
    def _get_node_type(self, step: ProcessStep) -> str:
        """Get React Flow node type based on step type"""
        type_mapping = {
            'ACTION': 'actionNode',
            'DOCUMENT': 'documentNode',
            'FEE': 'feeNode',
            'WAIT': 'waitNode',
            'DECISION': 'decisionNode',
            'INFO': 'infoNode'
        }
        return type_mapping.get(step.type.value, 'defaultNode')
    
    def get_optimal_path(self, start_step: str, end_step: str) -> List[str]:
        """Find optimal path between two steps"""
        try:
            path = nx.shortest_path(self.graph, start_step, end_step)
            return path
        except nx.NetworkXNoPath:
            return []
    
    def get_all_paths(self) -> List[List[str]]:
        """Get all possible paths through the graph"""
        # Find all paths from start nodes to end nodes
        start_nodes = [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0]
        end_nodes = [n for n in self.graph.nodes() if self.graph.out_degree(n) == 0]
        
        all_paths = []
        for start in start_nodes:
            for end in end_nodes:
                try:
                    paths = list(nx.all_simple_paths(self.graph, start, end))
                    all_paths.extend(paths)
                except nx.NetworkXNoPath:
                    continue
        
        return all_paths
    
    def estimate_completion_time(self, user_context: Dict[str, Any]) -> str:
        """Estimate total completion time for the process"""
        filtered_graph = self.filter_graph_for_user(user_context)
        
        total_days = 0
        for node_id in filtered_graph.nodes():
            step = self.steps.get(node_id)
            if step and step.duration:
                # Extract number of days from duration string
                import re
                match = re.search(r'(\d+)\s*(days?|weeks?|months?)', step.duration.lower())
                if match:
                    number = int(match.group(1))
                    unit = match.group(2)
                    if 'week' in unit:
                        total_days += number * 7
                    elif 'month' in unit:
                        total_days += number * 30
                    else:
                        total_days += number
        
        if total_days == 0:
            return "Time estimate not available"
        elif total_days < 7:
            return f"{total_days} days"
        elif total_days < 30:
            weeks = total_days // 7
            return f"{weeks} weeks"
        else:
            months = total_days // 30
            return f"{months} months"
    
    def estimate_total_cost(self, user_context: Dict[str, Any]) -> str:
        """Estimate total cost for the process"""
        filtered_graph = self.filter_graph_for_user(user_context)
        
        total_cost = 0
        for node_id in filtered_graph.nodes():
            step = self.steps.get(node_id)
            if step and step.cost:
                # Extract cost from cost string
                import re
                match = re.search(r'\$?([\d,]+(?:\.\d{2})?)', step.cost)
                if match:
                    cost_str = match.group(1).replace(',', '')
                    try:
                        total_cost += float(cost_str)
                    except ValueError:
                        pass
        
        if total_cost == 0:
            return "Cost estimate not available"
        else:
            return f"${total_cost:.2f} CAD"
    
    def get_required_documents(self, user_context: Dict[str, Any]) -> List[str]:
        """Get all required documents for the user"""
        filtered_graph = self.filter_graph_for_user(user_context)
        
        documents = set()
        for node_id in filtered_graph.nodes():
            step = self.steps.get(node_id)
            if step and step.required_documents:
                documents.update(step.required_documents)
        
        return list(documents) 