import React, { useCallback, useMemo } from 'react';
import ReactFlow, {
  Node,
  Edge,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
  Connection,
  EdgeChange,
  NodeChange,
} from 'reactflow';
import 'reactflow/dist/style.css';

interface ProcessStep {
  id: string;
  type: string;
  title: string;
  description?: string;
  cost?: string;
  duration?: string;
  required_documents?: string[];
  conditions?: string[];
  depends_on?: string[];
  outputs?: string[];
}

interface ProcessFlowProps {
  steps: ProcessStep[];
  onNodeClick?: (node: Node) => void;
  selectedNode?: string | null;
}

const ProcessFlow: React.FC<ProcessFlowProps> = ({ 
  steps, 
  onNodeClick, 
  selectedNode 
}) => {
  // Convert steps to React Flow nodes
  const initialNodes: Node[] = useMemo(() => {
    return steps.map((step, index) => ({
      id: step.id,
      type: step.type.toLowerCase() + 'Node',
      position: { 
        x: (index % 3) * 300, 
        y: Math.floor(index / 3) * 150 
      },
      data: { 
        label: step.title,
        step: step,
        isSelected: selectedNode === step.id
      },
      style: {
        width: 200,
        minHeight: 80,
      }
    }));
  }, [steps, selectedNode]);

  // Create edges from dependencies
  const initialEdges: Edge[] = useMemo(() => {
    const edges: Edge[] = [];
    steps.forEach(step => {
      if (step.depends_on) {
        step.depends_on.forEach(depId => {
          edges.push({
            id: `${depId}-${step.id}`,
            source: depId,
            target: step.id,
            type: 'smoothstep',
            animated: false,
          });
        });
      }
    });
    return edges;
  }, [steps]);

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  const onConnect = useCallback(
    (params: Connection) => setEdges((eds) => addEdge(params, eds)),
    [setEdges],
  );

  const handleNodeClick = useCallback(
    (event: React.MouseEvent, node: Node) => {
      onNodeClick?.(node);
    },
    [onNodeClick]
  );

  // Custom node types
  const nodeTypes = useMemo(() => ({
    actionNode: ActionNode,
    documentNode: DocumentNode,
    feeNode: FeeNode,
    waitNode: WaitNode,
    decisionNode: DecisionNode,
    infoNode: InfoNode,
  }), []);

  return (
    <div style={{ width: '100%', height: '600px' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeClick={handleNodeClick}
        nodeTypes={nodeTypes}
        fitView
        attributionPosition="bottom-left"
      >
        <Controls />
        <Background color="#aaa" gap={16} />
      </ReactFlow>
    </div>
  );
};

// Custom Node Components
const BaseNode: React.FC<{ data: any }> = ({ data }) => {
  const { step, isSelected } = data;
  
  return (
    <div className={`step-node ${step.type.toLowerCase()} ${isSelected ? 'ring-2 ring-primary-500' : ''}`}>
      <div className="font-semibold text-sm mb-1">{step.title}</div>
      {step.description && (
        <div className="text-xs text-gray-600 mb-2">{step.description}</div>
      )}
      <div className="flex flex-wrap gap-1">
        {step.cost && (
          <span className="text-xs bg-yellow-100 text-yellow-800 px-2 py-1 rounded">
            {step.cost}
          </span>
        )}
        {step.duration && (
          <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">
            {step.duration}
          </span>
        )}
      </div>
    </div>
  );
};

const ActionNode: React.FC<{ data: any }> = ({ data }) => (
  <BaseNode data={data} />
);

const DocumentNode: React.FC<{ data: any }> = ({ data }) => (
  <BaseNode data={data} />
);

const FeeNode: React.FC<{ data: any }> = ({ data }) => (
  <BaseNode data={data} />
);

const WaitNode: React.FC<{ data: any }> = ({ data }) => (
  <BaseNode data={data} />
);

const DecisionNode: React.FC<{ data: any }> = ({ data }) => (
  <BaseNode data={data} />
);

const InfoNode: React.FC<{ data: any }> = ({ data }) => (
  <BaseNode data={data} />
);

export default ProcessFlow; 