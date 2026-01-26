import { useRef, useEffect, useState, useCallback } from 'react';
import ForceGraph3D from 'react-force-graph-3d';
import axios from 'axios';
import * as THREE from 'three';

interface GraphNode {
    id: string;
    group: string;
    risk_factor: string;
    val: number;
    x?: number;
    y?: number;
    z?: number;
}

interface GraphLink {
    source: string | GraphNode;
    target: string | GraphNode;
    value: number;
}

interface GraphData {
    nodes: GraphNode[];
    links: GraphLink[];
}

const riskColors: Record<string, string> = {
    low: '#22c55e',
    medium: '#f59e0b',
    high: '#ef4444',
    critical: '#7c3aed',
};

export default function RiskGraph3D() {
    const fgRef = useRef<any>(null);
    const [graphData, setGraphData] = useState<GraphData>({ nodes: [], links: [] });
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const fetchGraphData = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            const token = localStorage.getItem('auth_token');
            const response = await axios.get<GraphData>('http://localhost:8000/graph/risk', {
                headers: { Authorization: `Bearer ${token}` },
            });
            setGraphData(response.data);
        } catch (err: any) {
            setError(err.message || 'Failed to fetch graph data');
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchGraphData();
    }, [fetchGraphData]);

    useEffect(() => {
        if (fgRef.current && graphData.nodes.length > 0) {
            fgRef.current.d3Force('charge')?.strength(-150);
            fgRef.current.d3Force('link')?.distance(80);
        }
    }, [graphData]);

    const handleNodeClick = useCallback((node: GraphNode) => {
        if (fgRef.current) {
            const distance = 100;
            const distRatio = 1 + distance / Math.hypot(node.x || 0, node.y || 0, node.z || 0);
            fgRef.current.cameraPosition(
                { x: (node.x || 0) * distRatio, y: (node.y || 0) * distRatio, z: (node.z || 0) * distRatio },
                node,
                1500
            );
        }
    }, []);

    if (loading) {
        return (
            <div className="flex items-center justify-center h-[600px] bg-gray-900 rounded-xl">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="flex flex-col items-center justify-center h-[600px] bg-gray-900 rounded-xl text-white">
                <p className="text-red-400 mb-4">Error: {error}</p>
                <button
                    onClick={fetchGraphData}
                    className="px-4 py-2 bg-primary rounded-lg hover:opacity-80 transition"
                >
                    Retry
                </button>
            </div>
        );
    }

    if (graphData.nodes.length === 0) {
        return (
            <div className="flex flex-col items-center justify-center h-[600px] bg-gray-900 rounded-xl text-white">
                <p className="text-gray-400 mb-4">No graph data available. Train the model first.</p>
                <button
                    onClick={fetchGraphData}
                    className="px-4 py-2 bg-primary rounded-lg hover:opacity-80 transition"
                >
                    Refresh
                </button>
            </div>
        );
    }

    return (
        <div className="relative h-[600px] bg-gray-900 rounded-xl overflow-hidden">
            <div className="absolute top-4 right-4 z-10 flex gap-2">
                <button
                    onClick={fetchGraphData}
                    className="px-4 py-2 bg-white/10 hover:bg-white/20 backdrop-blur-md rounded-lg text-white text-sm transition"
                >
                    🔄 Refresh
                </button>
            </div>
            <div className="absolute top-4 left-4 z-10 flex flex-col gap-1 bg-black/50 backdrop-blur-md p-3 rounded-lg">
                <span className="text-white text-xs font-bold uppercase mb-1">Risk Levels</span>
                {Object.entries(riskColors).map(([level, color]) => (
                    <div key={level} className="flex items-center gap-2">
                        <span className="w-3 h-3 rounded-full" style={{ backgroundColor: color }}></span>
                        <span className="text-white text-xs capitalize">{level}</span>
                    </div>
                ))}
            </div>
            <ForceGraph3D
                ref={fgRef}
                graphData={graphData}
                nodeLabel={(node: GraphNode) => `${node.risk_factor} (${node.group})`}
                nodeColor={(node: GraphNode) => riskColors[node.group] || '#888'}
                nodeVal={(node: GraphNode) => node.val * 3}
                linkWidth={(link: GraphLink) => (link.value || 0.5) * 2}
                linkOpacity={0.6}
                backgroundColor="#111827"
                onNodeClick={handleNodeClick}
                nodeThreeObject={(node: GraphNode) => {
                    const geometry = new THREE.SphereGeometry(5, 16, 16);
                    const material = new THREE.MeshLambertMaterial({
                        color: riskColors[node.group] || '#888',
                        transparent: true,
                        opacity: 0.9,
                    });
                    return new THREE.Mesh(geometry, material);
                }}
            />
        </div>
    );
}
