import RiskGraph3D from '../components/RiskGraph3D';

export default function GraphPage() {
    return (
        <div className="space-y-6">
            <div>
                <h1 className="text-3xl font-bold tracking-tight">Risk Node Graph</h1>
                <p className="text-gray-500 dark:text-gray-400 mt-1">
                    Explore the relationships between risk factors in the Episodic Knowledge Mesh.
                </p>
            </div>
            <RiskGraph3D />
        </div>
    );
}
