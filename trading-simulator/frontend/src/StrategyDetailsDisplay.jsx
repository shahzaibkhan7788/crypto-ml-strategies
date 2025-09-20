import { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";

export default function StrategyDetailsDisplay() {
  const { tableName } = useParams();
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Extract only the model name from the table name
  const modelName = tableName.split("_").slice(4, -1).join(" ");

  useEffect(() => {
    const fetchTable = async () => {
      try {
        const res = await fetch(`http://localhost:3001/api/strategies/table_data?table_name=${tableName}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const result = await res.json();
        setData(result.rows || []);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    fetchTable();
  }, [tableName]);

  if (loading) return <p className="text-center py-6 text-gray-600">Loading table data...</p>;
  if (error) return <p className="text-red-500 text-center py-6">Error: {error}</p>;
  if (data.length === 0) return <p className="text-center py-6 text-gray-500">No data found for this table.</p>;

  return (
    <div className="p-8 space-y-6">
      <Link to="/" className="text-green-600 underline mb-4 inline-block">⬅ Back to all strategies</Link>
      
      <h2 className="text-3xl font-extrabold mb-6 text-center text-green-700">{modelName}</h2>

      <div className="overflow-x-auto rounded-xl shadow-lg border border-green-400">
        <table className="min-w-full text-sm text-left text-gray-700">
          <thead className="bg-gradient-to-r from-green-400 to-emerald-600 text-white sticky top-0">
            <tr>
              {Object.keys(data[0]).map(key => (
                <th key={key} className="px-4 py-3 uppercase tracking-wide">{key.replace(/_/g, " ")}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.map((row, idx) => (
              <tr key={idx} className={`transition hover:bg-green-50 ${idx % 2 === 0 ? "bg-white" : "bg-green-50"}`}>
                {Object.values(row).map((val, i) => (
                  <td key={i} className="px-4 py-2">
                    {typeof val === "number" ? val.toFixed(2) : val}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="mt-4 text-sm text-gray-500 text-center">
        Total Rows: <span className="font-semibold text-green-600">{data.length}</span>
      </p>
    </div>
  );
}
