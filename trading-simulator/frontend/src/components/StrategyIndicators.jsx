import { useEffect, useState } from "react";

export default function StrategyIndicators({ strategyName }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await fetch(`http://localhost:3001/api/strategy/${strategyName}/indicators`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = await res.json();
        setData(json.data);
      } catch (err) {
        console.error("Error fetching:", err);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [strategyName]);

  if (loading) return <div className="p-4 text-center">Loading...</div>;
  if (!data) return <div className="p-4 text-center">No data found.</div>;

  // Card 1 - Strategy details
  const alwaysInclude = ["strategy_name", "exchange", "symbol", "time_horizon"];
  const strategyDetails = alwaysInclude.map((key) => ({
    name: key,
    value: data[key],
  }));

  // Card 2 - Boolean indicators
  const booleanIndicators = Object.entries(data)
    .filter(([key, value]) => typeof value === "boolean")
    .map(([key, value]) => ({ name: key, value: value ? "True" : "False" }));

  // Card 3 - Other parameters
  const otherParams = Object.entries(data)
    .filter(
      ([key, value]) =>
        !alwaysInclude.includes(key) &&
        typeof value !== "boolean"
    )
    .map(([key, value]) => ({ name: key, value }));

  return (
    <div className="p-6">
      <h2 className="text-2xl font-bold mb-6">
        Indicators – {data.strategy_name}
      </h2>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        
        {/* Card 1 */}
        <div className="bg-white rounded-xl shadow p-4 border border-gray-200 h-64 flex flex-col">
          <h3 className="text-lg font-semibold mb-3">Strategy Details</h3>
          <div className="space-y-2 text-sm">
            {strategyDetails.map((item, idx) => (
              <div key={idx} className="flex justify-between">
                <span className="font-medium">{item.name}</span>
                <span className="text-gray-700">{item.value}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Card 2 */}
        <div className="bg-white rounded-xl shadow p-4 border border-gray-200 h-64 flex flex-col">
          <h3 className="text-lg font-semibold mb-3">Boolean Indicators</h3>
          <div className="overflow-y-auto pr-2 space-y-1 text-sm">
            {booleanIndicators.map((item, idx) => (
              <div key={idx} className="flex justify-between">
                <span className="font-medium">{item.name}</span>
                <span className={item.value === "True" ? "text-green-600" : "text-red-500"}>
                  {item.value}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Card 3 */}
        <div className="bg-white rounded-xl shadow p-4 border border-gray-200 h-64 flex flex-col">
          <h3 className="text-lg font-semibold mb-3">Other Parameters</h3>
          <div className="overflow-y-auto pr-2 space-y-1 text-sm">
            {otherParams.map((item, idx) => (
              <div key={idx} className="flex justify-between">
                <span className="font-medium">{item.name}</span>
                <span className="text-gray-700">{String(item.value)}</span>
              </div>
            ))}
          </div>
        </div>

      </div>
    </div>
  );
}
