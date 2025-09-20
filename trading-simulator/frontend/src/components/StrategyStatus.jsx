import { useEffect, useState } from "react";

export default function StrategyStatus({ strategyName }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  // Mapping category prefixes to friendly display names
  const categoryTitles = {
    profitability: "Profitability Metrics",
    activity: "Trading Activity",
    risk: "Risk Assessment",
    efficiency: "Capital Efficiency",
    consistency: "Performance Consistency",
    portfolio: "Portfolio Impact",
    execution: "Execution Quality",
  };

  const categories = Object.keys(categoryTitles);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await fetch(`http://localhost:3001/api/strategy/${strategyName}/full-data`);
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

  return (
    <div className="p-4">
      <h2 className="text-2xl font-bold mb-6">
        Performance Overview
      </h2>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {categories.map((category) => {
          const categoryData = Object.entries(data)
            .filter(([key]) => key.startsWith(category + "_"))
            .map(([key, value]) => ({
              name: key.replace(category + "_", ""),
              value,
            }));

          return (
            <div
              key={category}
              className="bg-white rounded-lg shadow-md p-4 border border-gray-200"
            >
              <h3 className="text-lg font-semibold mb-2">
                {categoryTitles[category]}
              </h3>
              <div className="space-y-1">
                {categoryData.map((item, idx) => (
                  <div key={idx} className="flex justify-between text-sm">
                    <span className="font-medium">{item.name}</span>
                    <span className="text-gray-700">{item.value}</span>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
