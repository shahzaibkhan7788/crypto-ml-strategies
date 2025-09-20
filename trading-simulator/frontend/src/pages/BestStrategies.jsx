import { useEffect, useState, useMemo } from "react";
import { useNavigate } from "react-router-dom";


export default function BestStrategies() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [strategies, setStrategies] = useState([]);
  const [error, setError] = useState(null);
  const [filters, setFilters] = useState({
    model: "",
    symbol: "",
    time_horizon: "",
  });

  // Fetch strategies
  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await fetch("http://localhost:3001/api/strategies/all_best_tables");
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        setStrategies(data.results || []);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  // Filters & sorting
  const sortedFilteredStrategies = useMemo(() => {
    let data = [...strategies];
    if (filters.model) data = data.filter(item => item.model === filters.model);
    if (filters.symbol) data = data.filter(item => item.symbol === filters.symbol);
    if (filters.time_horizon) data = data.filter(item => item.time_horizon === filters.time_horizon);
    return data.sort((a, b) => b.max_last_row_pnl_sum - a.max_last_row_pnl_sum);
  }, [strategies, filters]);

  // Navigate on model click
  const handleModelClick = (strategy) => {
    navigate(`/strategy/${strategy.table_name}`);
  };

  // Stats calculation
  const stats = useMemo(() => {
    if (sortedFilteredStrategies.length === 0) return {};
    const maxItem = sortedFilteredStrategies.reduce((prev, curr) =>
      curr.max_last_row_pnl_sum > prev.max_last_row_pnl_sum ? curr : prev
    );
    const minItem = sortedFilteredStrategies.reduce((prev, curr) =>
      curr.max_last_row_pnl_sum < prev.max_last_row_pnl_sum ? curr : prev
    );
    const avg = sortedFilteredStrategies.reduce((sum, item) => sum + item.max_last_row_pnl_sum, 0) / sortedFilteredStrategies.length;
    return { max: maxItem, min: minItem, avg };
  }, [sortedFilteredStrategies]);

  // Unique filters
  const uniqueModels = [...new Set(strategies.map(s => s.model))];
  const uniqueSymbols = [...new Set(strategies.map(s => s.symbol))];
  const uniqueTimeHorizons = [...new Set(strategies.map(s => s.time_horizon))];

  // Helper functions for colors
  const getModelColor = (model) => {
    const colors = [
      "bg-emerald-100 text-emerald-800",
      "bg-indigo-100 text-indigo-800",
      "bg-pink-100 text-pink-800",
      "bg-yellow-100 text-yellow-800",
      "bg-purple-100 text-purple-800",
      "bg-teal-100 text-teal-800"
    ];
    const index = model.charCodeAt(0) % colors.length;
    return colors[index];
  };

  const getPnlColor = (value) => {
    if (value > 1000) return "bg-green-100 text-green-800";
    if (value > 500) return "bg-amber-100 text-amber-800";
    return "bg-red-100 text-red-800";
  };

  if (loading) return <div className="text-center text-gray-600 py-6">Loading strategies...</div>;
  if (error) return <div className="text-red-500 text-center py-6">Error: {error}</div>;
  if (strategies.length === 0) return <div className="text-center text-gray-500 py-6">No strategies found.</div>;
  

  

  return (
    <div className="p-8 space-y-8">
            {/* Page Heading */}
      <div className="text-center mb-4">
        <h1 className="text-4xl md:text-5xl font-extrabold text-gray-800">
          Best Trading Strategies
        </h1>
        <p className="text-gray-600 text-base md:text-lg mt-2">
          Explore, filter, and compare top-performing strategies to optimize your trading decisions.
        </p>
      </div>
      {/* Stats Section */}
      {stats.max && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="p-6 rounded-xl shadow-lg text-white bg-gradient-to-r from-green-400 to-emerald-600">
            <div className="flex items-center gap-3">
              <span className="text-xl">⬆️</span>
              <h3 className="text-lg font-medium">Highest PnL</h3>
            </div>
            <p className="text-3xl font-bold mt-2">{stats.max.max_last_row_pnl_sum.toFixed(2)}</p>
            <p className="text-sm opacity-80">Model: {stats.max.model}</p>
          </div>
          <div className="p-6 rounded-xl shadow-lg text-white bg-gradient-to-r from-green-500 to-emerald-700">
            <div className="flex items-center gap-3">
              <span className="text-xl">📊</span>
              <h3 className="text-lg font-medium">Average PnL</h3>
            </div>
            <p className="text-3xl font-bold mt-2">{stats.avg.toFixed(2)}</p>
          </div>
          <div className="p-6 rounded-xl shadow-lg text-white bg-gradient-to-r from-green-600 to-lime-600">
            <div className="flex items-center gap-3">
              <span className="text-xl">⬇️</span>
              <h3 className="text-lg font-medium">Lowest PnL</h3>
            </div>
            <p className="text-3xl font-bold mt-2">{stats.min.max_last_row_pnl_sum.toFixed(2)}</p>
            <p className="text-sm opacity-80">Model: {stats.min.model}</p>
          </div>
        </div>
      )}

      {/* Filters */}
      <div className="bg-white p-4 rounded-xl shadow-lg flex flex-wrap justify-center gap-4">
        <select className="border rounded-full px-4 py-2 text-sm shadow-sm" value={filters.model} onChange={e => setFilters({ ...filters, model: e.target.value })}>
          <option value="">All Models</option>
          {uniqueModels.map((m, idx) => (<option key={idx} value={m}>{m}</option>))}
        </select>
        <select className="border rounded-full px-4 py-2 text-sm shadow-sm" value={filters.symbol} onChange={e => setFilters({ ...filters, symbol: e.target.value })}>
          <option value="">All Symbols</option>
          {uniqueSymbols.map((s, idx) => (<option key={idx} value={s}>{s}</option>))}
        </select>
        <select className="border rounded-full px-4 py-2 text-sm shadow-sm" value={filters.time_horizon} onChange={e => setFilters({ ...filters, time_horizon: e.target.value })}>
          <option value="">All Time Horizons</option>
          {uniqueTimeHorizons.map((t, idx) => (<option key={idx} value={t}>{t}</option>))}
        </select>
        <button className="bg-gray-200 text-sm px-4 py-2 rounded-full shadow-sm hover:bg-gray-300 transition" onClick={() => setFilters({ model: "", symbol: "", time_horizon: "" })}>
          Reset Filters
        </button>
      </div>

      {/* Summary Table */}
      <div className="overflow-x-auto bg-white rounded-xl shadow-lg border border-green-400">
        <table className="min-w-full text-sm text-left text-gray-700">
          <thead className="bg-gradient-to-r from-green-400 to-emerald-600 text-white text-xs uppercase font-semibold">
            <tr>
              <th className="px-6 py-4">Model</th>
              <th className="px-6 py-4">Exchange</th>
              <th className="px-6 py-4">Symbol</th>
              <th className="px-6 py-4">Time Horizon</th>
              <th className="px-6 py-4">Max PnL (last row)</th>
            </tr>
          </thead>
          <tbody>
            {sortedFilteredStrategies.map((item, idx) => (
              <tr key={idx} className={`border-b transition hover:bg-green-50 ${idx % 2 === 0 ? "bg-white" : "bg-green-50"}`}>
                <td className="px-6 py-3">
                  <span
                    className={`px-3 py-1 rounded-full font-semibold cursor-pointer ${getModelColor(item.model)}`}
                    onClick={() => handleModelClick(item)}
                  >
                    {item.model}
                  </span>
                </td>
                <td className="px-6 py-3">{item.exchange}</td>
                <td className="px-6 py-3">{item.symbol}</td>
                <td className="px-6 py-3">{item.time_horizon}</td>
                <td className="px-6 py-3">
                  <span className={`px-3 py-1 rounded-full font-bold ${getPnlColor(item.max_last_row_pnl_sum)}`}>
                    {item.max_last_row_pnl_sum?.toFixed(2)}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
