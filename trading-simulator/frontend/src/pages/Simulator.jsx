import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { useNavigate } from "react-router-dom";


export default function Simulator() {
  const navigate = useNavigate();
  const [strategies, setStrategies] = useState([]);
  const [filtered, setFiltered] = useState([]);

  const [filters, setFilters] = useState({
    exchange: '',
    symbol: '',
    time_horizon: '',
  });

  useEffect(() => {
    fetch('http://localhost:3001/api/strategies')
      .then(res => res.json())
      .then(data => {
        if (data.success && Array.isArray(data.data)) {
          setStrategies(data.data);
          setFiltered(data.data);
        } else {
          console.warn('Invalid response:', data);
        }
      })
      .catch(err => console.error('❌ Error fetching strategies:', err));
  }, []);

  const handleFilterChange = (e) => {
    const { name, value } = e.target;
    const newFilters = { ...filters, [name]: value };
    setFilters(newFilters);

    const filteredData = strategies.filter(s => {
      const matchExchange = newFilters.exchange ? s.exchange === newFilters.exchange : true;
      const matchSymbol = newFilters.symbol ? s.symbol === newFilters.symbol : true;
      const matchTimeHorizon = newFilters.time_horizon ? s.time_horizon === newFilters.time_horizon : true;
      return matchExchange && matchSymbol && matchTimeHorizon;
    });

    filteredData.sort((a, b) => (b.pnl_sum || 0) - (a.pnl_sum || 0));
    setFiltered(filteredData);
  };

  const uniqueExchanges = Array.from(new Set(strategies.map(s => s.exchange))).filter(Boolean);
  const uniqueSymbols = Array.from(new Set(strategies.map(s => s.symbol))).filter(Boolean);
  const uniqueTimeHorizons = Array.from(new Set(strategies.map(s => s.time_horizon))).filter(Boolean);

  const pnlValues = filtered.map(s => s.pnl_sum || 0);
  const totalStrategies = filtered.length;
  const maxPnl = pnlValues.length ? Math.max(...pnlValues) : 0;
  const minPnl = pnlValues.length ? Math.min(...pnlValues) : 0;
  const avgPnl = pnlValues.length ? pnlValues.reduce((a, b) => a + b, 0) / pnlValues.length : 0;

  const sortedFiltered = [...filtered].sort((a, b) => (b.pnl_sum || 0) - (a.pnl_sum || 0));

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-100 to-emerald-50 p-6">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-4xl font-bold text-center text-slate-800 mb-2">Strategies Performance</h1>
        <p className="text-center text-gray-600 mb-8">Monitor and analyze your trading strategies with real-time performance metrics</p>

        {/* Filters */}
        <div className="flex flex-col md:flex-row md:items-center justify-center gap-6 mb-8">
          <select
            name="exchange"
            value={filters.exchange}
            onChange={handleFilterChange}
            className="w-full md:w-1/4 px-4 py-2 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-emerald-400"
          >
            <option value="">All Exchanges</option>
            {uniqueExchanges.map(exch => (
              <option key={exch} value={exch}>{exch}</option>
            ))}
          </select>

          <select
            name="symbol"
            value={filters.symbol}
            onChange={handleFilterChange}
            className="w-full md:w-1/4 px-4 py-2 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-emerald-400"
          >
            <option value="">All Symbols</option>
            {uniqueSymbols.map(sym => (
              <option key={sym} value={sym}>{sym}</option>
            ))}
          </select>

          <select
            name="time_horizon"
            value={filters.time_horizon}
            onChange={handleFilterChange}
            className="w-full md:w-1/4 px-4 py-2 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-emerald-400"
          >
            <option value="">All Time Horizons</option>
            {uniqueTimeHorizons.map(th => (
              <option key={th} value={th}>{th}</option>
            ))}
          </select>
        </div>

        {/* Summary Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-6 mb-8">
          {[
            { label: 'Total Strategies', value: totalStrategies, color: 'text-slate-800' },
            { label: 'Max PnL', value: `${maxPnl.toFixed(2)}%`, color: 'text-green-600' },
            { label: 'Min PnL', value: `${minPnl.toFixed(2)}%`, color: 'text-red-600' },
            { label: 'Avg PnL', value: `${avgPnl.toFixed(2)}%`, color: avgPnl >= 0 ? 'text-green-500' : 'text-red-500' }
          ].map((card, idx) => (
            <div key={idx} className="bg-white p-6 rounded-xl shadow-lg border border-gray-200 transform hover:-translate-y-1 transition-all">
              <p className="text-sm text-gray-500">{card.label}</p>
              <h2 className={`text-3xl font-extrabold ${card.color}`}>{card.value}</h2>
            </div>
          ))}
        </div>

        {/* Table */}
        <div className="overflow-x-auto bg-white shadow-lg rounded-lg border border-gray-200">
          <table className="min-w-full text-sm text-left">
            <thead className="bg-green-700 text-white">
              <tr>
                <th className="px-5 py-3">Strategy</th>
                <th className="px-5 py-3">Datetime</th>
                <th className="px-5 py-3">Exchange</th>
                <th className="px-5 py-3">Symbol</th>
                <th className="px-5 py-3">Time Horizon</th>
                <th className="px-5 py-3">Total PnL</th>
              </tr>
            </thead>
            <tbody>
              {sortedFiltered.length > 0 ? (
                sortedFiltered.map((s, index) => (
                  <tr key={index} className="hover:bg-gray-100 border-b transition">
                    <td className="px-5 py-3 font-bold text-gray-800 capitalize">
                      <Link
                        to={`/strategies/${s.strategy_name}`}
                        className="text-emerald-700 hover:underline"
                      >
                        {s.strategy_name.replace('strategy_', 'Strategy ')}
                      </Link>
                    </td>
                    <td className="px-5 py-3">{s.datetime || 'N/A'}</td>
                    <td className="px-5 py-3">{s.exchange}</td>
                    <td className="px-5 py-3">{s.symbol}</td>
                    <td className="px-5 py-3">{s.time_horizon}</td>
                    <td className="px-5 py-3">
                      <span
                        className={`inline-block px-3 py-1 text-sm font-bold rounded-full ${
                          s.pnl_sum > 5
                            ? 'bg-green-200 text-green-900'
                            : s.pnl_sum >= 0
                            ? 'bg-green-100 text-green-800'
                            : s.pnl_sum > -5
                            ? 'bg-red-100 text-red-700'
                            : 'bg-red-200 text-red-900'
                        }`}
                      >
                        {s.pnl_sum !== null ? `${s.pnl_sum.toFixed(2)}%` : 'N/A'}
                      </span>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan="6" className="text-center text-gray-500 py-6">
                    No strategy data available.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
