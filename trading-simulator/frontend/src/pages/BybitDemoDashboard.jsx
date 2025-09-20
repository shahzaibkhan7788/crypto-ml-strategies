import React, { useEffect, useState } from 'react';
import { useNavigate } from "react-router-dom";


const BybitDemoDashboard = () => {
  const [data, setData] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('bybit_demo_btc');

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await fetch('http://localhost:3001/api/bybit-demo/all');
        const json = await res.json();
        if (!res.ok) throw new Error(json.error || 'Failed to fetch');
        setData(json.data);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) return <div className="text-center mt-20 text-xl">Loading...</div>;
  if (error) return <div className="text-center mt-20 text-red-500">{error}</div>;

  const tables = Object.keys(data);

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">


    <div className="text-center mb-8">
        <h1 className="text-4xl font-extrabold text-emerald-700 mb-2">
            Bybit Demo Trade Center
        </h1>
        <p className="text-gray-600 text-lg mb-1">
            Track and manage BTC, ETH, and SOL trades in real-time from your Bybit demo account.
        </p>
        <p className="text-gray-400 text-sm italic">
            Interactive dashboard with live PnL, trade status, and detailed trade history.
        </p>
    </div>


      {/* Tabs */}
      <div className="flex justify-center mb-6 space-x-4">
        {tables.map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-2 rounded-lg font-semibold ${
              activeTab === tab ? 'bg-emerald-700 text-white shadow-lg' : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            {tab.replace('bybit_demo_', '').toUpperCase()}
          </button>
        ))}
      </div>

      {/* Table */}
      <div className="overflow-x-auto rounded-lg shadow-lg border">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-emerald-700 text-white">
            <tr>
              {data[activeTab].length > 0 &&
                Object.keys(data[activeTab][0]).map((col) => (
                  <th key={col} className="px-4 py-2 text-left text-sm font-medium">
                    {col.replace(/_/g, ' ').toUpperCase()}
                  </th>
                ))}
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {data[activeTab].map((row) => (
              <tr key={row.id} className="hover:bg-gray-50">
                {Object.values(row).map((val, idx) => (
                  <td key={idx} className="px-4 py-2 text-sm text-gray-700">
                    {val === null ? '-' : val.toString()}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        {data[activeTab].length === 0 && (
          <div className="text-center py-6 text-gray-500">No records found</div>
        )}
      </div>
    </div>
  );
};

export default BybitDemoDashboard;
