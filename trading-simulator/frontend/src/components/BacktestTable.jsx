import { useEffect, useState } from "react";

export default function BacktestTable({ strategyName }) {
  const [records, setRecords] = useState([]);
  const [loading, setLoading] = useState(true);
  const [currentPage, setCurrentPage] = useState(1);
  const rowsPerPage = 10;

  useEffect(() => {
    const fetchBacktestData = async () => {
      try {
        console.log(`Fetching backtest for: ${strategyName}`);
        const res = await fetch(`http://localhost:3001/api/backtest/${strategyName}`);
        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        const data = await res.json();
        setRecords(Array.isArray(data.records) ? data.records : []);
      } catch (err) {
        console.error("Error fetching backtest:", err);
        setRecords([]);
      } finally {
        setLoading(false);
      }
    };

    fetchBacktestData();
  }, [strategyName]);

  const totalPages = Math.ceil(records.length / rowsPerPage);
  const currentData = records.slice(
    (currentPage - 1) * rowsPerPage,
    currentPage * rowsPerPage
  );

  const goToPage = (page) => {
    if (page >= 1 && page <= totalPages) setCurrentPage(page);
  };

  const getPageNumbers = () => {
    const maxVisible = 3;
    let start = Math.max(currentPage - 1, 1);
    let end = Math.min(start + maxVisible - 1, totalPages);

    if (end - start < maxVisible - 1) {
      start = Math.max(end - maxVisible + 1, 1);
    }

    const pages = [];
    for (let i = start; i <= end; i++) {
      pages.push(i);
    }
    return pages;
  };

  if (loading) return <div className="text-center p-4">Loading...</div>;

  if (!records.length) {
    return (
      <div className="text-center p-4">
        <h2 className="text-xl font-bold">Backtest Records – {strategyName}</h2>
        <p>No data available for "{strategyName}".</p>
      </div>
    );
  }

  return (
    <div className="p-4">
      <h2 className="text-2xl font-bold mb-4">Backtest Records – {strategyName}</h2>
      <div className="overflow-x-auto rounded-lg shadow-lg">
        <table className="min-w-full border-collapse bg-white">
          <thead>
            <tr className="bg-black text-white">
              <th className="py-2 px-4 border">Date</th>
              <th className="py-2 px-4 border">Predicted Direction</th>
              <th className="py-2 px-4 border">Action</th>
              <th className="py-2 px-4 border">Buy Price</th>
              <th className="py-2 px-4 border">Sell Price</th>
              <th className="py-2 px-4 border">Balance</th>
              <th className="py-2 px-4 border">PNL</th>
              <th className="py-2 px-4 border">PNL Sum</th>
            </tr>
          </thead>
          <tbody>
            {currentData.map((row, idx) => (
              <tr
                key={idx}
                className={`${idx % 2 === 0 ? "bg-gray-50" : "bg-white"} hover:bg-yellow-100 transition`}
              >
                <td className="py-2 px-4 border">{row.datetime}</td>
                <td className="py-2 px-4 border">{row.predicted_direction}</td>
                <td className="py-2 px-4 border">{row.action}</td>
                <td className="py-2 px-4 border">{row.buy_price}</td>
                <td className="py-2 px-4 border">{row.sell_price}</td>
                <td className="py-2 px-4 border">{row.balance}</td>
                <td className="py-2 px-4 border">{row.pnl}</td>
                <td className="py-2 px-4 border">{row.pnl_sum}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination Controls */}
      <div className="flex justify-center items-center mt-4 gap-2">
        <button
          onClick={() => goToPage(currentPage - 1)}
          disabled={currentPage === 1}
          className="px-3 py-1 bg-gray-300 rounded disabled:opacity-50"
        >
          Prev
        </button>

        {getPageNumbers().map((page) => (
          <button
            key={page}
            onClick={() => goToPage(page)}
            className={`px-3 py-1 rounded ${
              currentPage === page ? "bg-blue-500 text-white" : "bg-gray-200"
            }`}
          >
            {page}
          </button>
        ))}

        <button
          onClick={() => goToPage(currentPage + 1)}
          disabled={currentPage === totalPages}
          className="px-3 py-1 bg-gray-300 rounded disabled:opacity-50"
        >
          Next
        </button>
      </div>
    </div>
  );
}
