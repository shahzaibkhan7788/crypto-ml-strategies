import React from "react";
import { useNavigate } from "react-router-dom";

export default function Dashboard() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-100 via-gray-50 to-gray-100 p-6 flex flex-col justify-between">
      <div className="max-w-7xl mx-auto flex-1">
        {/* Title */}
        <h1 className="text-4xl md:text-5xl font-extrabold text-center text-gray-800 mb-2">
          Trading Dashboard
        </h1>
        <p className="text-center text-gray-600 mb-8 text-base md:text-lg">
          Explore, simulate, and optimize your trading strategies with advanced models and real-time insights.
        </p>

        {/* Feature Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-10">
          
          {/* Simulator */}
          <div className="bg-gradient-to-br from-green-50 via-green-100 to-green-50 rounded-2xl shadow-lg p-6 flex flex-col justify-between transform hover:-translate-y-1 hover:shadow-xl transition min-h-[280px]">
            <div>
              <h2 className="text-xl md:text-2xl font-bold text-green-700 mb-2">Simulator</h2>
              <p className="text-gray-700 text-sm md:text-base mb-4">
                Test multiple trading strategies with hundreds of indicators. Visualize performance in real-time and optimize parameters.
              </p>
            </div>
            <button
              onClick={() => navigate("/simulator")}
              className="px-5 py-3 text-sm md:text-lg bg-gradient-to-r from-green-500 to-green-600 text-white font-semibold rounded-lg shadow hover:from-green-600 hover:to-green-700 transition"
            >
              Go to Simulator
            </button>
          </div>

          {/* Register Strategies */}
          <div className="bg-gradient-to-br from-blue-50 via-blue-100 to-blue-50 rounded-2xl shadow-lg p-6 flex flex-col justify-between transform hover:-translate-y-1 hover:shadow-xl transition min-h-[280px]">
            <div>
              <h2 className="text-xl md:text-2xl font-bold text-blue-700 mb-2">Register Strategies</h2>
              <p className="text-gray-700 text-sm md:text-base mb-4">
                Add your own strategies or select from prebuilt options. Monitor and fine-tune their performance for better results.
              </p>
            </div>
            <button
              onClick={() => navigate("/accounts?tab=register")}
              className="px-5 py-3 text-sm md:text-lg bg-gradient-to-r from-blue-500 to-blue-600 text-white font-semibold rounded-lg shadow hover:from-blue-600 hover:to-blue-700 transition"
            >
              Register Now
            </button>
          </div>

          {/* Open & Closed Trades */}
          <div className="bg-gradient-to-br from-purple-50 via-purple-100 to-purple-50 rounded-2xl shadow-lg p-6 flex flex-col justify-between transform hover:-translate-y-1 hover:shadow-xl transition min-h-[280px]">
            <div>
              <h2 className="text-xl md:text-2xl font-bold text-purple-700 mb-2">Open & Closed Trades</h2>
              <p className="text-gray-700 text-sm md:text-base mb-4">
                Track and analyze all trades with comprehensive metrics. Make informed decisions using historical and real-time performance data.
              </p>
            </div>
            <button
              onClick={() => navigate("/live_trade_hub")}
              className="px-5 py-3 text-sm md:text-lg bg-gradient-to-r from-purple-500 to-purple-600 text-white font-semibold rounded-lg shadow hover:from-purple-600 hover:to-purple-700 transition"
            >
              View Trades
            </button>
          </div>

          {/* Models & Performance */}
          <div className="bg-gradient-to-br from-red-50 via-red-100 to-red-50 rounded-2xl shadow-lg p-6 flex flex-col justify-between transform hover:-translate-y-1 hover:shadow-xl transition min-h-[280px]">
            <div>
              <h2 className="text-xl md:text-2xl font-bold text-red-700 mb-2">Models & Performance</h2>
              <p className="text-gray-700 text-sm md:text-base mb-4">
                Compare strategy models, evaluate their effectiveness, and choose the best-performing ones with backend computations.
              </p>
            </div>
            <button
              onClick={() => navigate("/model_score_board")}
              className="px-5 py-3 text-sm md:text-lg bg-gradient-to-r from-red-500 to-red-600 text-white font-semibold rounded-lg shadow hover:from-red-600 hover:to-red-700 transition"
            >
              Explore Models
            </button>
          </div>

        </div>
      </div>

      <div className="min-h-screen bg-gradient-to-br from-gray-100 via-gray-50 to-gray-100 p-6 flex flex-col">
  <div className="max-w-7xl mx-auto flex-1">
   
    {/* Feature Cards */}
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6"> {/* smaller mb */}
      {/* ...cards here... */}
    </div>

    {/* Extra Description (moved here just after cards) */}
    <div className="bg-gradient-to-r from-gray-50 via-gray-100 to-gray-50 p-6 rounded-2xl shadow-lg text-center">
      <h3 className="text-2xl md:text-3xl font-bold text-gray-800 mb-1">
        Maximize Your Trading Potential
      </h3>
      <p className="text-gray-700 text-sm md:text-base">
        Leverage hundreds of indicators, advanced simulations, and real-time strategy evaluation to select the most profitable trades, optimize performance, and make informed decisions across multiple markets.
      </p>
    </div>
  </div>
</div>

    </div>
  );
}
