import { useParams } from 'react-router-dom';
import { useEffect, useState } from 'react';
import PnlGraph from '../components/PnlGraph';
import BacktestTable from '../components/BacktestTable';
import StrategyStatus from '../components/StrategyStatus';
import StrategyIndicators from '../components/StrategyIndicators'; // ✅ New import
import { useNavigate } from "react-router-dom";


export default function StrategyDetail() {
  const { strategy_name } = useParams();
  const [strategyData, setStrategyData] = useState([]);

  useEffect(() => {
    const fetchData = async () => {
      console.log(`🔄 Sending request for strategy: ${strategy_name}`);
      try {
        const res = await fetch(`http://localhost:3001/api/strategy/${strategy_name}/data`);
        if (!res.ok) {
          throw new Error(`HTTP error! status: ${res.status}`);
        }

        const response = await res.json();
        const finalData = Array.isArray(response)
          ? response
          : response.data || [];

        console.log("📦 Data used by PnlGraph:", finalData);
        setStrategyData(finalData);
      } catch (err) {
        console.error('❌ Error fetching strategy data:', err);
      }
    };

    fetchData();
  }, [strategy_name]);

  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold text-slate-800 mb-6">
        {strategy_name}
      </h1>

      {/* 🆕 Strategy Status component */}
      <div className="bg-white p-4 rounded shadow mb-8">
        <StrategyStatus strategyName={strategy_name} />
      </div>

      {/* 🆕 Strategy Indicators component */}
      <div className="bg-white p-4 rounded shadow mb-8">
        <StrategyIndicators strategyName={strategy_name} />
      </div>

      {/* 📈 Graph */}
      <div className="bg-white p-4 rounded shadow mb-8">
        <PnlGraph data={strategyData} />
      </div>

      {/* 📊 Table */}
      <div className="bg-white p-4 rounded shadow">
        <BacktestTable strategyName={strategy_name} />
      </div>
    </div>
  );
}
