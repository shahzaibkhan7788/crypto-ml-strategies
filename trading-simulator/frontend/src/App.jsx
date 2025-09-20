import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import Simulator from './pages/Simulator';
import StrategyDetail from './pages/StrategyDetail';
import UserAccounts from './pages/UserAccounts';
import BestStrategies from './pages/BestStrategies'; 
import BybitDemoDashboard from './pages/BybitDemoDashboard';
import StrategyDetailsDisplay from "./StrategyDetailsDisplay";


export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-100">
        <Navbar />
        <div className="container mx-auto p-4">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/simulator" element={<Simulator />} />
            <Route path="/strategies/:strategy_name" element={<StrategyDetail />} />
            <Route path="/accounts" element={<UserAccounts />} />
            <Route path="/model_score_board" element={<BestStrategies />} /> 
             <Route path="/live_trade_hub" element={<BybitDemoDashboard />} />
            <Route path="/strategy/:tableName" element={<StrategyDetailsDisplay />} />
          </Routes>
        </div>
      </div>
    </BrowserRouter>
  );
}
