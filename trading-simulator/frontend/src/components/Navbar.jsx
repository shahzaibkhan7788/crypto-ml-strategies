import React from 'react';
import { Link } from 'react-router-dom';
import logo from './3-19.jpg'; // Adjust path if needed

const Header = () => {
  return (
    <header className="bg-gradient-to-r from-emerald-700 via-slate-900 to-emerald-700 text-white shadow-lg">
      <div className="max-w-7xl mx-auto px-4 py-5 flex items-center justify-between">
        
        {/* Logo + Brand Name */}
        <div className="flex items-center gap-4">
          <img
            src={logo}
            alt="TradeTactix Logo"
            className="h-12 w-12 rounded-full shadow-md border-2 border-emerald-300 object-cover"
          />
          <span className="text-3xl font-extrabold tracking-wide text-emerald-300 select-none cursor-default">
            TradeTactix
          </span>
        </div>

        {/* Navigation Links */}
        <nav className="flex gap-10 text-lg font-semibold tracking-wide">
          <Link to="/" className="hover:text-emerald-300 transition duration-300 ease-in-out">
            Dashboard
          </Link>
          <Link to="/simulator" className="hover:text-emerald-300 transition duration-300 ease-in-out">
            Simulator
          </Link>
          <Link to="/accounts" className="hover:text-emerald-300 transition duration-300 ease-in-out">
            User Accounts
          </Link>
          <Link to="/model_score_board" className="hover:text-emerald-300 transition duration-300 ease-in-out">
            Models Scoreboard
          </Link>
          <Link to="/live_trade_hub" className="hover:text-emerald-300 transition duration-300 ease-in-out">
            Live Trade Hub
          </Link>
        </nav>

        {/* Dark Mode Toggle */}
        <div className="flex items-center">
          <button
            aria-label="Toggle dark mode"
            className="p-3 rounded-full hover:bg-emerald-600 transition duration-300"
            title="Toggle dark mode"
          >
          </button>
        </div>
      </div>
    </header>
  );
};

export default Header;
