import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';  // This must match your CSS file path
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);