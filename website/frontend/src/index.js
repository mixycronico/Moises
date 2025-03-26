import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

// Importar estilos
import './styles/index.css';
import './styles/login.css';
import './styles/dashboard.css';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);