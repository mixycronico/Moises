import React, { useState, useEffect } from 'react';
import { Routes, Route, Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import axios from 'axios';

// Componentes
import Loading from '../components/Loading';

const InvestorDashboard = ({ user, onLogout }) => {
  const [loading, setLoading] = useState(true);
  const [portfolioData, setPortfolioData] = useState(null);
  const [transactionsData, setTransactionsData] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    document.title = 'Panel de Inversor | Genesis';
    
    const fetchData = async () => {
      try {
        setLoading(true);
        
        // Obtener datos del portafolio
        const portfolioResponse = await axios.get('/api/investor/portfolio');
        setPortfolioData(portfolioResponse.data);
        
        // Obtener datos de transacciones
        const transactionsResponse = await axios.get('/api/investor/transactions');
        setTransactionsData(transactionsResponse.data);
      } catch (error) {
        console.error('Error al obtener datos:', error);
        if (error.response && error.response.status === 401) {
          // Si hay error de autenticación, cerrar sesión
          onLogout();
        }
      } finally {
        setLoading(false);
      }
    };
    
    fetchData();
  }, [onLogout]);

  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

  const handleLogout = () => {
    onLogout();
  };

  if (loading) {
    return <Loading message="Cargando datos del inversor..." />;
  }

  // Componente para la vista general (dashboard principal)
  const Overview = () => (
    <div className="dashboard-content">
      <h1>Bienvenido, {user.username}</h1>
      
      {/* Resumen de Balance */}
      <div className="balance-overview">
        <div className="balance-card card">
          <h3>Balance Total</h3>
          <p className="balance-amount">${portfolioData.balance.toLocaleString()}</p>
          <div className="balance-details">
            <div>
              <span>Invertido:</span>
              <span>${portfolioData.invested.toLocaleString()}</span>
            </div>
            <div>
              <span>Disponible:</span>
              <span>${portfolioData.available.toLocaleString()}</span>
            </div>
          </div>
        </div>
        
        <div className="performance-card card">
          <h3>Rendimiento</h3>
          <div className="performance-metrics">
            <div className="metric">
              <span className="metric-label">Diario</span>
              <span className="metric-value" style={{ color: portfolioData.performance.daily >= 0 ? 'var(--success-color)' : 'var(--danger-color)' }}>
                {portfolioData.performance.daily >= 0 ? '+' : ''}{portfolioData.performance.daily}%
              </span>
            </div>
            <div className="metric">
              <span className="metric-label">Semanal</span>
              <span className="metric-value" style={{ color: portfolioData.performance.weekly >= 0 ? 'var(--success-color)' : 'var(--danger-color)' }}>
                {portfolioData.performance.weekly >= 0 ? '+' : ''}{portfolioData.performance.weekly}%
              </span>
            </div>
            <div className="metric">
              <span className="metric-label">Mensual</span>
              <span className="metric-value" style={{ color: portfolioData.performance.monthly >= 0 ? 'var(--success-color)' : 'var(--danger-color)' }}>
                {portfolioData.performance.monthly >= 0 ? '+' : ''}{portfolioData.performance.monthly}%
              </span>
            </div>
            <div className="metric">
              <span className="metric-label">Anual</span>
              <span className="metric-value" style={{ color: portfolioData.performance.yearly >= 0 ? 'var(--success-color)' : 'var(--danger-color)' }}>
                {portfolioData.performance.yearly >= 0 ? '+' : ''}{portfolioData.performance.yearly}%
              </span>
            </div>
          </div>
        </div>
      </div>
      
      {/* Inversiones actuales */}
      <div className="investments-section">
        <h2>Tus Inversiones</h2>
        <div className="investments-list">
          {portfolioData.investments.map((investment) => (
            <div key={investment.id} className="investment-card card">
              <div className="investment-header">
                <h3>{investment.name}</h3>
                <span className="investment-symbol">{investment.symbol}</span>
              </div>
              <div className="investment-details">
                <div className="investment-amount">
                  <span>Cantidad:</span>
                  <span>{investment.amount} {investment.symbol}</span>
                </div>
                <div className="investment-value">
                  <span>Valor:</span>
                  <span>${investment.value_usd.toLocaleString()}</span>
                </div>
                <div className="investment-change">
                  <span>24h:</span>
                  <span style={{ color: investment.change_24h >= 0 ? 'var(--success-color)' : 'var(--danger-color)' }}>
                    {investment.change_24h >= 0 ? '+' : ''}{investment.change_24h}%
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
      
      {/* Transacciones recientes */}
      <div className="transactions-section">
        <h2>Transacciones Recientes</h2>
        <div className="transactions-list">
          <table className="transactions-table">
            <thead>
              <tr>
                <th>Activo</th>
                <th>Tipo</th>
                <th>Cantidad</th>
                <th>Precio</th>
                <th>Total</th>
                <th>Fecha</th>
                <th>Estado</th>
              </tr>
            </thead>
            <tbody>
              {transactionsData.slice(0, 5).map((transaction) => (
                <tr key={transaction.id}>
                  <td>
                    <div className="asset-info">
                      <span className="asset-name">{transaction.asset}</span>
                      <span className="asset-symbol">{transaction.symbol}</span>
                    </div>
                  </td>
                  <td>
                    <span className={`transaction-type ${transaction.type.toLowerCase()}`}>
                      {transaction.type}
                    </span>
                  </td>
                  <td>{transaction.amount} {transaction.symbol}</td>
                  <td>${transaction.price_usd.toLocaleString()}</td>
                  <td>${transaction.total_usd.toLocaleString()}</td>
                  <td>{new Date(transaction.date).toLocaleDateString()}</td>
                  <td>
                    <span className={`transaction-status ${transaction.status.toLowerCase()}`}>
                      {transaction.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          
          <div className="view-all-link">
            <Link to="/investor/transactions">Ver todas las transacciones</Link>
          </div>
        </div>
      </div>
    </div>
  );

  // Componente para la vista de portafolio
  const Portfolio = () => (
    <div className="dashboard-content">
      <h1>Tu Portafolio</h1>
      
      {/* Información detallada del portafolio */}
      <div className="portfolio-summary card">
        <div className="portfolio-header">
          <div>
            <h3>Balance Total</h3>
            <p className="balance-amount">${portfolioData.balance.toLocaleString()}</p>
          </div>
          <div className="portfolio-allocation">
            <div className="allocation-bar">
              <div 
                className="allocation-filled"
                style={{ width: `${(portfolioData.invested / portfolioData.balance) * 100}%` }}
              ></div>
            </div>
            <div className="allocation-details">
              <div>
                <span>Invertido:</span>
                <span>${portfolioData.invested.toLocaleString()} ({((portfolioData.invested / portfolioData.balance) * 100).toFixed(1)}%)</span>
              </div>
              <div>
                <span>Disponible:</span>
                <span>${portfolioData.available.toLocaleString()} ({((portfolioData.available / portfolioData.balance) * 100).toFixed(1)}%)</span>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      {/* Lista completa de inversiones */}
      <div className="investments-full-list">
        <h2>Detalle de Inversiones</h2>
        <table className="investments-table">
          <thead>
            <tr>
              <th>Activo</th>
              <th>Cantidad</th>
              <th>Precio Actual</th>
              <th>Valor Total</th>
              <th>Cambio 24h</th>
              <th>Asignación</th>
            </tr>
          </thead>
          <tbody>
            {portfolioData.investments.map((investment) => (
              <tr key={investment.id}>
                <td>
                  <div className="asset-info">
                    <span className="asset-name">{investment.name}</span>
                    <span className="asset-symbol">{investment.symbol}</span>
                  </div>
                </td>
                <td>{investment.amount} {investment.symbol}</td>
                <td>${(investment.value_usd / investment.amount).toFixed(2)}</td>
                <td>${investment.value_usd.toLocaleString()}</td>
                <td style={{ color: investment.change_24h >= 0 ? 'var(--success-color)' : 'var(--danger-color)' }}>
                  {investment.change_24h >= 0 ? '+' : ''}{investment.change_24h}%
                </td>
                <td>
                  <div className="allocation-mini-bar">
                    <div 
                      className="allocation-mini-filled"
                      style={{ 
                        width: `${(investment.value_usd / portfolioData.invested) * 100}%`,
                        backgroundColor: investment.change_24h >= 0 ? 'var(--success-color)' : 'var(--danger-color)'
                      }}
                    ></div>
                  </div>
                  <span className="allocation-percent">
                    {((investment.value_usd / portfolioData.invested) * 100).toFixed(1)}%
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );

  // Componente para la vista de transacciones
  const Transactions = () => (
    <div className="dashboard-content">
      <h1>Historial de Transacciones</h1>
      
      {/* Filtros de transacciones */}
      <div className="transactions-filters card">
        <div className="filter-group">
          <label>Tipo:</label>
          <select>
            <option value="all">Todos</option>
            <option value="buy">Compra</option>
            <option value="sell">Venta</option>
          </select>
        </div>
        <div className="filter-group">
          <label>Estado:</label>
          <select>
            <option value="all">Todos</option>
            <option value="completed">Completado</option>
            <option value="pending">Pendiente</option>
            <option value="failed">Fallido</option>
          </select>
        </div>
        <div className="filter-group">
          <label>Ordenar por:</label>
          <select>
            <option value="date-desc">Fecha (Reciente)</option>
            <option value="date-asc">Fecha (Antiguo)</option>
            <option value="amount-desc">Monto (Mayor)</option>
            <option value="amount-asc">Monto (Menor)</option>
          </select>
        </div>
      </div>
      
      {/* Tabla completa de transacciones */}
      <div className="transactions-full-list">
        <table className="transactions-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Activo</th>
              <th>Tipo</th>
              <th>Cantidad</th>
              <th>Precio</th>
              <th>Total</th>
              <th>Fecha</th>
              <th>Estado</th>
            </tr>
          </thead>
          <tbody>
            {transactionsData.map((transaction) => (
              <tr key={transaction.id}>
                <td>#{transaction.id}</td>
                <td>
                  <div className="asset-info">
                    <span className="asset-name">{transaction.asset}</span>
                    <span className="asset-symbol">{transaction.symbol}</span>
                  </div>
                </td>
                <td>
                  <span className={`transaction-type ${transaction.type.toLowerCase()}`}>
                    {transaction.type}
                  </span>
                </td>
                <td>{transaction.amount} {transaction.symbol}</td>
                <td>${transaction.price_usd.toLocaleString()}</td>
                <td>${transaction.total_usd.toLocaleString()}</td>
                <td>{new Date(transaction.date).toLocaleString()}</td>
                <td>
                  <span className={`transaction-status ${transaction.status.toLowerCase()}`}>
                    {transaction.status}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );

  return (
    <div className="dashboard-container">
      {/* Sidebar para navegación */}
      <aside className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <div className="logo">
            <svg 
              width="28" 
              height="28" 
              viewBox="0 0 24 24" 
              fill="none" 
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2ZM12 20C7.59 20 4 16.41 4 12C4 7.59 7.59 4 12 4C16.41 4 20 7.59 20 12C20 16.41 16.41 20 12 20Z"
                fill="url(#sidebar-gradient)"
              />
              <path
                d="M12 11C13.1046 11 14 10.1046 14 9C14 7.89543 13.1046 7 12 7C10.8954 7 10 7.89543 10 9C10 10.1046 10.8954 11 12 11Z"
                fill="url(#sidebar-gradient)"
              />
              <path
                d="M12 13C9.79 13 8 14.79 8 17H16C16 14.79 14.21 13 12 13Z"
                fill="url(#sidebar-gradient)"
              />
              <defs>
                <linearGradient id="sidebar-gradient" x1="2" y1="12" x2="22" y2="12" gradientUnits="userSpaceOnUse">
                  <stop stopColor="#05b2dc" />
                  <stop offset="1" stopColor="#8a2be2" />
                </linearGradient>
              </defs>
            </svg>
            <span>GENESIS</span>
          </div>
          <button className="sidebar-close" onClick={toggleSidebar}>
            <svg width="24" height="24" fill="currentColor" viewBox="0 0 24 24">
              <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12 19 6.41z"/>
            </svg>
          </button>
        </div>
        
        <div className="sidebar-user">
          <div className="user-avatar">
            {user.username.charAt(0).toUpperCase()}
          </div>
          <div className="user-info">
            <span className="user-name">{user.username}</span>
            <span className="user-role">Inversor</span>
          </div>
        </div>
        
        <nav className="sidebar-nav">
          <Link to="/investor" className="nav-item" onClick={() => window.innerWidth < 768 && setSidebarOpen(false)}>
            <svg width="24" height="24" fill="currentColor" viewBox="0 0 24 24">
              <path d="M3 13h8V3H3v10zm0 8h8v-6H3v6zm10 0h8V11h-8v10zm0-18v6h8V3h-8z"/>
            </svg>
            <span>Vista General</span>
          </Link>
          <Link to="/investor/portfolio" className="nav-item" onClick={() => window.innerWidth < 768 && setSidebarOpen(false)}>
            <svg width="24" height="24" fill="currentColor" viewBox="0 0 24 24">
              <path d="M21 18v1c0 1.1-.9 2-2 2H5c-1.1 0-2-.9-2-2V5c0-1.1.9-2 2-2h14c1.1 0 2 .9 2 2v1h-9c-1.1 0-2 .9-2 2v8c0 1.1.9 2 2 2h9zm-9-2h10V8H12v8z"/>
            </svg>
            <span>Portafolio</span>
          </Link>
          <Link to="/investor/transactions" className="nav-item" onClick={() => window.innerWidth < 768 && setSidebarOpen(false)}>
            <svg width="24" height="24" fill="currentColor" viewBox="0 0 24 24">
              <path d="M19 3h-4.18C14.4 1.84 13.3 1 12 1s-2.4.84-2.82 2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-7 0c.55 0 1 .45 1 1s-.45 1-1 1-1-.45-1-1 .45-1 1-1zm2 14H7v-2h7v2zm3-4H7v-2h10v2zm0-4H7V7h10v2z"/>
            </svg>
            <span>Transacciones</span>
          </Link>
        </nav>
        
        <div className="sidebar-footer">
          <button className="logout-button" onClick={handleLogout}>
            <svg width="24" height="24" fill="currentColor" viewBox="0 0 24 24">
              <path d="M17 7l-1.41 1.41L18.17 11H8v2h10.17l-2.58 2.58L17 17l5-5zM4 5h8V3H4c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h8v-2H4V5z"/>
            </svg>
            <span>Cerrar Sesión</span>
          </button>
        </div>
      </aside>
      
      {/* Contenido principal */}
      <main className="dashboard-main">
        {/* Header superior con botón de menú */}
        <header className="dashboard-header">
          <button className="menu-toggle" onClick={toggleSidebar}>
            <svg width="24" height="24" fill="currentColor" viewBox="0 0 24 24">
              <path d="M3 18h18v-2H3v2zm0-5h18v-2H3v2zm0-7v2h18V6H3z"/>
            </svg>
          </button>
          <div className="header-title">
            <h1>Panel de Inversor</h1>
          </div>
          <div className="header-actions">
            <div className="user-info-mini" onClick={() => navigate('/investor/profile')}>
              <div className="user-avatar-mini">
                {user.username.charAt(0).toUpperCase()}
              </div>
              <span className="user-name-mini">{user.username}</span>
            </div>
          </div>
        </header>
        
        {/* Rutas para el contenido */}
        <div className="dashboard-content-container">
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/portfolio" element={<Portfolio />} />
            <Route path="/transactions" element={<Transactions />} />
          </Routes>
        </div>
      </main>
      
      {/* Overlay para cerrar sidebar en móvil */}
      {sidebarOpen && (
        <div 
          className="sidebar-overlay" 
          onClick={toggleSidebar}
        ></div>
      )}
      
      {/* Estilos específicos para el dashboard */}
      <style jsx="true">{`
        .dashboard-container {
          display: flex;
          min-height: 100vh;
          background-color: var(--background-color);
          color: var(--text-color);
          position: relative;
        }
        
        /* Sidebar */
        .sidebar {
          width: 260px;
          background-color: var(--card-background);
          border-right: 1px solid rgba(255, 255, 255, 0.1);
          display: flex;
          flex-direction: column;
          transition: transform 0.3s ease;
          position: fixed;
          top: 0;
          left: 0;
          bottom: 0;
          z-index: 1000;
        }
        
        @media (max-width: 768px) {
          .sidebar {
            transform: translateX(-100%);
          }
          
          .sidebar.open {
            transform: translateX(0);
          }
        }
        
        .sidebar-header {
          padding: 1.5rem;
          display: flex;
          justify-content: space-between;
          align-items: center;
          border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .sidebar-header .logo {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          font-family: var(--font-title);
          font-weight: bold;
          color: var(--primary-color);
          font-size: 1.25rem;
        }
        
        .sidebar-close {
          display: none;
          background: transparent;
          border: none;
          color: var(--text-secondary);
          cursor: pointer;
        }
        
        @media (max-width: 768px) {
          .sidebar-close {
            display: block;
          }
        }
        
        .sidebar-user {
          padding: 1.5rem;
          display: flex;
          align-items: center;
          gap: 1rem;
          border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .user-avatar {
          width: 40px;
          height: 40px;
          border-radius: 50%;
          background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
          display: flex;
          align-items: center;
          justify-content: center;
          font-weight: bold;
          box-shadow: var(--glow-primary);
        }
        
        .user-info {
          display: flex;
          flex-direction: column;
        }
        
        .user-name {
          font-weight: bold;
        }
        
        .user-role {
          font-size: 0.8rem;
          color: var(--text-secondary);
        }
        
        .sidebar-nav {
          flex: 1;
          padding: 1.5rem 0;
          display: flex;
          flex-direction: column;
        }
        
        .nav-item {
          display: flex;
          align-items: center;
          padding: 0.75rem 1.5rem;
          color: var(--text-secondary);
          text-decoration: none;
          transition: all 0.3s ease;
          gap: 0.75rem;
        }
        
        .nav-item:hover, .nav-item.active {
          background-color: rgba(5, 178, 220, 0.1);
          color: var(--primary-color);
        }
        
        .sidebar-footer {
          padding: 1.5rem;
          border-top: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .logout-button {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          color: var(--text-secondary);
          background: transparent;
          border: none;
          padding: 0.75rem;
          width: 100%;
          cursor: pointer;
          transition: all 0.3s ease;
          font-family: var(--font-body);
          font-size: 1rem;
        }
        
        .logout-button:hover {
          color: var(--danger-color);
        }
        
        /* Main content */
        .dashboard-main {
          flex: 1;
          margin-left: 260px;
          width: calc(100% - 260px);
        }
        
        @media (max-width: 768px) {
          .dashboard-main {
            margin-left: 0;
            width: 100%;
          }
        }
        
        .dashboard-header {
          padding: 1rem 1.5rem;
          display: flex;
          align-items: center;
          border-bottom: 1px solid rgba(255, 255, 255, 0.1);
          background-color: var(--card-background);
        }
        
        .menu-toggle {
          display: none;
          background: transparent;
          border: none;
          color: var(--text-color);
          margin-right: 1rem;
          cursor: pointer;
        }
        
        @media (max-width: 768px) {
          .menu-toggle {
            display: block;
          }
        }
        
        .header-title {
          flex: 1;
        }
        
        .header-title h1 {
          margin: 0;
          font-size: 1.25rem;
        }
        
        .user-info-mini {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          cursor: pointer;
        }
        
        .user-avatar-mini {
          width: 32px;
          height: 32px;
          border-radius: 50%;
          background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
          display: flex;
          align-items: center;
          justify-content: center;
          font-weight: bold;
          font-size: 0.8rem;
        }
        
        /* Overlay */
        .sidebar-overlay {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background-color: rgba(0, 0, 0, 0.5);
          z-index: 999;
          display: none;
        }
        
        @media (max-width: 768px) {
          .sidebar-overlay {
            display: block;
          }
        }
        
        /* Dashboard content */
        .dashboard-content-container {
          padding: 1.5rem;
          overflow-y: auto;
          max-height: calc(100vh - 64px); /* Altura de la ventana menos altura del header */
        }
        
        .dashboard-content h1 {
          margin-bottom: 1.5rem;
        }
        
        /* Balance overview */
        .balance-overview {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
          gap: 1.5rem;
          margin-bottom: 2rem;
        }
        
        .balance-card, .performance-card {
          padding: 1.5rem;
        }
        
        .balance-card h3, .performance-card h3 {
          margin-bottom: 1rem;
          color: var(--text-secondary);
          font-size: 1rem;
        }
        
        .balance-amount {
          font-size: 2rem;
          font-weight: bold;
          margin-bottom: 1rem;
          color: var(--primary-color);
        }
        
        .balance-details {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }
        
        .balance-details div {
          display: flex;
          justify-content: space-between;
        }
        
        .performance-metrics {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 1rem;
        }
        
        .metric {
          display: flex;
          flex-direction: column;
          gap: 0.25rem;
        }
        
        .metric-label {
          color: var(--text-secondary);
          font-size: 0.9rem;
        }
        
        .metric-value {
          font-size: 1.25rem;
          font-weight: bold;
        }
        
        /* Investments */
        .investments-section {
          margin-bottom: 2rem;
        }
        
        .investments-section h2 {
          margin-bottom: 1rem;
        }
        
        .investments-list {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
          gap: 1.5rem;
        }
        
        .investment-card {
          padding: 1.5rem;
        }
        
        .investment-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 1rem;
        }
        
        .investment-header h3 {
          margin-bottom: 0;
        }
        
        .investment-symbol {
          background-color: rgba(5, 178, 220, 0.2);
          color: var(--primary-color);
          padding: 0.25rem 0.5rem;
          border-radius: 4px;
          font-family: var(--font-title);
          font-size: 0.8rem;
        }
        
        .investment-details {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }
        
        .investment-details div {
          display: flex;
          justify-content: space-between;
          align-items: center;
        }
        
        /* Transactions */
        .transactions-section {
          margin-bottom: 2rem;
        }
        
        .transactions-section h2 {
          margin-bottom: 1rem;
        }
        
        .transactions-table {
          width: 100%;
          border-collapse: collapse;
          margin-bottom: 1rem;
        }
        
        .transactions-table th {
          text-align: left;
          padding: 0.75rem;
          border-bottom: 1px solid rgba(255, 255, 255, 0.1);
          color: var(--text-secondary);
        }
        
        .transactions-table td {
          padding: 0.75rem;
          border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .asset-info {
          display: flex;
          flex-direction: column;
        }
        
        .asset-symbol {
          font-size: 0.8rem;
          color: var(--text-secondary);
        }
        
        .transaction-type {
          padding: 0.25rem 0.5rem;
          border-radius: 4px;
          font-size: 0.8rem;
          text-transform: uppercase;
        }
        
        .transaction-type.buy {
          background-color: rgba(76, 175, 80, 0.2);
          color: var(--success-color);
        }
        
        .transaction-type.sell {
          background-color: rgba(255, 82, 82, 0.2);
          color: var(--danger-color);
        }
        
        .transaction-status {
          padding: 0.25rem 0.5rem;
          border-radius: 4px;
          font-size: 0.8rem;
        }
        
        .transaction-status.completed {
          background-color: rgba(76, 175, 80, 0.2);
          color: var(--success-color);
        }
        
        .transaction-status.pending {
          background-color: rgba(255, 193, 7, 0.2);
          color: var(--warning-color);
        }
        
        .transaction-status.failed {
          background-color: rgba(255, 82, 82, 0.2);
          color: var(--danger-color);
        }
        
        .view-all-link {
          text-align: center;
          margin-top: 1rem;
        }
        
        .view-all-link a {
          color: var(--primary-color);
          text-decoration: none;
        }
        
        /* Portfolio page */
        .portfolio-summary {
          margin-bottom: 2rem;
        }
        
        .portfolio-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 2rem;
        }
        
        .portfolio-allocation {
          flex: 1;
        }
        
        .allocation-bar {
          height: 8px;
          background-color: rgba(255, 255, 255, 0.1);
          border-radius: 4px;
          overflow: hidden;
          margin-bottom: 0.5rem;
        }
        
        .allocation-filled {
          height: 100%;
          background: linear-gradient(to right, var(--primary-color), var(--secondary-color));
        }
        
        .allocation-details {
          display: flex;
          justify-content: space-between;
        }
        
        .investments-full-list {
          margin-bottom: 2rem;
        }
        
        .investments-table {
          width: 100%;
          border-collapse: collapse;
        }
        
        .investments-table th {
          text-align: left;
          padding: 0.75rem;
          border-bottom: 1px solid rgba(255, 255, 255, 0.1);
          color: var(--text-secondary);
        }
        
        .investments-table td {
          padding: 0.75rem;
          border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .allocation-mini-bar {
          width: 100px;
          height: 6px;
          background-color: rgba(255, 255, 255, 0.1);
          border-radius: 3px;
          overflow: hidden;
          display: inline-block;
          margin-right: 0.5rem;
        }
        
        .allocation-mini-filled {
          height: 100%;
        }
        
        .allocation-percent {
          font-size: 0.8rem;
          color: var(--text-secondary);
        }
        
        /* Transactions page */
        .transactions-filters {
          display: flex;
          gap: 1.5rem;
          margin-bottom: 1.5rem;
          padding: 1rem;
          flex-wrap: wrap;
        }
        
        .filter-group {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }
        
        .filter-group label {
          color: var(--text-secondary);
          font-size: 0.9rem;
        }
        
        .filter-group select {
          padding: 0.5rem;
          background-color: rgba(10, 14, 23, 0.5);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: var(--border-radius-sm);
          color: var(--text-color);
          font-family: var(--font-body);
        }
        
        .transactions-full-list {
          overflow-x: auto;
        }
      `}</style>
    </div>
  );
};

export default InvestorDashboard;