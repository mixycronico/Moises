import React, { useState, useEffect } from 'react';
import { Routes, Route, Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import axios from 'axios';

// Componentes
import Loading from '../components/Loading';

const AdminDashboard = ({ user, onLogout }) => {
  const [loading, setLoading] = useState(true);
  const [investorsData, setInvestorsData] = useState(null);
  const [systemStatus, setSystemStatus] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    document.title = 'Panel de Administrador | Genesis';
    
    const fetchData = async () => {
      try {
        setLoading(true);
        
        // Obtener lista de inversores
        const investorsResponse = await axios.get('/api/admin/investors');
        setInvestorsData(investorsResponse.data);
        
        // Obtener estado del sistema
        const statusResponse = await axios.get('/api/admin/system-status');
        setSystemStatus(statusResponse.data);
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
    return <Loading message="Cargando panel de administrador..." />;
  }

  // Componente para la vista general (dashboard principal)
  const Overview = () => (
    <div className="dashboard-content">
      <h1>Panel de Administrador</h1>
      
      {/* Estado del sistema */}
      <div className="system-status-cards">
        <div className="status-card card">
          <h3>Estado del Sistema</h3>
          <div className={`status-indicator ${systemStatus.system.status.toLowerCase()}`}>
            {systemStatus.system.status === 'operational' ? 'Operativo' : 'Con problemas'}
          </div>
          <div className="status-metrics">
            <div className="metric">
              <span className="metric-label">Memoria</span>
              <div className="progress-bar">
                <div 
                  className="progress-fill" 
                  style={{ 
                    width: `${systemStatus.system.memory_usage}%`,
                    backgroundColor: systemStatus.system.memory_usage > 80 ? 'var(--danger-color)' : 'var(--primary-color)'
                  }}
                ></div>
              </div>
              <span className="metric-value">{systemStatus.system.memory_usage}%</span>
            </div>
            <div className="metric">
              <span className="metric-label">CPU</span>
              <div className="progress-bar">
                <div 
                  className="progress-fill" 
                  style={{ 
                    width: `${systemStatus.system.cpu_usage}%`,
                    backgroundColor: systemStatus.system.cpu_usage > 80 ? 'var(--danger-color)' : 'var(--primary-color)'
                  }}
                ></div>
              </div>
              <span className="metric-value">{systemStatus.system.cpu_usage}%</span>
            </div>
          </div>
          <div className="status-details">
            <div><span>Tiempo activo:</span> <span>{systemStatus.system.uptime}</span></div>
            <div><span>Usuarios activos:</span> <span>{systemStatus.system.active_users}</span></div>
          </div>
        </div>
        
        <div className="status-card card">
          <h3>Estado del Mercado</h3>
          <div className={`status-indicator ${systemStatus.market.status === 'open' ? 'operational' : 'issue'}`}>
            {systemStatus.market.status === 'open' ? 'Abierto' : 'Cerrado'}
          </div>
          <div className="status-details">
            <div><span>Exchanges conectados:</span> <span>{systemStatus.market.connected_exchanges}</span></div>
            <div><span>Pares activos:</span> <span>{systemStatus.market.active_pairs}</span></div>
            <div><span>Latencia de datos:</span> <span>{systemStatus.market.data_latency}s</span></div>
          </div>
        </div>
        
        <div className="status-card card">
          <h3>Operaciones</h3>
          <div className="status-details">
            <div><span>Transacciones hoy:</span> <span>{systemStatus.operations.transactions_today}</span></div>
            <div><span>Operaciones pendientes:</span> <span>{systemStatus.operations.pending_operations}</span></div>
            <div><span>Tasa de éxito:</span> <span>{systemStatus.operations.success_rate}%</span></div>
          </div>
        </div>
      </div>
      
      {/* Resumen de inversores */}
      <div className="investors-summary-section">
        <div className="section-header">
          <h2>Resumen de Inversores</h2>
          <Link to="/admin/investors" className="view-all-link">Ver todos</Link>
        </div>
        
        <div className="investors-metrics card">
          <div className="investor-metric">
            <span className="investor-metric-value">{investorsData.length}</span>
            <span className="investor-metric-label">Total Inversores</span>
          </div>
          <div className="investor-metric">
            <span className="investor-metric-value">
              {investorsData.filter(investor => investor.status === 'active').length}
            </span>
            <span className="investor-metric-label">Activos</span>
          </div>
          <div className="investor-metric">
            <span className="investor-metric-value">
              ${investorsData.reduce((sum, investor) => sum + investor.balance, 0).toLocaleString()}
            </span>
            <span className="investor-metric-label">Capital Total</span>
          </div>
          <div className="investor-metric">
            <span className="investor-metric-value">
              ${investorsData.reduce((sum, investor) => sum + investor.invested, 0).toLocaleString()}
            </span>
            <span className="investor-metric-label">Invertido Total</span>
          </div>
        </div>
        
        <div className="investors-list">
          <table className="investors-table">
            <thead>
              <tr>
                <th>Usuario</th>
                <th>Email</th>
                <th>Balance</th>
                <th>Invertido</th>
                <th>Rendimiento</th>
                <th>Estado</th>
                <th>Acciones</th>
              </tr>
            </thead>
            <tbody>
              {investorsData.slice(0, 5).map((investor) => (
                <tr key={investor.id}>
                  <td>{investor.username}</td>
                  <td>{investor.email}</td>
                  <td>${investor.balance.toLocaleString()}</td>
                  <td>${investor.invested.toLocaleString()}</td>
                  <td style={{ color: investor.performance >= 0 ? 'var(--success-color)' : 'var(--danger-color)' }}>
                    {investor.performance >= 0 ? '+' : ''}{investor.performance}%
                  </td>
                  <td>
                    <span className={`status-badge ${investor.status}`}>
                      {investor.status === 'active' ? 'Activo' : 'Inactivo'}
                    </span>
                  </td>
                  <td>
                    <button className="action-button">
                      <svg width="18" height="18" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M12 8c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2zm0 2c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm0 6c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2z"/>
                      </svg>
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
  
  // Componente para la gestión de inversores
  const InvestorsManagement = () => (
    <div className="dashboard-content">
      <h1>Gestión de Inversores</h1>
      
      {/* Filtros de inversores */}
      <div className="investor-filters card">
        <div className="search-bar">
          <svg width="20" height="20" fill="currentColor" viewBox="0 0 24 24">
            <path d="M15.5 14h-.79l-.28-.27C15.41 12.59 16 11.11 16 9.5 16 5.91 13.09 3 9.5 3S3 5.91 3 9.5 5.91 16 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"/>
          </svg>
          <input 
            type="text" 
            placeholder="Buscar inversor..." 
            className="search-input"
          />
        </div>
        <div className="filter-options">
          <div className="filter-group">
            <label>Estado:</label>
            <select>
              <option value="all">Todos</option>
              <option value="active">Activos</option>
              <option value="inactive">Inactivos</option>
            </select>
          </div>
          <div className="filter-group">
            <label>Ordenar por:</label>
            <select>
              <option value="username">Usuario (A-Z)</option>
              <option value="balance-desc">Balance (Mayor)</option>
              <option value="balance-asc">Balance (Menor)</option>
              <option value="performance-desc">Rendimiento (Mayor)</option>
              <option value="performance-asc">Rendimiento (Menor)</option>
            </select>
          </div>
        </div>
      </div>
      
      {/* Tabla completa de inversores */}
      <div className="investors-full-list card">
        <table className="investors-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Usuario</th>
              <th>Email</th>
              <th>Balance</th>
              <th>Invertido</th>
              <th>Disponible</th>
              <th>Rendimiento</th>
              <th>Estado</th>
              <th>Acciones</th>
            </tr>
          </thead>
          <tbody>
            {investorsData.map((investor) => (
              <tr key={investor.id}>
                <td>#{investor.id}</td>
                <td>{investor.username}</td>
                <td>{investor.email}</td>
                <td>${investor.balance.toLocaleString()}</td>
                <td>${investor.invested.toLocaleString()}</td>
                <td>${(investor.balance - investor.invested).toLocaleString()}</td>
                <td style={{ color: investor.performance >= 0 ? 'var(--success-color)' : 'var(--danger-color)' }}>
                  {investor.performance >= 0 ? '+' : ''}{investor.performance}%
                </td>
                <td>
                  <span className={`status-badge ${investor.status}`}>
                    {investor.status === 'active' ? 'Activo' : 'Inactivo'}
                  </span>
                </td>
                <td>
                  <div className="action-buttons">
                    <button className="action-button view">
                      <svg width="18" height="18" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M12 4.5C7 4.5 2.73 7.61 1 12c1.73 4.39 6 7.5 11 7.5s9.27-3.11 11-7.5c-1.73-4.39-6-7.5-11-7.5zM12 17c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5zm0-8c-1.66 0-3 1.34-3 3s1.34 3 3 3 3-1.34 3-3-1.34-3-3-3z"/>
                      </svg>
                    </button>
                    <button className="action-button edit">
                      <svg width="18" height="18" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25zM20.71 7.04c.39-.39.39-1.02 0-1.41l-2.34-2.34c-.39-.39-1.02-.39-1.41 0l-1.83 1.83 3.75 3.75 1.83-1.83z"/>
                      </svg>
                    </button>
                    <button className={`action-button ${investor.status === 'active' ? 'disable' : 'enable'}`}>
                      {investor.status === 'active' ? (
                        <svg width="18" height="18" fill="currentColor" viewBox="0 0 24 24">
                          <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm5 11H7v-2h10v2z"/>
                        </svg>
                      ) : (
                        <svg width="18" height="18" fill="currentColor" viewBox="0 0 24 24">
                          <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm5 11h-4v4h-2v-4H7v-2h4V7h2v4h4v2z"/>
                        </svg>
                      )}
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        
        <div className="pagination">
          <button className="pagination-button prev">
            <svg width="24" height="24" fill="currentColor" viewBox="0 0 24 24">
              <path d="M15.41 7.41L14 6l-6 6 6 6 1.41-1.41L10.83 12z"/>
            </svg>
          </button>
          <div className="pagination-numbers">
            <button className="pagination-number active">1</button>
            <button className="pagination-number">2</button>
            <button className="pagination-number">3</button>
          </div>
          <button className="pagination-button next">
            <svg width="24" height="24" fill="currentColor" viewBox="0 0 24 24">
              <path d="M10 6L8.59 7.41 13.17 12l-4.58 4.59L10 18l6-6z"/>
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
  
  // Componente para la vista de estado del sistema
  const SystemMonitoring = () => (
    <div className="dashboard-content">
      <h1>Monitoreo del Sistema</h1>
      
      {/* Panel de estado general */}
      <div className="system-overview card">
        <div className="system-header">
          <h2>Estado General</h2>
          <span className={`system-status ${systemStatus.system.status.toLowerCase()}`}>
            {systemStatus.system.status === 'operational' ? 'Sistema Operativo' : 'Sistema con Problemas'}
          </span>
        </div>
        
        <div className="system-metrics">
          <div className="system-metric">
            <div className="metric-header">
              <span className="metric-title">Memoria</span>
              <span className="metric-value">{systemStatus.system.memory_usage}%</span>
            </div>
            <div className="progress-bar">
              <div 
                className="progress-fill" 
                style={{ 
                  width: `${systemStatus.system.memory_usage}%`,
                  backgroundColor: systemStatus.system.memory_usage > 80 ? 'var(--danger-color)' : 
                                  systemStatus.system.memory_usage > 60 ? 'var(--warning-color)' : 
                                  'var(--primary-color)'
                }}
              ></div>
            </div>
          </div>
          
          <div className="system-metric">
            <div className="metric-header">
              <span className="metric-title">CPU</span>
              <span className="metric-value">{systemStatus.system.cpu_usage}%</span>
            </div>
            <div className="progress-bar">
              <div 
                className="progress-fill" 
                style={{ 
                  width: `${systemStatus.system.cpu_usage}%`,
                  backgroundColor: systemStatus.system.cpu_usage > 80 ? 'var(--danger-color)' : 
                                  systemStatus.system.cpu_usage > 60 ? 'var(--warning-color)' : 
                                  'var(--primary-color)'
                }}
              ></div>
            </div>
          </div>
        </div>
        
        <div className="system-stats">
          <div className="stat-item">
            <span className="stat-label">Tiempo activo</span>
            <span className="stat-value">{systemStatus.system.uptime}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Usuarios activos</span>
            <span className="stat-value">{systemStatus.system.active_users}</span>
          </div>
        </div>
      </div>
      
      {/* Componentes del sistema */}
      <div className="system-components">
        <h2>Componentes</h2>
        
        <div className="components-grid">
          <div className="component-card card">
            <div className="component-header">
              <h3>Mercado</h3>
              <span className={`component-status ${systemStatus.market.status === 'open' ? 'operational' : 'issue'}`}>
                {systemStatus.market.status === 'open' ? 'Operativo' : 'Con problemas'}
              </span>
            </div>
            <div className="component-details">
              <div className="detail-item">
                <span className="detail-label">Exchanges</span>
                <span className="detail-value">{systemStatus.market.connected_exchanges}</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Pares activos</span>
                <span className="detail-value">{systemStatus.market.active_pairs}</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Latencia</span>
                <span className="detail-value">{systemStatus.market.data_latency}s</span>
              </div>
            </div>
            <div className="component-actions">
              <button className="component-button">Ver detalles</button>
            </div>
          </div>
          
          <div className="component-card card">
            <div className="component-header">
              <h3>Operaciones</h3>
              <span className={`component-status ${systemStatus.operations.success_rate > 95 ? 'operational' : 'issue'}`}>
                {systemStatus.operations.success_rate > 95 ? 'Operativo' : 'Con problemas'}
              </span>
            </div>
            <div className="component-details">
              <div className="detail-item">
                <span className="detail-label">Transacciones hoy</span>
                <span className="detail-value">{systemStatus.operations.transactions_today}</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Pendientes</span>
                <span className="detail-value">{systemStatus.operations.pending_operations}</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Tasa de éxito</span>
                <span className="detail-value">{systemStatus.operations.success_rate}%</span>
              </div>
            </div>
            <div className="component-actions">
              <button className="component-button">Ver detalles</button>
            </div>
          </div>
        </div>
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
            <span className="user-role">Administrador</span>
          </div>
        </div>
        
        <nav className="sidebar-nav">
          <Link to="/admin" className="nav-item" onClick={() => window.innerWidth < 768 && setSidebarOpen(false)}>
            <svg width="24" height="24" fill="currentColor" viewBox="0 0 24 24">
              <path d="M3 13h8V3H3v10zm0 8h8v-6H3v6zm10 0h8V11h-8v10zm0-18v6h8V3h-8z"/>
            </svg>
            <span>Vista General</span>
          </Link>
          <Link to="/admin/investors" className="nav-item" onClick={() => window.innerWidth < 768 && setSidebarOpen(false)}>
            <svg width="24" height="24" fill="currentColor" viewBox="0 0 24 24">
              <path d="M16 11c1.66 0 2.99-1.34 2.99-3S17.66 5 16 5c-1.66 0-3 1.34-3 3s1.34 3 3 3zm-8 0c1.66 0 2.99-1.34 2.99-3S9.66 5 8 5C6.34 5 5 6.34 5 8s1.34 3 3 3zm0 2c-2.33 0-7 1.17-7 3.5V19h14v-2.5c0-2.33-4.67-3.5-7-3.5zm8 0c-.29 0-.62.02-.97.05 1.16.84 1.97 1.97 1.97 3.45V19h6v-2.5c0-2.33-4.67-3.5-7-3.5z"/>
            </svg>
            <span>Inversores</span>
          </Link>
          <Link to="/admin/system" className="nav-item" onClick={() => window.innerWidth < 768 && setSidebarOpen(false)}>
            <svg width="24" height="24" fill="currentColor" viewBox="0 0 24 24">
              <path d="M15 9H9v6h6V9zm-2 4h-2v-2h2v2zm8-2V9h-2V7c0-1.1-.9-2-2-2h-2V3h-2v2h-2V3H9v2H7c-1.1 0-2 .9-2 2v2H3v2h2v2H3v2h2v2c0 1.1.9 2 2 2h2v2h2v-2h2v2h2v-2h2c1.1 0 2-.9 2-2v-2h2v-2h-2v-2h2zm-4 6H7V7h10v10z"/>
            </svg>
            <span>Sistema</span>
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
            <h1>Panel de Administrador</h1>
          </div>
          <div className="header-actions">
            <div className="user-info-mini" onClick={() => navigate('/admin/profile')}>
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
            <Route path="/investors" element={<InvestorsManagement />} />
            <Route path="/system" element={<SystemMonitoring />} />
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
          max-height: calc(100vh - 64px);
        }
        
        .dashboard-content h1 {
          margin-bottom: 1.5rem;
        }
        
        /* System status cards */
        .system-status-cards {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
          gap: 1.5rem;
          margin-bottom: 2rem;
        }
        
        .status-card {
          padding: 1.5rem;
        }
        
        .status-card h3 {
          margin-bottom: 1rem;
          color: var(--text-secondary);
          font-size: 1rem;
        }
        
        .status-indicator {
          display: inline-block;
          padding: 0.25rem 0.75rem;
          border-radius: 50px;
          font-size: 0.85rem;
          margin-bottom: 1rem;
        }
        
        .status-indicator.operational {
          background-color: rgba(76, 175, 80, 0.2);
          color: var(--success-color);
        }
        
        .status-indicator.issue {
          background-color: rgba(255, 82, 82, 0.2);
          color: var(--danger-color);
        }
        
        .status-metrics {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 1rem;
          margin-bottom: 1rem;
        }
        
        .metric {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }
        
        .metric-label {
          color: var(--text-secondary);
          font-size: 0.9rem;
        }
        
        .progress-bar {
          height: 8px;
          background-color: rgba(255, 255, 255, 0.1);
          border-radius: 4px;
          overflow: hidden;
        }
        
        .progress-fill {
          height: 100%;
          background-color: var(--primary-color);
        }
        
        .metric-value {
          font-size: 0.9rem;
          color: var(--text-secondary);
        }
        
        .status-details {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }
        
        .status-details div {
          display: flex;
          justify-content: space-between;
        }
        
        /* Investors summary */
        .investors-summary-section {
          margin-bottom: 2rem;
        }
        
        .section-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 1rem;
        }
        
        .view-all-link {
          color: var(--primary-color);
          text-decoration: none;
          font-size: 0.9rem;
        }
        
        .investors-metrics {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
          gap: 1.5rem;
          padding: 1.5rem;
          margin-bottom: 1.5rem;
        }
        
        .investor-metric {
          display: flex;
          flex-direction: column;
          align-items: center;
          text-align: center;
        }
        
        .investor-metric-value {
          font-size: 1.8rem;
          font-weight: bold;
          color: var(--primary-color);
          margin-bottom: 0.5rem;
        }
        
        .investor-metric-label {
          color: var(--text-secondary);
          font-size: 0.9rem;
        }
        
        .investors-table {
          width: 100%;
          border-collapse: collapse;
        }
        
        .investors-table th {
          text-align: left;
          padding: 0.75rem;
          border-bottom: 1px solid rgba(255, 255, 255, 0.1);
          color: var(--text-secondary);
        }
        
        .investors-table td {
          padding: 0.75rem;
          border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .status-badge {
          display: inline-block;
          padding: 0.25rem 0.5rem;
          border-radius: 4px;
          font-size: 0.8rem;
        }
        
        .status-badge.active {
          background-color: rgba(76, 175, 80, 0.2);
          color: var(--success-color);
        }
        
        .status-badge.inactive {
          background-color: rgba(255, 82, 82, 0.2);
          color: var(--danger-color);
        }
        
        .action-button {
          background: transparent;
          border: none;
          color: var(--text-secondary);
          cursor: pointer;
          padding: 0.3rem;
          border-radius: 4px;
          transition: all 0.3s ease;
        }
        
        .action-button:hover {
          background-color: rgba(255, 255, 255, 0.1);
          color: var(--primary-color);
        }
        
        /* Investor management page */
        .investor-filters {
          display: flex;
          flex-direction: column;
          gap: 1rem;
          margin-bottom: 1.5rem;
          padding: 1.5rem;
        }
        
        .search-bar {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          background-color: rgba(10, 14, 23, 0.5);
          border-radius: var(--border-radius-sm);
          padding: 0.5rem 1rem;
          border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .search-input {
          background: transparent;
          border: none;
          color: var(--text-color);
          font-family: var(--font-body);
          font-size: 1rem;
          width: 100%;
          outline: none;
        }
        
        .filter-options {
          display: flex;
          flex-wrap: wrap;
          gap: 1.5rem;
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
          min-width: 150px;
        }
        
        .investors-full-list {
          padding: 1.5rem;
          overflow-x: auto;
        }
        
        .action-buttons {
          display: flex;
          gap: 0.5rem;
        }
        
        .action-button.view:hover {
          color: var(--primary-color);
        }
        
        .action-button.edit:hover {
          color: var(--secondary-color);
        }
        
        .action-button.disable:hover {
          color: var(--danger-color);
        }
        
        .action-button.enable:hover {
          color: var(--success-color);
        }
        
        .pagination {
          display: flex;
          justify-content: center;
          align-items: center;
          margin-top: 1.5rem;
        }
        
        .pagination-button {
          background: transparent;
          border: none;
          color: var(--text-secondary);
          cursor: pointer;
        }
        
        .pagination-numbers {
          display: flex;
          gap: 0.25rem;
        }
        
        .pagination-number {
          width: 30px;
          height: 30px;
          display: flex;
          align-items: center;
          justify-content: center;
          border-radius: 50%;
          background: transparent;
          border: none;
          color: var(--text-secondary);
          cursor: pointer;
        }
        
        .pagination-number.active {
          background-color: var(--primary-color);
          color: var(--text-color);
        }
        
        /* System monitoring page */
        .system-overview {
          margin-bottom: 2rem;
          padding: 1.5rem;
        }
        
        .system-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 1.5rem;
        }
        
        .system-header h2 {
          margin: 0;
        }
        
        .system-status {
          padding: 0.25rem 0.75rem;
          border-radius: 50px;
          font-size: 0.85rem;
        }
        
        .system-status.operational {
          background-color: rgba(76, 175, 80, 0.2);
          color: var(--success-color);
        }
        
        .system-status.issue {
          background-color: rgba(255, 82, 82, 0.2);
          color: var(--danger-color);
        }
        
        .system-metrics {
          margin-bottom: 1.5rem;
        }
        
        .system-metric {
          margin-bottom: 1rem;
        }
        
        .metric-header {
          display: flex;
          justify-content: space-between;
          margin-bottom: 0.5rem;
        }
        
        .metric-title {
          color: var(--text-secondary);
        }
        
        .system-stats {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
          gap: 1.5rem;
        }
        
        .stat-item {
          display: flex;
          flex-direction: column;
          gap: 0.25rem;
        }
        
        .stat-label {
          color: var(--text-secondary);
          font-size: 0.9rem;
        }
        
        .stat-value {
          font-size: 1.1rem;
          font-weight: 500;
        }
        
        .system-components {
          margin-bottom: 2rem;
        }
        
        .system-components h2 {
          margin-bottom: 1rem;
        }
        
        .components-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
          gap: 1.5rem;
        }
        
        .component-card {
          padding: 1.5rem;
        }
        
        .component-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 1rem;
        }
        
        .component-header h3 {
          margin: 0;
          color: var(--primary-color);
        }
        
        .component-status {
          padding: 0.25rem 0.5rem;
          border-radius: 4px;
          font-size: 0.8rem;
        }
        
        .component-status.operational {
          background-color: rgba(76, 175, 80, 0.2);
          color: var(--success-color);
        }
        
        .component-status.issue {
          background-color: rgba(255, 82, 82, 0.2);
          color: var(--danger-color);
        }
        
        .component-details {
          margin-bottom: 1.5rem;
        }
        
        .detail-item {
          display: flex;
          justify-content: space-between;
          margin-bottom: 0.5rem;
        }
        
        .detail-label {
          color: var(--text-secondary);
        }
        
        .component-actions {
          text-align: right;
        }
        
        .component-button {
          background-color: transparent;
          border: 1px solid var(--primary-color);
          color: var(--primary-color);
          padding: 0.5rem 1rem;
          border-radius: var(--border-radius-sm);
          cursor: pointer;
          transition: all 0.3s ease;
        }
        
        .component-button:hover {
          background-color: var(--primary-color);
          color: var(--text-color);
        }
      `}</style>
    </div>
  );
};

export default AdminDashboard;