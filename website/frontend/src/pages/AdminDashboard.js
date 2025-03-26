import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { FaUsers, FaServer, FaExchangeAlt, FaBrain, FaChartLine, FaUserAlt, FaExclamationTriangle } from 'react-icons/fa';
import axios from 'axios';
import { useAuth } from '../utils/AuthContext';

/**
 * Dashboard para usuarios con rol 'admin'
 */
const AdminDashboard = () => {
  // Estados para los datos
  const [investors, setInvestors] = useState([]);
  const [systemStatus, setSystemStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Obtener datos de autenticación
  const { user } = useAuth();
  
  // Cargar datos para el dashboard de admin
  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        // Obtener lista de inversores
        const investorsResponse = await axios.get('/api/admin/investors', { withCredentials: true });
        setInvestors(investorsResponse.data);
        
        // Obtener estado del sistema
        const systemResponse = await axios.get('/api/admin/system-status', { withCredentials: true });
        setSystemStatus(systemResponse.data);
        
        setLoading(false);
      } catch (err) {
        console.error('Error al cargar datos:', err);
        setError('Error al cargar la información del sistema');
        setLoading(false);
      }
    };
    
    fetchData();
  }, []);
  
  // Variantes de animación
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };
  
  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        duration: 0.4,
        ease: "easeOut"
      }
    }
  };
  
  // Renderizar estado de carga
  if (loading) {
    return (
      <div className="dashboard-loading">
        <div className="dashboard-spinner"></div>
        <p>Cargando datos del sistema...</p>
      </div>
    );
  }
  
  // Renderizar error
  if (error) {
    return (
      <div className="dashboard-error">
        <div className="error-icon">
          <FaExclamationTriangle />
        </div>
        <h2>Error de Sistema</h2>
        <p>{error}</p>
        <button onClick={() => window.location.reload()}>
          Reintentar
        </button>
      </div>
    );
  }

  return (
    <div className="admin-dashboard">
      <div className="dashboard-container">
        {/* Encabezado */}
        <div className="dashboard-header">
          <div className="dashboard-title">
            <h1>Panel de Administración</h1>
            <p>Bienvenido, <span>{user?.username}</span></p>
          </div>
          
          <div className="dashboard-date">
            {new Date().toLocaleDateString('es-ES', { 
              weekday: 'long', 
              year: 'numeric', 
              month: 'long', 
              day: 'numeric' 
            })}
          </div>
        </div>
        
        {/* Tarjetas de estado */}
        <motion.div
          className="dashboard-cards"
          variants={containerVariants}
          initial="hidden"
          animate="visible"
        >
          {/* Estado del sistema */}
          <motion.div 
            className="dashboard-card system-card"
            variants={itemVariants}
          >
            <div className="card-icon">
              <FaServer />
            </div>
            <div className="card-info">
              <h3>Estado del Sistema</h3>
              <div className={`system-status ${systemStatus?.system.status.toLowerCase()}`}>
                {systemStatus?.system.status}
              </div>
              <div className="system-metrics">
                <div className="metric">
                  <span>CPU:</span>
                  <div className="progress-bar">
                    <div 
                      className="progress-fill" 
                      style={{ width: `${systemStatus?.system.cpu_usage}%` }}
                    ></div>
                  </div>
                  <span>{systemStatus?.system.cpu_usage}%</span>
                </div>
                <div className="metric">
                  <span>Memoria:</span>
                  <div className="progress-bar">
                    <div 
                      className="progress-fill" 
                      style={{ width: `${systemStatus?.system.memory_usage}%` }}
                    ></div>
                  </div>
                  <span>{systemStatus?.system.memory_usage}%</span>
                </div>
              </div>
              <div className="system-details">
                <div>
                  <span>Tiempo de actividad: </span>
                  <span className="highlight">{systemStatus?.system.uptime}</span>
                </div>
                <div>
                  <span>Usuarios activos: </span>
                  <span className="highlight">{systemStatus?.system.active_users}</span>
                </div>
              </div>
            </div>
          </motion.div>
          
          {/* Estado del mercado */}
          <motion.div 
            className="dashboard-card market-card"
            variants={itemVariants}
          >
            <div className="card-icon">
              <FaExchangeAlt />
            </div>
            <div className="card-info">
              <h3>Estado del Mercado</h3>
              <div className={`market-status ${systemStatus?.market.status.toLowerCase()}`}>
                {systemStatus?.market.status}
              </div>
              <div className="card-details">
                <div>
                  <span>Exchanges conectados: </span>
                  <span className="highlight">{systemStatus?.market.connected_exchanges}</span>
                </div>
                <div>
                  <span>Pares activos: </span>
                  <span className="highlight">{systemStatus?.market.active_pairs}</span>
                </div>
                <div>
                  <span>Latencia de datos: </span>
                  <span className="highlight">{systemStatus?.market.data_latency}s</span>
                </div>
              </div>
            </div>
          </motion.div>
          
          {/* Operaciones */}
          <motion.div 
            className="dashboard-card operations-card"
            variants={itemVariants}
          >
            <div className="card-icon">
              <FaChartLine />
            </div>
            <div className="card-info">
              <h3>Operaciones</h3>
              <div className="operations-stats">
                <div className="operations-item">
                  <div className="operations-value">{systemStatus?.operations.transactions_today}</div>
                  <div className="operations-label">Transacciones hoy</div>
                </div>
                <div className="operations-item">
                  <div className="operations-value">{systemStatus?.operations.pending_operations}</div>
                  <div className="operations-label">Operaciones pendientes</div>
                </div>
                <div className="operations-item">
                  <div className="operations-value">{systemStatus?.operations.success_rate}%</div>
                  <div className="operations-label">Tasa de éxito</div>
                </div>
              </div>
            </div>
          </motion.div>
        </motion.div>
        
        {/* Inversores */}
        <motion.div
          className="dashboard-section"
          variants={containerVariants}
          initial="hidden"
          animate="visible"
        >
          <motion.div 
            className="section-header"
            variants={itemVariants}
          >
            <h2>
              <FaUsers />
              <span>Inversores Registrados</span>
            </h2>
          </motion.div>
          
          <motion.div 
            className="investors-list"
            variants={itemVariants}
          >
            <div className="investors-header">
              <div className="investor-cell">Usuario</div>
              <div className="investor-cell">Balance</div>
              <div className="investor-cell">Invertido</div>
              <div className="investor-cell">Rendimiento</div>
              <div className="investor-cell">Estado</div>
            </div>
            
            {investors.map((investor) => (
              <motion.div 
                key={investor.id}
                className="investor-row"
                whileHover={{ 
                  backgroundColor: 'rgba(12, 198, 222, 0.05)',
                  transition: { duration: 0.2 }
                }}
              >
                <div className="investor-cell investor-info">
                  <div className="investor-avatar">
                    <FaUserAlt />
                  </div>
                  <div className="investor-details">
                    <div className="investor-name">{investor.username}</div>
                    <div className="investor-email">{investor.email}</div>
                  </div>
                </div>
                <div className="investor-cell">${investor.balance.toLocaleString()}</div>
                <div className="investor-cell">${investor.invested.toLocaleString()}</div>
                <div className={`investor-cell investor-performance ${investor.performance >= 0 ? 'positive' : 'negative'}`}>
                  {investor.performance}%
                </div>
                <div className="investor-cell">
                  <span className={`investor-status ${investor.status}`}>
                    {investor.status}
                  </span>
                </div>
              </motion.div>
            ))}
          </motion.div>
        </motion.div>
        
        {/* Sistema IA */}
        <motion.div
          className="dashboard-section ai-section"
          variants={containerVariants}
          initial="hidden"
          animate="visible"
        >
          <motion.div 
            className="section-header"
            variants={itemVariants}
          >
            <h2>
              <FaBrain />
              <span>Estado de IA</span>
            </h2>
          </motion.div>
          
          <motion.div 
            className="ai-components"
            variants={itemVariants}
          >
            <div className="ai-component">
              <div className="ai-component-header">
                <div className={`ai-status ${systemStatus?.system.status.toLowerCase()}`}></div>
                <h3>Aetherion</h3>
              </div>
              <div className="ai-component-details">
                <div className="ai-detail">
                  <span>Nivel de conciencia: 3</span>
                  <div className="consciousness-level">
                    <div className="consciousness-bar"></div>
                    <div className="consciousness-bar"></div>
                    <div className="consciousness-bar"></div>
                    <div className="consciousness-bar empty"></div>
                    <div className="consciousness-bar empty"></div>
                  </div>
                </div>
                <div className="ai-detail">
                  <span>Modo operativo: Estándar</span>
                </div>
                <div className="ai-detail">
                  <span>Ciclos de adaptación: 157</span>
                </div>
              </div>
            </div>
            
            <div className="ai-component">
              <div className="ai-component-header">
                <div className={`ai-status ${systemStatus?.system.status.toLowerCase()}`}></div>
                <h3>DeepSeek</h3>
              </div>
              <div className="ai-component-details">
                <div className="ai-detail">
                  <span>API Calls: 532</span>
                </div>
                <div className="ai-detail">
                  <span>Tasa de éxito: 99.7%</span>
                </div>
                <div className="ai-detail">
                  <span>Estado API: Operacional</span>
                </div>
              </div>
            </div>
            
            <div className="ai-component">
              <div className="ai-component-header">
                <div className={`ai-status ${systemStatus?.system.status.toLowerCase()}`}></div>
                <h3>Buddha</h3>
              </div>
              <div className="ai-component-details">
                <div className="ai-detail">
                  <span>Precisión: 86.5%</span>
                </div>
                <div className="ai-detail">
                  <span>Ciclos de aprendizaje: 1,245</span>
                </div>
                <div className="ai-detail">
                  <span>Optimizaciones activas: 3</span>
                </div>
              </div>
            </div>
            
            <div className="ai-component">
              <div className="ai-component-header">
                <div className={`ai-status ${systemStatus?.system.status.toLowerCase()}`}></div>
                <h3>Gabriel</h3>
              </div>
              <div className="ai-component-details">
                <div className="ai-detail">
                  <span>Estado emocional: SERENE</span>
                </div>
                <div className="ai-detail">
                  <span>Adaptaciones de comportamiento: 75</span>
                </div>
                <div className="ai-detail">
                  <span>Coherencia lógica: 94.8%</span>
                </div>
              </div>
            </div>
          </motion.div>
        </motion.div>
      </div>
    </div>
  );
};

export default AdminDashboard;

// Estilos específicos para el dashboard de administrador
const styles = `
  .admin-dashboard {
    padding-top: 70px;
    min-height: 100vh;
    background-color: var(--color-background);
  }
  
  .dashboard-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: var(--spacing-lg);
  }
  
  .dashboard-loading {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding-top: 70px;
  }
  
  .dashboard-spinner {
    width: 50px;
    height: 50px;
    border: 3px solid rgba(12, 198, 222, 0.3);
    border-radius: 50%;
    border-top-color: var(--color-primary);
    animation: spin 1s ease-in-out infinite;
    margin-bottom: var(--spacing-md);
  }
  
  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
  
  .dashboard-loading p {
    color: var(--color-primary);
    font-family: var(--font-secondary);
  }
  
  .dashboard-error {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding-top: 70px;
    text-align: center;
  }
  
  .error-icon {
    font-size: 3rem;
    color: var(--color-danger);
    margin-bottom: var(--spacing-md);
  }
  
  .dashboard-error h2 {
    color: var(--color-danger);
    margin-bottom: var(--spacing-md);
  }
  
  .dashboard-error p {
    color: var(--color-text);
    margin-bottom: var(--spacing-lg);
    max-width: 500px;
  }
  
  .dashboard-error button {
    padding: var(--spacing-sm) var(--spacing-lg);
    background: linear-gradient(45deg, var(--color-primary), var(--color-secondary));
    color: var(--color-background);
    border: none;
    border-radius: var(--border-radius-md);
    font-family: var(--font-display);
    font-weight: 500;
    cursor: pointer;
    box-shadow: var(--shadow-soft);
    transition: all var(--transition-normal);
  }
  
  .dashboard-error button:hover {
    transform: translateY(-3px);
    box-shadow: var(--shadow-glow);
  }
  
  .dashboard-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: var(--spacing-lg);
    padding-bottom: var(--spacing-md);
    border-bottom: 1px solid var(--color-border);
  }
  
  .dashboard-title h1 {
    font-size: 2rem;
    color: var(--color-primary);
    margin-bottom: var(--spacing-sm);
  }
  
  .dashboard-title p {
    color: var(--color-text-secondary);
  }
  
  .dashboard-title p span {
    color: var(--color-text);
    font-weight: 600;
  }
  
  .dashboard-date {
    color: var(--color-text-secondary);
    font-family: var(--font-secondary);
    text-transform: capitalize;
  }
  
  .dashboard-cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: var(--spacing-lg);
    margin-bottom: var(--spacing-xl);
  }
  
  .dashboard-card {
    background: rgba(22, 43, 77, 0.7);
    border-radius: var(--border-radius-lg);
    border: 1px solid var(--color-border);
    padding: var(--spacing-lg);
    display: flex;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-medium);
  }
  
  .dashboard-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: var(--hologram-shimmer);
    animation: shimmer 6s infinite;
    pointer-events: none;
  }
  
  .card-icon {
    width: 50px;
    height: 50px;
    background: linear-gradient(135deg, var(--color-primary), var(--color-secondary));
    border-radius: var(--border-radius-circle);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    color: var(--color-background);
    margin-right: var(--spacing-md);
    box-shadow: 0 0 15px rgba(12, 198, 222, 0.3);
  }
  
  .card-info {
    flex: 1;
  }
  
  .card-info h3 {
    color: var(--color-text);
    font-size: 1.2rem;
    margin-bottom: var(--spacing-sm);
    font-family: var(--font-secondary);
    font-weight: 600;
  }
  
  .system-status, .market-status {
    display: inline-block;
    padding: var(--spacing-xs) var(--spacing-sm);
    border-radius: var(--border-radius-sm);
    font-family: var(--font-display);
    font-weight: 600;
    font-size: 0.9rem;
    margin-bottom: var(--spacing-sm);
  }
  
  .system-status.operational, .market-status.open {
    background-color: rgba(62, 255, 163, 0.1);
    color: var(--color-success);
  }
  
  .system-status.degraded, .market-status.limited {
    background-color: rgba(255, 185, 48, 0.1);
    color: var(--color-warning);
  }
  
  .system-status.down, .market-status.closed {
    background-color: rgba(255, 83, 113, 0.1);
    color: var(--color-danger);
  }
  
  .system-metrics {
    margin-bottom: var(--spacing-sm);
  }
  
  .metric {
    display: flex;
    align-items: center;
    margin-bottom: var(--spacing-xs);
    font-size: 0.9rem;
  }
  
  .metric span:first-child {
    width: 70px;
    color: var(--color-text-secondary);
  }
  
  .metric span:last-child {
    width: 40px;
    text-align: right;
    color: var(--color-text);
  }
  
  .progress-bar {
    flex: 1;
    height: 8px;
    background-color: rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    margin: 0 var(--spacing-sm);
    overflow: hidden;
  }
  
  .progress-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--color-primary), var(--color-secondary));
    border-radius: 4px;
    transition: width 0.5s ease;
  }
  
  .card-details {
    color: var(--color-text-secondary);
    font-size: 0.9rem;
  }
  
  .card-details .highlight {
    color: var(--color-text);
    font-weight: 600;
  }
  
  .operations-stats {
    display: flex;
    justify-content: space-between;
  }
  
  .operations-item {
    text-align: center;
  }
  
  .operations-value {
    font-size: 1.5rem;
    font-family: var(--font-display);
    color: var(--color-primary);
    margin-bottom: var(--spacing-xs);
  }
  
  .operations-label {
    font-size: 0.8rem;
    color: var(--color-text-secondary);
  }
  
  .dashboard-section {
    margin-bottom: var(--spacing-xl);
  }
  
  .section-header {
    margin-bottom: var(--spacing-md);
  }
  
  .section-header h2 {
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
    color: var(--color-primary);
    font-size: 1.5rem;
  }
  
  .investors-list {
    background: rgba(22, 43, 77, 0.7);
    border-radius: var(--border-radius-lg);
    border: 1px solid var(--color-border);
    overflow: hidden;
    box-shadow: var(--shadow-medium);
  }
  
  .investors-header {
    display: grid;
    grid-template-columns: 3fr 1fr 1fr 1fr 1fr;
    background: rgba(12, 198, 222, 0.1);
    padding: var(--spacing-md);
    font-weight: 600;
    color: var(--color-primary);
    border-bottom: 1px solid var(--color-border);
  }
  
  .investor-row {
    display: grid;
    grid-template-columns: 3fr 1fr 1fr 1fr 1fr;
    padding: var(--spacing-md);
    border-bottom: 1px solid var(--color-border);
    transition: all var(--transition-fast);
  }
  
  .investor-row:last-child {
    border-bottom: none;
  }
  
  .investor-cell {
    display: flex;
    align-items: center;
  }
  
  .investor-info {
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
  }
  
  .investor-avatar {
    width: 40px;
    height: 40px;
    border-radius: var(--border-radius-circle);
    background: linear-gradient(135deg, var(--color-primary), var(--color-secondary));
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--color-background);
  }
  
  .investor-details {
    display: flex;
    flex-direction: column;
  }
  
  .investor-name {
    color: var(--color-text);
    font-weight: 600;
  }
  
  .investor-email {
    color: var(--color-text-secondary);
    font-size: 0.8rem;
  }
  
  .investor-performance {
    font-weight: 600;
  }
  
  .investor-performance.positive {
    color: var(--color-success);
  }
  
  .investor-performance.negative {
    color: var(--color-danger);
  }
  
  .investor-status {
    display: inline-block;
    padding: 3px var(--spacing-sm);
    border-radius: var(--border-radius-sm);
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: capitalize;
  }
  
  .investor-status.active {
    background-color: rgba(62, 255, 163, 0.1);
    color: var(--color-success);
  }
  
  .investor-status.inactive {
    background-color: rgba(255, 83, 113, 0.1);
    color: var(--color-danger);
  }
  
  .ai-section {
    margin-bottom: 0;
  }
  
  .ai-components {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: var(--spacing-md);
  }
  
  .ai-component {
    background: rgba(22, 43, 77, 0.7);
    border-radius: var(--border-radius-lg);
    border: 1px solid var(--color-border);
    padding: var(--spacing-md);
    box-shadow: var(--shadow-medium);
    transition: all var(--transition-normal);
    position: relative;
    overflow: hidden;
  }
  
  .ai-component::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: var(--hologram-shimmer);
    animation: shimmer 8s infinite;
    pointer-events: none;
  }
  
  .ai-component:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 25px rgba(12, 198, 222, 0.2);
  }
  
  .ai-component-header {
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
    margin-bottom: var(--spacing-md);
  }
  
  .ai-status {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    position: relative;
  }
  
  .ai-status.operational {
    background-color: var(--color-success);
    box-shadow: 0 0 10px var(--color-success);
  }
  
  .ai-status.operational::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    border-radius: 50%;
    background-color: var(--color-success);
    opacity: 0.5;
    animation: pulse 2s infinite;
  }
  
  .ai-status.degraded {
    background-color: var(--color-warning);
    box-shadow: 0 0 10px var(--color-warning);
  }
  
  .ai-status.down {
    background-color: var(--color-danger);
    box-shadow: 0 0 10px var(--color-danger);
  }
  
  .ai-component-header h3 {
    color: var(--color-primary);
    font-family: var(--font-display);
    font-size: 1.2rem;
  }
  
  .ai-component-details {
    color: var(--color-text-secondary);
    font-size: 0.9rem;
  }
  
  .ai-detail {
    margin-bottom: var(--spacing-sm);
  }
  
  .consciousness-level {
    display: flex;
    gap: 3px;
    margin-top: 3px;
  }
  
  .consciousness-bar {
    height: 6px;
    width: 20px;
    background: linear-gradient(90deg, var(--color-primary), var(--color-secondary));
    border-radius: 3px;
  }
  
  .consciousness-bar.empty {
    background: rgba(255, 255, 255, 0.1);
  }
  
  @keyframes pulse {
    0% {
      transform: scale(1);
      opacity: 0.5;
    }
    50% {
      transform: scale(1.5);
      opacity: 0;
    }
    100% {
      transform: scale(1);
      opacity: 0;
    }
  }
  
  /* Responsive */
  @media (max-width: 768px) {
    .dashboard-header {
      flex-direction: column;
      align-items: flex-start;
      gap: var(--spacing-sm);
    }
    
    .investors-header, .investor-row {
      grid-template-columns: 2fr 1fr 1fr 1fr;
    }
    
    .investors-header .investor-cell:nth-child(2),
    .investor-row .investor-cell:nth-child(2) {
      display: none;
    }
  }
  
  @media (max-width: 480px) {
    .dashboard-container {
      padding: var(--spacing-md);
    }
    
    .dashboard-cards {
      gap: var(--spacing-md);
    }
    
    .investors-header, .investor-row {
      grid-template-columns: 2fr 1fr 1fr;
    }
    
    .investors-header .investor-cell:nth-child(3),
    .investor-row .investor-cell:nth-child(3),
    .investors-header .investor-cell:nth-child(4),
    .investor-row .investor-cell:nth-child(4) {
      display: none;
    }
    
    .ai-components {
      grid-template-columns: 1fr;
    }
  }
`;

// Insertar estilos en el documento si no existen ya
if (typeof document !== 'undefined') {
  const styleId = 'admin-dashboard-styles';
  
  if (!document.getElementById(styleId)) {
    const styleElement = document.createElement('style');
    styleElement.id = styleId;
    styleElement.textContent = styles;
    document.head.appendChild(styleElement);
  }
}