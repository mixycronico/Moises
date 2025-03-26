import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { FaChartLine, FaCoins, FaExchangeAlt, FaArrowUp, FaArrowDown, FaHistory } from 'react-icons/fa';
import axios from 'axios';
import { useAuth } from '../utils/AuthContext';

/**
 * Dashboard para usuarios con rol 'investor'
 */
const InvestorDashboard = () => {
  // Estados para los datos
  const [portfolio, setPortfolio] = useState(null);
  const [transactions, setTransactions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Obtener datos de autenticación
  const { user } = useAuth();
  
  // Cargar datos del portafolio
  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        // Obtener portafolio
        const portfolioResponse = await axios.get('/api/investor/portfolio', { withCredentials: true });
        setPortfolio(portfolioResponse.data);
        
        // Obtener transacciones
        const transactionsResponse = await axios.get('/api/investor/transactions', { withCredentials: true });
        setTransactions(transactionsResponse.data);
        
        setLoading(false);
      } catch (err) {
        console.error('Error al cargar datos:', err);
        setError('Error al cargar los datos de tu portafolio');
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
        <p>Cargando datos del portafolio...</p>
      </div>
    );
  }
  
  // Renderizar error
  if (error) {
    return (
      <div className="dashboard-error">
        <div className="error-icon">
          <FaExchangeAlt />
        </div>
        <h2>Error de Conexión</h2>
        <p>{error}</p>
        <button onClick={() => window.location.reload()}>
          Reintentar
        </button>
      </div>
    );
  }

  return (
    <div className="investor-dashboard">
      <div className="dashboard-container">
        {/* Encabezado */}
        <div className="dashboard-header">
          <div className="dashboard-title">
            <h1>Dashboard de Inversor</h1>
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
        
        {/* Tarjetas principales */}
        <motion.div
          className="dashboard-cards"
          variants={containerVariants}
          initial="hidden"
          animate="visible"
        >
          {/* Tarjeta de balance */}
          <motion.div 
            className="dashboard-card balance-card"
            variants={itemVariants}
          >
            <div className="card-icon">
              <FaCoins />
            </div>
            <div className="card-info">
              <h3>Balance Total</h3>
              <div className="card-value">${portfolio?.balance.toLocaleString()}</div>
              <div className="card-details">
                <div>
                  <span>Invertido: </span>
                  <span className="highlight">${portfolio?.invested.toLocaleString()}</span>
                </div>
                <div>
                  <span>Disponible: </span>
                  <span className="highlight">${portfolio?.available.toLocaleString()}</span>
                </div>
              </div>
            </div>
          </motion.div>
          
          {/* Tarjeta de rendimiento */}
          <motion.div 
            className="dashboard-card performance-card"
            variants={itemVariants}
          >
            <div className="card-icon">
              <FaChartLine />
            </div>
            <div className="card-info">
              <h3>Rendimiento</h3>
              <div className="performance-grid">
                <div className="performance-item">
                  <div className="performance-label">Hoy</div>
                  <div className={`performance-value ${portfolio?.performance.daily >= 0 ? 'positive' : 'negative'}`}>
                    {portfolio?.performance.daily >= 0 ? <FaArrowUp /> : <FaArrowDown />}
                    {Math.abs(portfolio?.performance.daily)}%
                  </div>
                </div>
                
                <div className="performance-item">
                  <div className="performance-label">Esta semana</div>
                  <div className={`performance-value ${portfolio?.performance.weekly >= 0 ? 'positive' : 'negative'}`}>
                    {portfolio?.performance.weekly >= 0 ? <FaArrowUp /> : <FaArrowDown />}
                    {Math.abs(portfolio?.performance.weekly)}%
                  </div>
                </div>
                
                <div className="performance-item">
                  <div className="performance-label">Este mes</div>
                  <div className={`performance-value ${portfolio?.performance.monthly >= 0 ? 'positive' : 'negative'}`}>
                    {portfolio?.performance.monthly >= 0 ? <FaArrowUp /> : <FaArrowDown />}
                    {Math.abs(portfolio?.performance.monthly)}%
                  </div>
                </div>
                
                <div className="performance-item">
                  <div className="performance-label">Este año</div>
                  <div className={`performance-value ${portfolio?.performance.yearly >= 0 ? 'positive' : 'negative'}`}>
                    {portfolio?.performance.yearly >= 0 ? <FaArrowUp /> : <FaArrowDown />}
                    {Math.abs(portfolio?.performance.yearly)}%
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </motion.div>
        
        {/* Portafolio */}
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
              <FaCoins />
              <span>Mi Portafolio</span>
            </h2>
          </motion.div>
          
          <motion.div 
            className="portfolio-grid"
            variants={itemVariants}
          >
            <div className="portfolio-header">
              <div className="portfolio-cell">Activo</div>
              <div className="portfolio-cell">Cantidad</div>
              <div className="portfolio-cell">Valor (USD)</div>
              <div className="portfolio-cell">Cambio 24h</div>
            </div>
            
            {portfolio?.investments.map((investment) => (
              <motion.div 
                key={investment.id}
                className="portfolio-row"
                whileHover={{ 
                  backgroundColor: 'rgba(12, 198, 222, 0.05)',
                  transition: { duration: 0.2 }
                }}
              >
                <div className="portfolio-cell asset-cell">
                  <div className="asset-icon">{investment.symbol.slice(0, 1)}</div>
                  <div className="asset-info">
                    <div className="asset-name">{investment.name}</div>
                    <div className="asset-symbol">{investment.symbol}</div>
                  </div>
                </div>
                <div className="portfolio-cell">{investment.amount}</div>
                <div className="portfolio-cell">${investment.value_usd.toLocaleString()}</div>
                <div className={`portfolio-cell change-cell ${investment.change_24h >= 0 ? 'positive' : 'negative'}`}>
                  {investment.change_24h >= 0 ? <FaArrowUp /> : <FaArrowDown />}
                  {Math.abs(investment.change_24h)}%
                </div>
              </motion.div>
            ))}
          </motion.div>
        </motion.div>
        
        {/* Transacciones recientes */}
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
              <FaHistory />
              <span>Transacciones Recientes</span>
            </h2>
          </motion.div>
          
          <motion.div 
            className="transactions-list"
            variants={itemVariants}
          >
            {transactions.length > 0 ? (
              transactions.map((transaction) => (
                <motion.div 
                  key={transaction.id}
                  className="transaction-item"
                  whileHover={{ 
                    backgroundColor: 'rgba(12, 198, 222, 0.05)',
                    transition: { duration: 0.2 }
                  }}
                >
                  <div className={`transaction-type ${transaction.type.toLowerCase()}`}>
                    {transaction.type}
                  </div>
                  <div className="transaction-details">
                    <div className="transaction-asset">
                      <span className="asset-symbol">{transaction.symbol}</span>
                      <span className="asset-name">{transaction.asset}</span>
                    </div>
                    <div className="transaction-amount">
                      {transaction.amount} {transaction.symbol}
                    </div>
                    <div className="transaction-price">
                      Precio: ${transaction.price_usd}
                    </div>
                    <div className="transaction-total">
                      Total: ${transaction.total_usd}
                    </div>
                  </div>
                  <div className="transaction-meta">
                    <div className="transaction-date">
                      {new Date(transaction.date).toLocaleDateString('es-ES', {
                        year: 'numeric',
                        month: 'short',
                        day: 'numeric',
                        hour: '2-digit',
                        minute: '2-digit'
                      })}
                    </div>
                    <div className={`transaction-status ${transaction.status.toLowerCase()}`}>
                      {transaction.status}
                    </div>
                  </div>
                </motion.div>
              ))
            ) : (
              <div className="empty-list">
                <p>No hay transacciones recientes</p>
              </div>
            )}
          </motion.div>
        </motion.div>
      </div>
    </div>
  );
};

export default InvestorDashboard;

// Estilos específicos para el dashboard de inversor
const styles = `
  .investor-dashboard {
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
    margin-bottom: var(--spacing-lg);
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
  
  .card-value {
    font-size: 2rem;
    font-family: var(--font-display);
    color: var(--color-primary);
    margin-bottom: var(--spacing-sm);
  }
  
  .card-details {
    color: var(--color-text-secondary);
    font-size: 0.9rem;
  }
  
  .card-details .highlight {
    color: var(--color-text);
    font-weight: 600;
  }
  
  .performance-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: var(--spacing-sm);
  }
  
  .performance-item {
    padding: var(--spacing-sm);
    background: rgba(255, 255, 255, 0.05);
    border-radius: var(--border-radius-sm);
  }
  
  .performance-label {
    font-size: 0.8rem;
    color: var(--color-text-secondary);
    margin-bottom: 2px;
  }
  
  .performance-value {
    display: flex;
    align-items: center;
    gap: 5px;
    font-family: var(--font-display);
    font-weight: 600;
  }
  
  .performance-value.positive {
    color: var(--color-success);
  }
  
  .performance-value.negative {
    color: var(--color-danger);
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
  
  .portfolio-grid {
    background: rgba(22, 43, 77, 0.7);
    border-radius: var(--border-radius-lg);
    border: 1px solid var(--color-border);
    overflow: hidden;
    box-shadow: var(--shadow-medium);
  }
  
  .portfolio-header {
    display: grid;
    grid-template-columns: 3fr 1fr 1fr 1fr;
    background: rgba(12, 198, 222, 0.1);
    padding: var(--spacing-md);
    font-weight: 600;
    color: var(--color-primary);
    border-bottom: 1px solid var(--color-border);
  }
  
  .portfolio-row {
    display: grid;
    grid-template-columns: 3fr 1fr 1fr 1fr;
    padding: var(--spacing-md);
    border-bottom: 1px solid var(--color-border);
    transition: all var(--transition-fast);
  }
  
  .portfolio-row:last-child {
    border-bottom: none;
  }
  
  .portfolio-cell {
    display: flex;
    align-items: center;
  }
  
  .asset-cell {
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
  }
  
  .asset-icon {
    width: 40px;
    height: 40px;
    border-radius: var(--border-radius-circle);
    background: linear-gradient(135deg, var(--color-primary), var(--color-secondary));
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: var(--font-display);
    font-weight: bold;
    color: var(--color-background);
  }
  
  .asset-info {
    display: flex;
    flex-direction: column;
  }
  
  .asset-name {
    color: var(--color-text);
    font-weight: 600;
  }
  
  .asset-symbol {
    color: var(--color-text-secondary);
    font-size: 0.8rem;
  }
  
  .change-cell {
    display: flex;
    align-items: center;
    gap: 5px;
    font-weight: 600;
  }
  
  .change-cell.positive {
    color: var(--color-success);
  }
  
  .change-cell.negative {
    color: var(--color-danger);
  }
  
  .transactions-list {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-md);
  }
  
  .transaction-item {
    background: rgba(22, 43, 77, 0.7);
    border-radius: var(--border-radius-lg);
    border: 1px solid var(--color-border);
    padding: var(--spacing-md);
    display: flex;
    align-items: center;
    gap: var(--spacing-md);
    transition: all var(--transition-fast);
    box-shadow: var(--shadow-soft);
  }
  
  .transaction-type {
    padding: var(--spacing-sm) var(--spacing-md);
    border-radius: var(--border-radius-sm);
    font-family: var(--font-display);
    font-weight: 600;
    min-width: 80px;
    text-align: center;
  }
  
  .transaction-type.buy {
    background-color: rgba(62, 255, 163, 0.1);
    color: var(--color-success);
  }
  
  .transaction-type.sell {
    background-color: rgba(255, 83, 113, 0.1);
    color: var(--color-danger);
  }
  
  .transaction-details {
    flex: 1;
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: var(--spacing-sm) var(--spacing-lg);
  }
  
  .transaction-asset {
    display: flex;
    flex-direction: column;
  }
  
  .transaction-asset .asset-symbol {
    font-family: var(--font-display);
    color: var(--color-primary);
    font-weight: 600;
    font-size: 0.9rem;
  }
  
  .transaction-asset .asset-name {
    color: var(--color-text-secondary);
    font-size: 0.8rem;
  }
  
  .transaction-amount {
    font-family: var(--font-secondary);
    color: var(--color-text);
    font-weight: 600;
  }
  
  .transaction-price, .transaction-total {
    color: var(--color-text-secondary);
    font-size: 0.9rem;
  }
  
  .transaction-meta {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: var(--spacing-sm);
  }
  
  .transaction-date {
    color: var(--color-text-secondary);
    font-size: 0.8rem;
  }
  
  .transaction-status {
    padding: 3px var(--spacing-sm);
    border-radius: var(--border-radius-sm);
    font-size: 0.8rem;
    font-weight: 600;
  }
  
  .transaction-status.completed {
    background-color: rgba(62, 255, 163, 0.1);
    color: var(--color-success);
  }
  
  .transaction-status.pending {
    background-color: rgba(255, 185, 48, 0.1);
    color: var(--color-warning);
  }
  
  .transaction-status.failed {
    background-color: rgba(255, 83, 113, 0.1);
    color: var(--color-danger);
  }
  
  .empty-list {
    padding: var(--spacing-lg);
    text-align: center;
    color: var(--color-text-secondary);
    background: rgba(22, 43, 77, 0.5);
    border-radius: var(--border-radius-lg);
    border: 1px dashed var(--color-border);
  }
  
  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
  
  /* Responsive */
  @media (max-width: 768px) {
    .dashboard-header {
      flex-direction: column;
      align-items: flex-start;
      gap: var(--spacing-sm);
    }
    
    .portfolio-header, .portfolio-row {
      grid-template-columns: 2fr 1fr 1fr 1fr;
      font-size: 0.9rem;
    }
    
    .transaction-item {
      flex-direction: column;
      align-items: flex-start;
    }
    
    .transaction-meta {
      align-items: flex-start;
      width: 100%;
      flex-direction: row;
      justify-content: space-between;
      margin-top: var(--spacing-sm);
    }
  }
  
  @media (max-width: 480px) {
    .dashboard-container {
      padding: var(--spacing-md);
    }
    
    .dashboard-card {
      padding: var(--spacing-md);
    }
    
    .portfolio-header, .portfolio-row {
      grid-template-columns: 2fr 1fr 1fr;
    }
    
    .portfolio-header .portfolio-cell:nth-child(3),
    .portfolio-row .portfolio-cell:nth-child(3) {
      display: none;
    }
    
    .transaction-details {
      grid-template-columns: 1fr;
      gap: var(--spacing-xs);
    }
  }
`;

// Insertar estilos en el documento si no existen ya
if (typeof document !== 'undefined') {
  const styleId = 'investor-dashboard-styles';
  
  if (!document.getElementById(styleId)) {
    const styleElement = document.createElement('style');
    styleElement.id = styleId;
    styleElement.textContent = styles;
    document.head.appendChild(styleElement);
  }
}