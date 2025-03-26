import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { FaUsers, FaBrain, FaShield, FaDatabase, FaServer, FaExclamationTriangle, FaUserShield, FaUserCog } from 'react-icons/fa';
import axios from 'axios';
import { useAuth } from '../utils/AuthContext';

/**
 * Dashboard para usuarios con rol 'super_admin'
 */
const SuperAdminDashboard = () => {
  // Estados para los datos
  const [admins, setAdmins] = useState([]);
  const [systemStats, setSystemStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Obtener datos de autenticación
  const { user } = useAuth();
  
  // Cargar datos para el dashboard de super admin
  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        // Obtener lista de administradores
        const adminsResponse = await axios.get('/api/super-admin/admins', { withCredentials: true });
        setAdmins(adminsResponse.data);
        
        // Obtener estadísticas avanzadas del sistema
        const statsResponse = await axios.get('/api/super-admin/system-stats', { withCredentials: true });
        setSystemStats(statsResponse.data);
        
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
    <div className="super-admin-dashboard">
      <div className="dashboard-container">
        {/* Encabezado */}
        <div className="dashboard-header">
          <div className="dashboard-title">
            <h1>Panel de Super Administración</h1>
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
        
        {/* Estadísticas del sistema */}
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
              <FaServer />
              <span>Vista General del Sistema</span>
            </h2>
          </motion.div>
          
          <motion.div
            className="system-stats-grid"
            variants={itemVariants}
          >
            {/* Estado general */}
            <div className="stat-card">
              <div className="stat-header">
                <div className={`stat-status ${systemStats?.system.status.toLowerCase()}`}></div>
                <h3>Estado General</h3>
              </div>
              <div className="stat-content">
                <div className="stat-value">
                  <span className={`system-status ${systemStats?.system.status.toLowerCase()}`}>
                    {systemStats?.system.status}
                  </span>
                </div>
                <div className="stat-details">
                  <div className="stat-item">
                    <span className="stat-label">Tiempo de actividad:</span>
                    <span className="stat-text">{systemStats?.system.uptime}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Usuarios activos:</span>
                    <span className="stat-text">{systemStats?.system.active_users}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Tasa de errores:</span>
                    <span className="stat-text">{systemStats?.system.error_rate}%</span>
                  </div>
                </div>
              </div>
            </div>
            
            {/* Recursos */}
            <div className="stat-card">
              <div className="stat-header">
                <div className="stat-icon">
                  <FaDatabase />
                </div>
                <h3>Recursos</h3>
              </div>
              <div className="stat-content">
                <div className="server-resources">
                  <div className="resource-item">
                    <div className="resource-header">
                      <span>CPU</span>
                      <span>{systemStats?.system.cpu_usage}%</span>
                    </div>
                    <div className="resource-bar">
                      <div className="resource-fill" style={{ width: `${systemStats?.system.cpu_usage}%` }}></div>
                    </div>
                  </div>
                  
                  <div className="resource-item">
                    <div className="resource-header">
                      <span>Memoria</span>
                      <span>{systemStats?.system.memory_usage}%</span>
                    </div>
                    <div className="resource-bar">
                      <div className="resource-fill" style={{ width: `${systemStats?.system.memory_usage}%` }}></div>
                    </div>
                  </div>
                  
                  <div className="resource-item">
                    <div className="resource-header">
                      <span>Base de datos</span>
                      <span>{systemStats?.system.database_size} MB</span>
                    </div>
                  </div>
                  
                  <div className="resource-item">
                    <div className="resource-header">
                      <span>Carga del servidor</span>
                    </div>
                    <div className="server-load-graph">
                      {systemStats?.system.server_load.map((load, index) => (
                        <div 
                          key={index} 
                          className="load-bar" 
                          style={{ height: `${load * 4}px` }}
                          title={`${load}`}
                        ></div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
            
            {/* Seguridad */}
            <div className="stat-card">
              <div className="stat-header">
                <div className="stat-icon">
                  <FaShield />
                </div>
                <h3>Seguridad</h3>
              </div>
              <div className="stat-content">
                <div className="security-status">
                  <div className="security-threat">
                    <span className="threat-label">Nivel de amenaza:</span>
                    <span className={`threat-level ${systemStats?.security.threat_level}`}>
                      {systemStats?.security.threat_level}
                    </span>
                  </div>
                  
                  <div className="security-details">
                    <div className="security-item">
                      <span className="security-label">Intentos bloqueados:</span>
                      <span className="security-value">{systemStats?.security.blocked_attempts}</span>
                    </div>
                    
                    <div className="security-item">
                      <span className="security-label">Última actualización:</span>
                      <span className="security-value">
                        {new Date(systemStats?.security.last_update).toLocaleString('es-ES')}
                      </span>
                    </div>
                    
                    <div className="security-item">
                      <span className="security-label">Estado de encriptación:</span>
                      <span className="security-value encryption-status">
                        {systemStats?.security.encryption_status}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </motion.div>
        
        {/* Componentes IA */}
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
              <FaBrain />
              <span>Componentes Avanzados de IA</span>
            </h2>
          </motion.div>
          
          <motion.div
            className="ai-advanced-stats"
            variants={itemVariants}
          >
            <div className="ai-advanced-card">
              <div className="ai-advanced-header">
                <h3>Aetherion</h3>
                <div className={`ai-advanced-status ${systemStats?.ai_components.aetherion.status}`}>
                  {systemStats?.ai_components.aetherion.status}
                </div>
              </div>
              
              <div className="ai-advanced-metrics">
                <div className="ai-metric">
                  <span className="ai-metric-label">Nivel de conciencia</span>
                  <div className="ai-metric-value">
                    {systemStats?.ai_components.aetherion.consciousness_level}/5
                  </div>
                  <div className="ai-level-bars">
                    {[...Array(5)].map((_, i) => (
                      <div 
                        key={i} 
                        className={`ai-level-bar ${i < systemStats?.ai_components.aetherion.consciousness_level ? 'filled' : ''}`}
                      ></div>
                    ))}
                  </div>
                </div>
                
                <div className="ai-metric">
                  <span className="ai-metric-label">Adaptaciones</span>
                  <div className="ai-metric-value">
                    {systemStats?.ai_components.aetherion.adaptation_count}
                  </div>
                </div>
                
                <div className="ai-metric">
                  <span className="ai-metric-label">Eficiencia</span>
                  <div className="ai-metric-value">
                    {systemStats?.ai_components.aetherion.efficiency}%
                  </div>
                  <div className="ai-progress-bar">
                    <div 
                      className="ai-progress-fill" 
                      style={{ width: `${systemStats?.ai_components.aetherion.efficiency}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="ai-components-grid">
              <div className="ai-component-card">
                <div className="ai-component-header">
                  <h3>DeepSeek</h3>
                  <div className={`ai-component-status ${systemStats?.ai_components.deepseek.status}`}>
                    {systemStats?.ai_components.deepseek.status}
                  </div>
                </div>
                
                <div className="ai-component-metrics">
                  <div className="ai-component-metric">
                    <span className="ai-component-label">API Calls Hoy</span>
                    <span className="ai-component-value">
                      {systemStats?.ai_components.deepseek.api_calls_today}
                    </span>
                  </div>
                  
                  <div className="ai-component-metric">
                    <span className="ai-component-label">Tasa de Éxito</span>
                    <span className="ai-component-value">
                      {systemStats?.ai_components.deepseek.success_rate}%
                    </span>
                  </div>
                </div>
              </div>
              
              <div className="ai-component-card">
                <div className="ai-component-header">
                  <h3>Buddha</h3>
                  <div className={`ai-component-status ${systemStats?.ai_components.buddha.status}`}>
                    {systemStats?.ai_components.buddha.status}
                  </div>
                </div>
                
                <div className="ai-component-metrics">
                  <div className="ai-component-metric">
                    <span className="ai-component-label">Precisión de Predicciones</span>
                    <span className="ai-component-value">
                      {systemStats?.ai_components.buddha.predictions_accuracy}%
                    </span>
                  </div>
                  
                  <div className="ai-component-metric">
                    <span className="ai-component-label">Ciclos de Aprendizaje</span>
                    <span className="ai-component-value">
                      {systemStats?.ai_components.buddha.learning_cycles}
                    </span>
                  </div>
                </div>
              </div>
              
              <div className="ai-component-card">
                <div className="ai-component-header">
                  <h3>Gabriel</h3>
                  <div className={`ai-component-status ${systemStats?.ai_components.gabriel.status}`}>
                    {systemStats?.ai_components.gabriel.status}
                  </div>
                </div>
                
                <div className="ai-component-metrics">
                  <div className="ai-component-metric">
                    <span className="ai-component-label">Estado Emocional</span>
                    <span className="ai-component-value emotional-state">
                      {systemStats?.ai_components.gabriel.current_state}
                    </span>
                  </div>
                  
                  <div className="ai-component-metric">
                    <span className="ai-component-label">Adaptaciones</span>
                    <span className="ai-component-value">
                      {systemStats?.ai_components.gabriel.behavior_adaptations}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </motion.div>
        
        {/* Administradores */}
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
              <span>Administradores</span>
            </h2>
          </motion.div>
          
          <motion.div 
            className="admins-list"
            variants={itemVariants}
          >
            <div className="admins-header">
              <div className="admin-cell">Usuario</div>
              <div className="admin-cell">Email</div>
              <div className="admin-cell">Último Login</div>
              <div className="admin-cell">Estado</div>
              <div className="admin-cell">Permisos</div>
            </div>
            
            {admins.map((admin) => (
              <motion.div 
                key={admin.id}
                className="admin-row"
                whileHover={{ 
                  backgroundColor: 'rgba(12, 198, 222, 0.05)',
                  transition: { duration: 0.2 }
                }}
              >
                <div className="admin-cell admin-info">
                  <div className="admin-avatar">
                    <FaUserCog />
                  </div>
                  <div className="admin-username">{admin.username}</div>
                </div>
                <div className="admin-cell admin-email">{admin.email}</div>
                <div className="admin-cell admin-last-login">
                  {new Date(admin.last_login).toLocaleString('es-ES')}
                </div>
                <div className="admin-cell">
                  <span className={`admin-status ${admin.status}`}>
                    {admin.status}
                  </span>
                </div>
                <div className="admin-cell admin-permissions">
                  {admin.permissions.map(permission => (
                    <span key={permission} className="admin-permission">
                      {permission}
                    </span>
                  ))}
                </div>
              </motion.div>
            ))}
          </motion.div>
        </motion.div>
        
        {/* Acciones rápidas */}
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
              <FaUserShield />
              <span>Acciones de Super Admin</span>
            </h2>
          </motion.div>
          
          <motion.div 
            className="quick-actions"
            variants={itemVariants}
          >
            <button className="action-button">
              Reiniciar Servidor
            </button>
            <button className="action-button">
              Recargar Configuración
            </button>
            <button className="action-button">
              Limpiar Caché de Sistema
            </button>
            <button className="action-button">
              Copia de Seguridad
            </button>
            <button className="action-button danger">
              Modo Mantenimiento
            </button>
          </motion.div>
        </motion.div>
      </div>
    </div>
  );
};

export default SuperAdminDashboard;

// Estilos específicos para el dashboard de super administrador
const styles = `
  .super-admin-dashboard {
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
  
  /* Sistema Stats Grid */
  .system-stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: var(--spacing-md);
  }
  
  .stat-card {
    background: rgba(22, 43, 77, 0.7);
    border-radius: var(--border-radius-lg);
    border: 1px solid var(--color-border);
    padding: var(--spacing-md);
    box-shadow: var(--shadow-medium);
    position: relative;
    overflow: hidden;
  }
  
  .stat-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: var(--hologram-shimmer);
    animation: shimmer 8s infinite;
    pointer-events: none;
    z-index: 0;
  }
  
  .stat-header {
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
    margin-bottom: var(--spacing-sm);
    position: relative;
    z-index: 1;
  }
  
  .stat-status, .ai-advanced-status, .ai-component-status {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    position: relative;
  }
  
  .stat-status.operational, .ai-advanced-status.active, .ai-component-status.active {
    background-color: var(--color-success);
    box-shadow: 0 0 10px var(--color-success);
  }
  
  .stat-status.operational::after, .ai-advanced-status.active::after, .ai-component-status.active::after {
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
  
  .stat-status.degraded, .ai-advanced-status.limited, .ai-component-status.limited {
    background-color: var(--color-warning);
    box-shadow: 0 0 10px var(--color-warning);
  }
  
  .stat-status.down, .ai-advanced-status.inactive, .ai-component-status.inactive {
    background-color: var(--color-danger);
    box-shadow: 0 0 10px var(--color-danger);
  }
  
  .stat-icon, .ai-icon {
    width: 30px;
    height: 30px;
    border-radius: var(--border-radius-circle);
    background: linear-gradient(135deg, var(--color-primary), var(--color-secondary));
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--color-background);
    box-shadow: 0 0 10px rgba(12, 198, 222, 0.3);
  }
  
  .stat-header h3 {
    color: var(--color-primary);
    font-family: var(--font-display);
    font-size: 1.2rem;
  }
  
  .stat-content {
    position: relative;
    z-index: 1;
  }
  
  .system-status {
    display: inline-block;
    padding: var(--spacing-xs) var(--spacing-sm);
    border-radius: var(--border-radius-sm);
    font-family: var(--font-display);
    font-weight: 600;
    font-size: 0.9rem;
    margin-bottom: var(--spacing-sm);
  }
  
  .system-status.operational {
    background-color: rgba(62, 255, 163, 0.1);
    color: var(--color-success);
  }
  
  .system-status.degraded {
    background-color: rgba(255, 185, 48, 0.1);
    color: var(--color-warning);
  }
  
  .system-status.down {
    background-color: rgba(255, 83, 113, 0.1);
    color: var(--color-danger);
  }
  
  .stat-details {
    margin-top: var(--spacing-sm);
  }
  
  .stat-item {
    margin-bottom: var(--spacing-xs);
  }
  
  .stat-label {
    color: var(--color-text-secondary);
    margin-right: var(--spacing-xs);
    font-size: 0.9rem;
  }
  
  .stat-text {
    color: var(--color-text);
    font-weight: 500;
    font-size: 0.9rem;
  }
  
  /* Server Resources */
  .server-resources {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-sm);
  }
  
  .resource-item {
    margin-bottom: var(--spacing-xs);
  }
  
  .resource-header {
    display: flex;
    justify-content: space-between;
    color: var(--color-text-secondary);
    font-size: 0.9rem;
    margin-bottom: 3px;
  }
  
  .resource-bar {
    height: 8px;
    background-color: rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    overflow: hidden;
  }
  
  .resource-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--color-primary), var(--color-secondary));
    border-radius: 4px;
    transition: width 0.5s ease;
  }
  
  .server-load-graph {
    display: flex;
    align-items: flex-end;
    height: 50px;
    gap: 3px;
    margin-top: var(--spacing-xs);
  }
  
  .load-bar {
    flex: 1;
    background: linear-gradient(180deg, var(--color-primary), var(--color-secondary));
    border-radius: 2px 2px 0 0;
    min-height: 3px;
    transition: height 0.5s ease;
  }
  
  /* Security */
  .security-status {
    position: relative;
  }
  
  .security-threat {
    margin-bottom: var(--spacing-md);
  }
  
  .threat-label {
    color: var(--color-text-secondary);
    font-size: 0.9rem;
    margin-right: var(--spacing-xs);
  }
  
  .threat-level {
    padding: 3px var(--spacing-sm);
    border-radius: var(--border-radius-sm);
    font-weight: 600;
    font-size: 0.9rem;
    text-transform: uppercase;
  }
  
  .threat-level.low {
    background-color: rgba(62, 255, 163, 0.1);
    color: var(--color-success);
  }
  
  .threat-level.medium {
    background-color: rgba(255, 185, 48, 0.1);
    color: var(--color-warning);
  }
  
  .threat-level.high {
    background-color: rgba(255, 83, 113, 0.1);
    color: var(--color-danger);
  }
  
  .security-details {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-xs);
  }
  
  .security-item {
    display: flex;
    justify-content: space-between;
  }
  
  .security-label {
    color: var(--color-text-secondary);
    font-size: 0.9rem;
  }
  
  .security-value {
    color: var(--color-text);
    font-weight: 500;
    font-size: 0.9rem;
  }
  
  .encryption-status {
    color: var(--color-success);
  }
  
  /* AI Components */
  .ai-advanced-stats {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-md);
  }
  
  .ai-advanced-card {
    background: rgba(22, 43, 77, 0.7);
    border-radius: var(--border-radius-lg);
    border: 1px solid var(--color-border);
    padding: var(--spacing-md);
    box-shadow: var(--shadow-medium);
    margin-bottom: var(--spacing-md);
    position: relative;
    overflow: hidden;
  }
  
  .ai-advanced-card::before {
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
  
  .ai-advanced-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: var(--spacing-md);
    position: relative;
    z-index: 1;
  }
  
  .ai-advanced-header h3 {
    color: var(--color-primary);
    font-family: var(--font-display);
    font-size: 1.2rem;
  }
  
  .ai-advanced-metrics {
    display: flex;
    justify-content: space-between;
    gap: var(--spacing-md);
    position: relative;
    z-index: 1;
  }
  
  .ai-metric {
    flex: 1;
  }
  
  .ai-metric-label {
    color: var(--color-text-secondary);
    font-size: 0.9rem;
    display: block;
    margin-bottom: var(--spacing-xs);
  }
  
  .ai-metric-value {
    color: var(--color-primary);
    font-size: 1.2rem;
    font-family: var(--font-display);
    font-weight: 600;
    margin-bottom: var(--spacing-xs);
  }
  
  .ai-level-bars {
    display: flex;
    gap: 3px;
  }
  
  .ai-level-bar {
    height: 8px;
    flex: 1;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 4px;
  }
  
  .ai-level-bar.filled {
    background: linear-gradient(90deg, var(--color-primary), var(--color-secondary));
  }
  
  .ai-progress-bar {
    height: 8px;
    background-color: rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    overflow: hidden;
  }
  
  .ai-progress-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--color-primary), var(--color-secondary));
    border-radius: 4px;
    transition: width 0.5s ease;
  }
  
  .ai-components-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: var(--spacing-md);
  }
  
  .ai-component-card {
    background: rgba(22, 43, 77, 0.7);
    border-radius: var(--border-radius-lg);
    border: 1px solid var(--color-border);
    padding: var(--spacing-md);
    box-shadow: var(--shadow-medium);
    position: relative;
    overflow: hidden;
    transition: all var(--transition-normal);
  }
  
  .ai-component-card::before {
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
  
  .ai-component-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 25px rgba(12, 198, 222, 0.2);
  }
  
  .ai-component-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: var(--spacing-sm);
    position: relative;
    z-index: 1;
  }
  
  .ai-component-header h3 {
    color: var(--color-primary);
    font-family: var(--font-display);
    font-size: 1.1rem;
  }
  
  .ai-component-metrics {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-sm);
    position: relative;
    z-index: 1;
  }
  
  .ai-component-metric {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  
  .ai-component-label {
    color: var(--color-text-secondary);
    font-size: 0.9rem;
  }
  
  .ai-component-value {
    color: var(--color-text);
    font-weight: 600;
    font-size: 0.9rem;
  }
  
  .emotional-state {
    color: var(--color-primary);
    padding: 2px var(--spacing-sm);
    background: rgba(12, 198, 222, 0.1);
    border-radius: var(--border-radius-sm);
  }
  
  /* Admins List */
  .admins-list {
    background: rgba(22, 43, 77, 0.7);
    border-radius: var(--border-radius-lg);
    border: 1px solid var(--color-border);
    overflow: hidden;
    box-shadow: var(--shadow-medium);
  }
  
  .admins-header {
    display: grid;
    grid-template-columns: 2fr 2fr 1fr 1fr 2fr;
    background: rgba(12, 198, 222, 0.1);
    padding: var(--spacing-md);
    font-weight: 600;
    color: var(--color-primary);
    border-bottom: 1px solid var(--color-border);
  }
  
  .admin-row {
    display: grid;
    grid-template-columns: 2fr 2fr 1fr 1fr 2fr;
    padding: var(--spacing-md);
    border-bottom: 1px solid var(--color-border);
    transition: all var(--transition-fast);
  }
  
  .admin-row:last-child {
    border-bottom: none;
  }
  
  .admin-cell {
    display: flex;
    align-items: center;
  }
  
  .admin-info {
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
  }
  
  .admin-avatar {
    width: 36px;
    height: 36px;
    border-radius: var(--border-radius-circle);
    background: linear-gradient(135deg, var(--color-primary), var(--color-secondary));
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--color-background);
  }
  
  .admin-username {
    color: var(--color-text);
    font-weight: 600;
  }
  
  .admin-email {
    color: var(--color-text-secondary);
    font-size: 0.9rem;
  }
  
  .admin-last-login {
    color: var(--color-text-secondary);
    font-size: 0.9rem;
  }
  
  .admin-status {
    display: inline-block;
    padding: 3px var(--spacing-sm);
    border-radius: var(--border-radius-sm);
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: capitalize;
  }
  
  .admin-status.active {
    background-color: rgba(62, 255, 163, 0.1);
    color: var(--color-success);
  }
  
  .admin-status.inactive {
    background-color: rgba(255, 83, 113, 0.1);
    color: var(--color-danger);
  }
  
  .admin-permissions {
    display: flex;
    flex-wrap: wrap;
    gap: var(--spacing-xs);
  }
  
  .admin-permission {
    padding: 2px var(--spacing-sm);
    background: rgba(146, 112, 255, 0.1);
    color: var(--color-secondary);
    border-radius: var(--border-radius-sm);
    font-size: 0.8rem;
    white-space: nowrap;
  }
  
  /* Quick Actions */
  .quick-actions {
    display: flex;
    flex-wrap: wrap;
    gap: var(--spacing-md);
  }
  
  .action-button {
    padding: var(--spacing-sm) var(--spacing-lg);
    background: rgba(12, 198, 222, 0.1);
    border: 1px solid var(--color-border);
    border-radius: var(--border-radius-md);
    color: var(--color-primary);
    font-family: var(--font-display);
    font-weight: 500;
    cursor: pointer;
    transition: all var(--transition-normal);
    position: relative;
    overflow: hidden;
  }
  
  .action-button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: var(--hologram-shimmer);
    transition: all var(--transition-normal);
  }
  
  .action-button:hover {
    background: rgba(12, 198, 222, 0.2);
    transform: translateY(-3px);
    box-shadow: var(--shadow-glow);
  }
  
  .action-button:hover::before {
    animation: shimmer 2s infinite;
  }
  
  .action-button.danger {
    background: rgba(255, 83, 113, 0.1);
    color: var(--color-danger);
  }
  
  .action-button.danger:hover {
    background: rgba(255, 83, 113, 0.2);
    box-shadow: 0 5px 15px rgba(255, 83, 113, 0.3);
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
    
    .ai-advanced-metrics {
      flex-direction: column;
    }
    
    .admins-header, .admin-row {
      grid-template-columns: 2fr 2fr 1fr;
    }
    
    .admins-header .admin-cell:nth-child(3),
    .admin-row .admin-cell:nth-child(3),
    .admins-header .admin-cell:nth-child(4),
    .admin-row .admin-cell:nth-child(4) {
      display: none;
    }
  }
  
  @media (max-width: 480px) {
    .dashboard-container {
      padding: var(--spacing-md);
    }
    
    .system-stats-grid {
      grid-template-columns: 1fr;
    }
    
    .ai-components-grid {
      grid-template-columns: 1fr;
    }
    
    .admins-header, .admin-row {
      grid-template-columns: 1fr 1fr;
    }
    
    .admins-header .admin-cell:nth-child(5),
    .admin-row .admin-cell:nth-child(5) {
      display: none;
    }
    
    .quick-actions {
      flex-direction: column;
    }
  }
`;

// Insertar estilos en el documento si no existen ya
if (typeof document !== 'undefined') {
  const styleId = 'super-admin-dashboard-styles';
  
  if (!document.getElementById(styleId)) {
    const styleElement = document.createElement('style');
    styleElement.id = styleId;
    styleElement.textContent = styles;
    document.head.appendChild(styleElement);
  }
}