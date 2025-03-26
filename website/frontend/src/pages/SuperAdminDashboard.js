import React, { useState, useEffect } from 'react';
import { Routes, Route, Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import axios from 'axios';

// Componentes
import Loading from '../components/Loading';

const SuperAdminDashboard = ({ user, onLogout }) => {
  const [loading, setLoading] = useState(true);
  const [adminsData, setAdminsData] = useState(null);
  const [systemStats, setSystemStats] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    document.title = 'Panel de Super Administrador | Genesis';
    
    const fetchData = async () => {
      try {
        setLoading(true);
        
        // Obtener lista de administradores
        const adminsResponse = await axios.get('/api/super-admin/admins');
        setAdminsData(adminsResponse.data);
        
        // Obtener estadísticas avanzadas del sistema
        const statsResponse = await axios.get('/api/super-admin/system-stats');
        setSystemStats(statsResponse.data);
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
    return <Loading message="Cargando panel de super administrador..." />;
  }

  // Componente para la vista general (dashboard principal)
  const Overview = () => (
    <div className="dashboard-content">
      <h1>Gestión Suprema del Sistema</h1>
      
      {/* Resumen superior */}
      <div className="overview-cards">
        <div className="overview-card card pulse-gentle">
          <h3>Estado General</h3>
          <div className={`status-badge ${systemStats.system.status.toLowerCase()}`}>
            {systemStats.system.status === 'operational' ? 'Operativo' : 'Con problemas'}
          </div>
          <p className="overview-description">
            El sistema está funcionando con normalidad con {systemStats.system.active_users} usuarios activos y un tiempo de actividad de {systemStats.system.uptime}.
          </p>
          <Link to="/super-admin/system" className="card-action-link">
            Ver detalles completos
          </Link>
        </div>
        
        <div className="overview-card card pulse-gentle">
          <h3>Componentes IA</h3>
          <div className="ai-components-summary">
            <div className={`ai-component-status ${systemStats.ai_components.aetherion.status === 'active' ? 'active' : 'inactive'}`}>
              <span>Aetherion</span>
              <div className="component-level">
                <div className="level-indicator" style={{ width: `${(systemStats.ai_components.aetherion.consciousness_level / 10) * 100}%` }}></div>
              </div>
            </div>
            <div className={`ai-component-status ${systemStats.ai_components.deepseek.status === 'active' ? 'active' : 'inactive'}`}>
              <span>DeepSeek</span>
              <div className="component-level">
                <div className="level-indicator" style={{ width: `${(systemStats.ai_components.deepseek.success_rate / 100) * 100}%` }}></div>
              </div>
            </div>
            <div className={`ai-component-status ${systemStats.ai_components.buddha.status === 'active' ? 'active' : 'inactive'}`}>
              <span>Buddha</span>
              <div className="component-level">
                <div className="level-indicator" style={{ width: `${(systemStats.ai_components.buddha.predictions_accuracy / 100) * 100}%` }}></div>
              </div>
            </div>
            <div className={`ai-component-status ${systemStats.ai_components.gabriel.status === 'active' ? 'active' : 'inactive'}`}>
              <span>Gabriel</span>
              <div className="component-level">
                <div className="level-indicator"></div>
              </div>
            </div>
          </div>
          <Link to="/super-admin/ai-components" className="card-action-link">
            Administrar componentes IA
          </Link>
        </div>
        
        <div className="overview-card card pulse-gentle">
          <h3>Seguridad</h3>
          <div className={`security-level ${systemStats.security.threat_level}`}>
            Nivel de amenaza: {systemStats.security.threat_level.toUpperCase()}
          </div>
          <div className="security-details">
            <p><strong>Intentos bloqueados:</strong> {systemStats.security.blocked_attempts}</p>
            <p><strong>Última actualización:</strong> {new Date(systemStats.security.last_update).toLocaleString()}</p>
            <p><strong>Encriptación:</strong> {systemStats.security.encryption_status === 'active' ? 'Activa' : 'Inactiva'}</p>
          </div>
          <Link to="/super-admin/security" className="card-action-link">
            Ir a panel de seguridad
          </Link>
        </div>
      </div>
      
      {/* Administradores */}
      <div className="admins-section">
        <div className="section-header">
          <h2>Administradores</h2>
          <Link to="/super-admin/admins" className="view-all-link">Gestionar administradores</Link>
        </div>
        
        <div className="admins-list card">
          <table className="admins-table">
            <thead>
              <tr>
                <th>Usuario</th>
                <th>Email</th>
                <th>Último acceso</th>
                <th>Estado</th>
                <th>Permisos</th>
                <th>Acciones</th>
              </tr>
            </thead>
            <tbody>
              {adminsData.map((admin) => (
                <tr key={admin.id}>
                  <td>{admin.username}</td>
                  <td>{admin.email}</td>
                  <td>{new Date(admin.last_login).toLocaleString()}</td>
                  <td>
                    <span className={`status-badge ${admin.status}`}>
                      {admin.status === 'active' ? 'Activo' : 'Inactivo'}
                    </span>
                  </td>
                  <td>
                    <div className="permissions-list">
                      {admin.permissions.map((permission, idx) => (
                        <span key={idx} className="permission-badge">
                          {permission}
                        </span>
                      ))}
                    </div>
                  </td>
                  <td>
                    <div className="action-buttons">
                      <button className="action-button edit">
                        <svg width="18" height="18" fill="currentColor" viewBox="0 0 24 24">
                          <path d="M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25zM20.71 7.04c.39-.39.39-1.02 0-1.41l-2.34-2.34c-.39-.39-1.02-.39-1.41 0l-1.83 1.83 3.75 3.75 1.83-1.83z"/>
                        </svg>
                      </button>
                      <button className="action-button delete">
                        <svg width="18" height="18" fill="currentColor" viewBox="0 0 24 24">
                          <path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/>
                        </svg>
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          
          <button className="add-admin-button">
            <svg width="18" height="18" fill="currentColor" viewBox="0 0 24 24">
              <path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z"/>
            </svg>
            Añadir Administrador
          </button>
        </div>
      </div>
      
      {/* Gráficos del sistema */}
      <div className="system-metrics-section">
        <h2>Métricas del Sistema</h2>
        
        <div className="metrics-cards">
          <div className="metric-card card">
            <h3>Carga del Servidor</h3>
            <div className="metric-chart">
              <div className="line-chart">
                {systemStats.system.server_load.map((load, idx) => (
                  <div 
                    key={idx} 
                    className="chart-bar" 
                    style={{ 
                      height: `${(load / 20) * 100}%`,
                      backgroundColor: load > 15 ? 'var(--danger-color)' : 
                                      load > 10 ? 'var(--warning-color)' : 
                                      'var(--primary-color)'
                    }}
                  >
                    <span className="chart-tooltip">{load}</span>
                  </div>
                ))}
              </div>
              <div className="chart-labels">
                <span>Ahora</span>
                <span>-1h</span>
                <span>-2h</span>
                <span>-3h</span>
                <span>-4h</span>
              </div>
            </div>
          </div>
          
          <div className="metric-card card">
            <h3>Tamaño de BD</h3>
            <div className="metric-big-value">
              {systemStats.system.database_size} <span className="metric-unit">MB</span>
            </div>
            <div className="metric-description">
              <div className="trend-indicator up">
                <svg width="14" height="14" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M7 14l5-5 5 5z"/>
                </svg>
                2.3% desde ayer
              </div>
            </div>
          </div>
          
          <div className="metric-card card">
            <h3>Tasa de Error</h3>
            <div className="metric-big-value">
              {systemStats.system.error_rate}%
            </div>
            <div className="metric-description">
              <div className="trend-indicator down">
                <svg width="14" height="14" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M7 10l5 5 5-5z"/>
                </svg>
                0.01% desde ayer
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
  
  // Componente para la gestión de administradores
  const AdminsManagement = () => (
    <div className="dashboard-content">
      <h1>Gestión de Administradores</h1>
      
      <div className="admins-actions card">
        <div className="search-bar">
          <svg width="20" height="20" fill="currentColor" viewBox="0 0 24 24">
            <path d="M15.5 14h-.79l-.28-.27C15.41 12.59 16 11.11 16 9.5 16 5.91 13.09 3 9.5 3S3 5.91 3 9.5 5.91 16 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"/>
          </svg>
          <input 
            type="text" 
            placeholder="Buscar administrador..." 
            className="search-input"
          />
        </div>
        
        <button className="add-admin-button">
          <svg width="18" height="18" fill="currentColor" viewBox="0 0 24 24">
            <path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z"/>
          </svg>
          Añadir Administrador
        </button>
      </div>
      
      <div className="admins-full-list card">
        <table className="admins-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Usuario</th>
              <th>Email</th>
              <th>Último acceso</th>
              <th>Estado</th>
              <th>Permisos</th>
              <th>Acciones</th>
            </tr>
          </thead>
          <tbody>
            {adminsData.map((admin) => (
              <tr key={admin.id}>
                <td>#{admin.id}</td>
                <td>{admin.username}</td>
                <td>{admin.email}</td>
                <td>{new Date(admin.last_login).toLocaleString()}</td>
                <td>
                  <span className={`status-badge ${admin.status}`}>
                    {admin.status === 'active' ? 'Activo' : 'Inactivo'}
                  </span>
                </td>
                <td>
                  <div className="permissions-list">
                    {admin.permissions.map((permission, idx) => (
                      <span key={idx} className="permission-badge">
                        {permission}
                      </span>
                    ))}
                  </div>
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
                    <button className={`action-button ${admin.status === 'active' ? 'disable' : 'enable'}`}>
                      {admin.status === 'active' ? (
                        <svg width="18" height="18" fill="currentColor" viewBox="0 0 24 24">
                          <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm5 11H7v-2h10v2z"/>
                        </svg>
                      ) : (
                        <svg width="18" height="18" fill="currentColor" viewBox="0 0 24 24">
                          <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm5 11h-4v4h-2v-4H7v-2h4V7h2v4h4v2z"/>
                        </svg>
                      )}
                    </button>
                    <button className="action-button delete">
                      <svg width="18" height="18" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/>
                      </svg>
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      
      {/* Modal de añadir/editar admin (oculto por defecto) */}
      <div className="admin-modal" style={{ display: 'none' }}>
        <div className="admin-modal-content card">
          <div className="admin-modal-header">
            <h2>Añadir Administrador</h2>
            <button className="close-modal">
              <svg width="24" height="24" fill="currentColor" viewBox="0 0 24 24">
                <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12 19 6.41z"/>
              </svg>
            </button>
          </div>
          
          <div className="admin-modal-body">
            <div className="form-group">
              <label htmlFor="admin-username">Nombre de usuario</label>
              <input 
                type="text" 
                id="admin-username" 
                className="form-input"
                placeholder="Ingrese nombre de usuario"
              />
            </div>
            
            <div className="form-group">
              <label htmlFor="admin-email">Email</label>
              <input 
                type="email" 
                id="admin-email" 
                className="form-input"
                placeholder="Ingrese email"
              />
            </div>
            
            <div className="form-group">
              <label htmlFor="admin-password">Contraseña</label>
              <input 
                type="password" 
                id="admin-password" 
                className="form-input"
                placeholder="Ingrese contraseña"
              />
            </div>
            
            <div className="form-group">
              <label>Permisos</label>
              <div className="permissions-checkboxes">
                <div className="permission-checkbox">
                  <input type="checkbox" id="perm-manage-users" />
                  <label htmlFor="perm-manage-users">Gestionar usuarios</label>
                </div>
                <div className="permission-checkbox">
                  <input type="checkbox" id="perm-view-reports" />
                  <label htmlFor="perm-view-reports">Ver reportes</label>
                </div>
                <div className="permission-checkbox">
                  <input type="checkbox" id="perm-edit-system" />
                  <label htmlFor="perm-edit-system">Editar sistema</label>
                </div>
                <div className="permission-checkbox">
                  <input type="checkbox" id="perm-manage-transactions" />
                  <label htmlFor="perm-manage-transactions">Gestionar transacciones</label>
                </div>
              </div>
            </div>
          </div>
          
          <div className="admin-modal-footer">
            <button className="cancel-button">Cancelar</button>
            <button className="save-button">Guardar</button>
          </div>
        </div>
      </div>
    </div>
  );
  
  // Componente para monitoreo de IA
  const AIMonitoring = () => (
    <div className="dashboard-content">
      <h1>Monitoreo de Componentes IA</h1>
      
      <div className="ai-components-cards">
        {/* Aetherion */}
        <div className="ai-component-card card">
          <div className="ai-component-header">
            <div className="ai-component-title">
              <h2>Aetherion</h2>
              <div className={`component-status ${systemStats.ai_components.aetherion.status}`}>
                {systemStats.ai_components.aetherion.status === 'active' ? 'Activo' : 'Inactivo'}
              </div>
            </div>
            <button className="component-toggle-button">
              {systemStats.ai_components.aetherion.status === 'active' ? 'Desactivar' : 'Activar'}
            </button>
          </div>
          
          <div className="ai-component-stats">
            <div className="ai-stat">
              <span className="ai-stat-label">Nivel de conciencia</span>
              <div className="progress-bar">
                <div 
                  className="progress-fill" 
                  style={{ 
                    width: `${(systemStats.ai_components.aetherion.consciousness_level / 10) * 100}%`,
                    background: 'linear-gradient(to right, var(--primary-color), var(--secondary-color))'
                  }}
                ></div>
              </div>
              <span className="ai-stat-value">{systemStats.ai_components.aetherion.consciousness_level}/10</span>
            </div>
            
            <div className="ai-stat">
              <span className="ai-stat-label">Adaptaciones</span>
              <span className="ai-stat-value">{systemStats.ai_components.aetherion.adaptation_count}</span>
            </div>
            
            <div className="ai-stat">
              <span className="ai-stat-label">Eficiencia</span>
              <span className="ai-stat-value">{systemStats.ai_components.aetherion.efficiency}%</span>
            </div>
          </div>
          
          <div className="ai-component-actions">
            <button className="component-action">
              <svg width="18" height="18" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 16.5l4-4h-3v-9h-2v9H8l4 4zm9-13h-6v1.99h6v14.03H3V5.49h6V3.5H3c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h18c1.1 0 2-.9 2-2v-14c0-1.1-.9-2-2-2z"/>
              </svg>
              Descargar logs
            </button>
            <button className="component-action">
              <svg width="18" height="18" fill="currentColor" viewBox="0 0 24 24">
                <path d="M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25zM20.71 7.04c.39-.39.39-1.02 0-1.41l-2.34-2.34c-.39-.39-1.02-.39-1.41 0l-1.83 1.83 3.75 3.75 1.83-1.83z"/>
              </svg>
              Configuración
            </button>
          </div>
        </div>
        
        {/* Buddha */}
        <div className="ai-component-card card">
          <div className="ai-component-header">
            <div className="ai-component-title">
              <h2>Buddha</h2>
              <div className={`component-status ${systemStats.ai_components.buddha.status}`}>
                {systemStats.ai_components.buddha.status === 'active' ? 'Activo' : 'Inactivo'}
              </div>
            </div>
            <button className="component-toggle-button">
              {systemStats.ai_components.buddha.status === 'active' ? 'Desactivar' : 'Activar'}
            </button>
          </div>
          
          <div className="ai-component-stats">
            <div className="ai-stat">
              <span className="ai-stat-label">Precisión de predicciones</span>
              <div className="progress-bar">
                <div 
                  className="progress-fill" 
                  style={{ 
                    width: `${systemStats.ai_components.buddha.predictions_accuracy}%`,
                    backgroundColor: 'var(--secondary-color)'
                  }}
                ></div>
              </div>
              <span className="ai-stat-value">{systemStats.ai_components.buddha.predictions_accuracy}%</span>
            </div>
            
            <div className="ai-stat">
              <span className="ai-stat-label">Ciclos de aprendizaje</span>
              <span className="ai-stat-value">{systemStats.ai_components.buddha.learning_cycles}</span>
            </div>
          </div>
          
          <div className="ai-component-actions">
            <button className="component-action">
              <svg width="18" height="18" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 16.5l4-4h-3v-9h-2v9H8l4 4zm9-13h-6v1.99h6v14.03H3V5.49h6V3.5H3c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h18c1.1 0 2-.9 2-2v-14c0-1.1-.9-2-2-2z"/>
              </svg>
              Descargar logs
            </button>
            <button className="component-action">
              <svg width="18" height="18" fill="currentColor" viewBox="0 0 24 24">
                <path d="M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25zM20.71 7.04c.39-.39.39-1.02 0-1.41l-2.34-2.34c-.39-.39-1.02-.39-1.41 0l-1.83 1.83 3.75 3.75 1.83-1.83z"/>
              </svg>
              Configuración
            </button>
          </div>
        </div>
        
        {/* DeepSeek */}
        <div className="ai-component-card card">
          <div className="ai-component-header">
            <div className="ai-component-title">
              <h2>DeepSeek</h2>
              <div className={`component-status ${systemStats.ai_components.deepseek.status}`}>
                {systemStats.ai_components.deepseek.status === 'active' ? 'Activo' : 'Inactivo'}
              </div>
            </div>
            <button className="component-toggle-button">
              {systemStats.ai_components.deepseek.status === 'active' ? 'Desactivar' : 'Activar'}
            </button>
          </div>
          
          <div className="ai-component-stats">
            <div className="ai-stat">
              <span className="ai-stat-label">Llamadas API hoy</span>
              <span className="ai-stat-value">{systemStats.ai_components.deepseek.api_calls_today}</span>
            </div>
            
            <div className="ai-stat">
              <span className="ai-stat-label">Tasa de éxito</span>
              <div className="progress-bar">
                <div 
                  className="progress-fill" 
                  style={{ 
                    width: `${systemStats.ai_components.deepseek.success_rate}%`,
                    backgroundColor: 'var(--accent-color)'
                  }}
                ></div>
              </div>
              <span className="ai-stat-value">{systemStats.ai_components.deepseek.success_rate}%</span>
            </div>
          </div>
          
          <div className="ai-component-actions">
            <button className="component-action">
              <svg width="18" height="18" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 16.5l4-4h-3v-9h-2v9H8l4 4zm9-13h-6v1.99h6v14.03H3V5.49h6V3.5H3c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h18c1.1 0 2-.9 2-2v-14c0-1.1-.9-2-2-2z"/>
              </svg>
              Descargar logs
            </button>
            <button className="component-action">
              <svg width="18" height="18" fill="currentColor" viewBox="0 0 24 24">
                <path d="M20 4H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2zm0 14H4V8l8 5 8-5v10zm-8-7L4 6h16l-8 5z"/>
              </svg>
              Probar API
            </button>
            <button className="component-action">
              <svg width="18" height="18" fill="currentColor" viewBox="0 0 24 24">
                <path d="M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25zM20.71 7.04c.39-.39.39-1.02 0-1.41l-2.34-2.34c-.39-.39-1.02-.39-1.41 0l-1.83 1.83 3.75 3.75 1.83-1.83z"/>
              </svg>
              Configuración
            </button>
          </div>
        </div>
        
        {/* Gabriel */}
        <div className="ai-component-card card">
          <div className="ai-component-header">
            <div className="ai-component-title">
              <h2>Gabriel</h2>
              <div className={`component-status ${systemStats.ai_components.gabriel.status}`}>
                {systemStats.ai_components.gabriel.status === 'active' ? 'Activo' : 'Inactivo'}
              </div>
            </div>
            <button className="component-toggle-button">
              {systemStats.ai_components.gabriel.status === 'active' ? 'Desactivar' : 'Activar'}
            </button>
          </div>
          
          <div className="ai-component-stats">
            <div className="ai-stat">
              <span className="ai-stat-label">Estado actual</span>
              <span className="ai-stat-value">{systemStats.ai_components.gabriel.current_state}</span>
            </div>
            
            <div className="ai-stat">
              <span className="ai-stat-label">Adaptaciones de comportamiento</span>
              <span className="ai-stat-value">{systemStats.ai_components.gabriel.behavior_adaptations}</span>
            </div>
            
            <div className="emotional-states-chart">
              <div className="emotional-state">
                <div className="emotional-state-bar" style={{ height: '70%', backgroundColor: 'var(--success-color)' }}></div>
                <span>SERENE</span>
              </div>
              <div className="emotional-state">
                <div className="emotional-state-bar" style={{ height: '50%', backgroundColor: 'var(--primary-color)' }}></div>
                <span>HOPEFUL</span>
              </div>
              <div className="emotional-state">
                <div className="emotional-state-bar" style={{ height: '30%', backgroundColor: 'var(--warning-color)' }}></div>
                <span>CAUTIOUS</span>
              </div>
              <div className="emotional-state">
                <div className="emotional-state-bar" style={{ height: '15%', backgroundColor: '#ff9800' }}></div>
                <span>RESTLESS</span>
              </div>
              <div className="emotional-state">
                <div className="emotional-state-bar" style={{ height: '5%', backgroundColor: 'var(--danger-color)' }}></div>
                <span>FEARFUL</span>
              </div>
            </div>
          </div>
          
          <div className="ai-component-actions">
            <button className="component-action">
              <svg width="18" height="18" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 16.5l4-4h-3v-9h-2v9H8l4 4zm9-13h-6v1.99h6v14.03H3V5.49h6V3.5H3c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h18c1.1 0 2-.9 2-2v-14c0-1.1-.9-2-2-2z"/>
              </svg>
              Descargar logs
            </button>
            <button className="component-action">
              <svg width="18" height="18" fill="currentColor" viewBox="0 0 24 24">
                <path d="M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25zM20.71 7.04c.39-.39.39-1.02 0-1.41l-2.34-2.34c-.39-.39-1.02-.39-1.41 0l-1.83 1.83 3.75 3.75 1.83-1.83z"/>
              </svg>
              Configuración
            </button>
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
          <div className="user-avatar super-admin">
            {user.username.charAt(0).toUpperCase()}
          </div>
          <div className="user-info">
            <span className="user-name">{user.username}</span>
            <span className="user-role super-admin">Super Administrador</span>
          </div>
        </div>
        
        <nav className="sidebar-nav">
          <Link to="/super-admin" className="nav-item" onClick={() => window.innerWidth < 768 && setSidebarOpen(false)}>
            <svg width="24" height="24" fill="currentColor" viewBox="0 0 24 24">
              <path d="M3 13h8V3H3v10zm0 8h8v-6H3v6zm10 0h8V11h-8v10zm0-18v6h8V3h-8z"/>
            </svg>
            <span>Vista General</span>
          </Link>
          <Link to="/super-admin/admins" className="nav-item" onClick={() => window.innerWidth < 768 && setSidebarOpen(false)}>
            <svg width="24" height="24" fill="currentColor" viewBox="0 0 24 24">
              <path d="M16 11c1.66 0 2.99-1.34 2.99-3S17.66 5 16 5c-1.66 0-3 1.34-3 3s1.34 3 3 3zm-8 0c1.66 0 2.99-1.34 2.99-3S9.66 5 8 5C6.34 5 5 6.34 5 8s1.34 3 3 3zm0 2c-2.33 0-7 1.17-7 3.5V19h14v-2.5c0-2.33-4.67-3.5-7-3.5zm8 0c-.29 0-.62.02-.97.05 1.16.84 1.97 1.97 1.97 3.45V19h6v-2.5c0-2.33-4.67-3.5-7-3.5z"/>
            </svg>
            <span>Administradores</span>
          </Link>
          <Link to="/super-admin/ai-components" className="nav-item" onClick={() => window.innerWidth < 768 && setSidebarOpen(false)}>
            <svg width="24" height="24" fill="currentColor" viewBox="0 0 24 24">
              <path d="M21 3H3c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h18c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm0 16H3V5h18v14zM5 10h9v2H5zm0-3h9v2H5z"/>
            </svg>
            <span>Componentes IA</span>
          </Link>
          <Link to="/super-admin/system" className="nav-item" onClick={() => window.innerWidth < 768 && setSidebarOpen(false)}>
            <svg width="24" height="24" fill="currentColor" viewBox="0 0 24 24">
              <path d="M15 9H9v6h6V9zm-2 4h-2v-2h2v2zm8-2V9h-2V7c0-1.1-.9-2-2-2h-2V3h-2v2h-2V3H9v2H7c-1.1 0-2 .9-2 2v2H3v2h2v2H3v2h2v2c0 1.1.9 2 2 2h2v2h2v-2h2v2h2v-2h2c1.1 0 2-.9 2-2v-2h2v-2h-2v-2h2zm-4 6H7V7h10v10z"/>
            </svg>
            <span>Sistema</span>
          </Link>
          <Link to="/super-admin/security" className="nav-item" onClick={() => window.innerWidth < 768 && setSidebarOpen(false)}>
            <svg width="24" height="24" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4zm0 10.99h7c-.53 4.12-3.28 7.79-7 8.94V12H5V6.3l7-3.11v8.8z"/>
            </svg>
            <span>Seguridad</span>
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
            <h1>Panel de Super Administrador</h1>
          </div>
          <div className="header-actions">
            <div className="user-info-mini" onClick={() => navigate('/super-admin/profile')}>
              <div className="user-avatar-mini super-admin">
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
            <Route path="/admins" element={<AdminsManagement />} />
            <Route path="/ai-components" element={<AIMonitoring />} />
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
        
        .user-avatar.super-admin {
          background: linear-gradient(135deg, #f5c401, var(--accent-color));
          box-shadow: 0 0 15px rgba(245, 196, 1, 0.5);
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
        
        .user-role.super-admin {
          color: #f5c401;
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
        
        .user-avatar-mini.super-admin {
          background: linear-gradient(135deg, #f5c401, var(--accent-color));
          box-shadow: 0 0 10px rgba(245, 196, 1, 0.4);
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
          font-size: 1.8rem;
          background: linear-gradient(135deg, var(--primary-color), var(--secondary-color), #f5c401);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
        }
        
        /* Overview cards */
        .overview-cards {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
          gap: 1.5rem;
          margin-bottom: 2rem;
        }
        
        .overview-card {
          padding: 1.5rem;
          display: flex;
          flex-direction: column;
        }
        
        .overview-card h3 {
          margin-bottom: 1rem;
          color: var(--primary-color);
        }
        
        .status-badge {
          display: inline-block;
          padding: 0.25rem 0.75rem;
          border-radius: 50px;
          font-size: 0.85rem;
          margin-bottom: 1rem;
        }
        
        .status-badge.operational, .status-badge.active {
          background-color: rgba(76, 175, 80, 0.2);
          color: var(--success-color);
        }
        
        .status-badge.issue, .status-badge.inactive {
          background-color: rgba(255, 82, 82, 0.2);
          color: var(--danger-color);
        }
        
        .security-level {
          display: inline-block;
          padding: 0.25rem 0.75rem;
          border-radius: 50px;
          font-size: 0.85rem;
          margin-bottom: 1rem;
        }
        
        .security-level.low {
          background-color: rgba(76, 175, 80, 0.2);
          color: var(--success-color);
        }
        
        .security-level.medium {
          background-color: rgba(255, 193, 7, 0.2);
          color: var(--warning-color);
        }
        
        .security-level.high {
          background-color: rgba(255, 82, 82, 0.2);
          color: var(--danger-color);
        }
        
        .overview-description {
          margin-bottom: 1.5rem;
          flex: 1;
        }
        
        .card-action-link {
          color: var(--primary-color);
          text-decoration: none;
          display: inline-flex;
          align-items: center;
          gap: 0.5rem;
          font-size: 0.9rem;
          margin-top: auto;
          transition: all 0.3s ease;
        }
        
        .card-action-link:hover {
          color: var(--accent-color);
        }
        
        .pulse-gentle {
          box-shadow: 0 0 0 rgba(5, 178, 220, 0.4);
          animation: pulse-gentle 2s infinite;
        }
        
        @keyframes pulse-gentle {
          0% {
            box-shadow: 0 0 0 0 rgba(5, 178, 220, 0.4);
          }
          70% {
            box-shadow: 0 0 0 10px rgba(5, 178, 220, 0);
          }
          100% {
            box-shadow: 0 0 0 0 rgba(5, 178, 220, 0);
          }
        }
        
        /* AI Components */
        .ai-components-summary {
          display: flex;
          flex-direction: column;
          gap: 0.75rem;
          margin-bottom: 1.5rem;
        }
        
        .ai-component-status {
          display: flex;
          justify-content: space-between;
          align-items: center;
        }
        
        .ai-component-status.active {
          color: var(--primary-color);
        }
        
        .ai-component-status.inactive {
          color: var(--text-secondary);
        }
        
        .component-level {
          width: 100px;
          height: 6px;
          background-color: rgba(255, 255, 255, 0.1);
          border-radius: 3px;
          overflow: hidden;
        }
        
        .level-indicator {
          height: 100%;
          background: linear-gradient(to right, var(--primary-color), var(--secondary-color));
        }
        
        /* Security details */
        .security-details {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
          margin-bottom: 1.5rem;
        }
        
        /* Admins section */
        .admins-section {
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
        
        .admins-list {
          padding: 1.5rem;
        }
        
        .admins-table {
          width: 100%;
          border-collapse: collapse;
          margin-bottom: 1rem;
        }
        
        .admins-table th {
          text-align: left;
          padding: 0.75rem;
          border-bottom: 1px solid rgba(255, 255, 255, 0.1);
          color: var(--text-secondary);
        }
        
        .admins-table td {
          padding: 0.75rem;
          border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .permissions-list {
          display: flex;
          flex-wrap: wrap;
          gap: 0.5rem;
        }
        
        .permission-badge {
          display: inline-block;
          padding: 0.2rem 0.5rem;
          border-radius: 4px;
          font-size: 0.8rem;
          background-color: rgba(5, 178, 220, 0.1);
          color: var(--primary-color);
        }
        
        .action-buttons {
          display: flex;
          gap: 0.5rem;
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
        
        .action-button.edit:hover {
          color: var(--primary-color);
        }
        
        .action-button.delete:hover {
          color: var(--danger-color);
        }
        
        .add-admin-button {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          margin-top: 1rem;
          padding: 0.5rem 1rem;
          background-color: var(--primary-color);
          color: var(--text-color);
          border: none;
          border-radius: var(--border-radius-sm);
          cursor: pointer;
          transition: all 0.3s ease;
          font-family: var(--font-body);
        }
        
        .add-admin-button:hover {
          background-color: var(--secondary-color);
        }
        
        /* System metrics */
        .system-metrics-section {
          margin-bottom: 2rem;
        }
        
        .system-metrics-section h2 {
          margin-bottom: 1rem;
        }
        
        .metrics-cards {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
          gap: 1.5rem;
        }
        
        .metric-card {
          padding: 1.5rem;
          text-align: center;
        }
        
        .metric-card h3 {
          margin-bottom: 1rem;
          color: var(--text-secondary);
        }
        
        .metric-chart {
          height: 150px;
          display: flex;
          flex-direction: column;
          margin-bottom: 1rem;
        }
        
        .line-chart {
          flex: 1;
          display: flex;
          align-items: flex-end;
          gap: 0.5rem;
          padding-bottom: 0.5rem;
        }
        
        .chart-bar {
          flex: 1;
          position: relative;
          transition: all 0.3s ease;
        }
        
        .chart-bar:hover .chart-tooltip {
          opacity: 1;
        }
        
        .chart-tooltip {
          position: absolute;
          top: -25px;
          left: 50%;
          transform: translateX(-50%);
          background-color: var(--card-background);
          padding: 0.25rem 0.5rem;
          border-radius: 4px;
          font-size: 0.8rem;
          opacity: 0;
          transition: all 0.3s ease;
        }
        
        .chart-labels {
          display: flex;
          justify-content: space-between;
          color: var(--text-secondary);
          font-size: 0.8rem;
        }
        
        .metric-big-value {
          font-size: 2.5rem;
          font-weight: bold;
          color: var(--primary-color);
          margin-bottom: 0.5rem;
        }
        
        .metric-unit {
          font-size: 1rem;
          color: var(--text-secondary);
        }
        
        .trend-indicator {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 0.25rem;
          font-size: 0.9rem;
        }
        
        .trend-indicator.up {
          color: var(--success-color);
        }
        
        .trend-indicator.down {
          color: var(--success-color);
        }
        
        /* AI Components page */
        .ai-components-cards {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
          gap: 1.5rem;
        }
        
        .ai-component-card {
          padding: 1.5rem;
        }
        
        .ai-component-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 1.5rem;
        }
        
        .ai-component-title {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }
        
        .ai-component-title h2 {
          margin: 0;
          color: var(--primary-color);
        }
        
        .component-status {
          display: inline-block;
          padding: 0.25rem 0.5rem;
          border-radius: 4px;
          font-size: 0.8rem;
        }
        
        .component-status.active {
          background-color: rgba(76, 175, 80, 0.2);
          color: var(--success-color);
        }
        
        .component-status.inactive {
          background-color: rgba(255, 82, 82, 0.2);
          color: var(--danger-color);
        }
        
        .component-toggle-button {
          padding: 0.5rem 1rem;
          border: none;
          border-radius: var(--border-radius-sm);
          background-color: var(--card-background);
          color: var(--text-color);
          cursor: pointer;
          transition: all 0.3s ease;
          border: 1px solid var(--primary-color);
        }
        
        .component-toggle-button:hover {
          background-color: var(--primary-color);
          color: var(--text-color);
        }
        
        .ai-component-stats {
          margin-bottom: 1.5rem;
        }
        
        .ai-stat {
          margin-bottom: 1rem;
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }
        
        .ai-stat-label {
          color: var(--text-secondary);
          font-size: 0.9rem;
        }
        
        .ai-stat-value {
          font-weight: 500;
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
          transition: width 0.3s ease;
        }
        
        .emotional-states-chart {
          display: flex;
          justify-content: space-between;
          align-items: flex-end;
          height: 80px;
          margin-top: 1rem;
        }
        
        .emotional-state {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 0.5rem;
          width: 40px;
        }
        
        .emotional-state-bar {
          width: 20px;
          border-radius: 2px 2px 0 0;
        }
        
        .emotional-state span {
          font-size: 0.7rem;
          color: var(--text-secondary);
          transform: rotate(-45deg);
          transform-origin: right center;
          white-space: nowrap;
        }
        
        .ai-component-actions {
          display: flex;
          flex-wrap: wrap;
          gap: 0.5rem;
        }
        
        .component-action {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          padding: 0.5rem 1rem;
          background-color: transparent;
          border: 1px solid var(--text-secondary);
          color: var(--text-secondary);
          border-radius: var(--border-radius-sm);
          cursor: pointer;
          transition: all 0.3s ease;
          font-size: 0.9rem;
        }
        
        .component-action:hover {
          border-color: var(--primary-color);
          color: var(--primary-color);
        }
        
        /* Modal styles */
        .admin-modal {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background-color: rgba(0, 0, 0, 0.7);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 1001;
        }
        
        .admin-modal-content {
          width: 90%;
          max-width: 500px;
          max-height: 90vh;
          overflow-y: auto;
          padding: 0;
        }
        
        .admin-modal-header {
          padding: 1.5rem;
          display: flex;
          justify-content: space-between;
          align-items: center;
          border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .admin-modal-header h2 {
          margin: 0;
        }
        
        .close-modal {
          background: transparent;
          border: none;
          color: var(--text-secondary);
          cursor: pointer;
        }
        
        .admin-modal-body {
          padding: 1.5rem;
        }
        
        .form-group {
          margin-bottom: 1.5rem;
        }
        
        .form-group label {
          display: block;
          margin-bottom: 0.5rem;
          color: var(--text-secondary);
        }
        
        .form-input {
          width: 100%;
          padding: 0.8rem;
          background-color: rgba(10, 14, 23, 0.5);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: var(--border-radius-sm);
          color: var(--text-color);
          font-family: var(--font-body);
        }
        
        .permissions-checkboxes {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
          gap: 0.75rem;
        }
        
        .permission-checkbox {
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }
        
        .admin-modal-footer {
          padding: 1.5rem;
          display: flex;
          justify-content: flex-end;
          gap: 1rem;
          border-top: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .cancel-button, .save-button {
          padding: 0.75rem 1.5rem;
          border-radius: var(--border-radius-sm);
          cursor: pointer;
          font-family: var(--font-body);
          transition: all 0.3s ease;
        }
        
        .cancel-button {
          background-color: transparent;
          border: 1px solid var(--text-secondary);
          color: var(--text-secondary);
        }
        
        .cancel-button:hover {
          border-color: var(--text-color);
          color: var(--text-color);
        }
        
        .save-button {
          background-color: var(--primary-color);
          border: none;
          color: var(--text-color);
        }
        
        .save-button:hover {
          background-color: var(--secondary-color);
        }
      `}</style>
    </div>
  );
};

export default SuperAdminDashboard;