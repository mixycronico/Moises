import React, { useState } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { FaUserAlt, FaChartLine, FaSignOutAlt, FaBars, FaTimes, FaCog, FaHome } from 'react-icons/fa';
import { useAuth } from '../utils/AuthContext';

/**
 * Barra de navegación principal con menú adaptativo
 */
const NavBar = () => {
  const [menuOpen, setMenuOpen] = useState(false);
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  // Determinar la ruta base según el rol del usuario
  const getBaseRoute = () => {
    if (!user) return '/';
    
    switch (user.role) {
      case 'investor':
        return '/investor';
      case 'admin':
        return '/admin';
      case 'super_admin':
        return '/super-admin';
      default:
        return '/';
    }
  };

  // Manejar cierre de sesión
  const handleLogout = async () => {
    await logout();
    navigate('/login');
    setMenuOpen(false);
  };

  // Cerrar menú al hacer clic en un enlace
  const closeMenu = () => {
    setMenuOpen(false);
  };

  // Variantes de animación para el menú móvil
  const menuVariants = {
    closed: {
      opacity: 0,
      x: '100%',
      transition: {
        type: 'tween',
        duration: 0.3
      }
    },
    open: {
      opacity: 1,
      x: 0,
      transition: {
        type: 'tween',
        duration: 0.3
      }
    }
  };

  // Variantes de animación para ítems del menú
  const itemVariants = {
    closed: { opacity: 0, x: 50 },
    open: (i) => ({
      opacity: 1,
      x: 0,
      transition: {
        delay: i * 0.1,
        duration: 0.3
      }
    })
  };

  return (
    <nav className="navbar">
      <div className="navbar-container">
        {/* Logo y nombre */}
        <Link to={getBaseRoute()} className="navbar-logo" onClick={closeMenu}>
          <div className="navbar-logo-icon">G</div>
          <span className="navbar-logo-text">GENESIS</span>
        </Link>

        {/* Botón de menú móvil */}
        <button 
          className="navbar-menu-toggle" 
          onClick={() => setMenuOpen(!menuOpen)}
          aria-label={menuOpen ? "Cerrar menú" : "Abrir menú"}
        >
          {menuOpen ? <FaTimes /> : <FaBars />}
        </button>

        {/* Menú de navegación (escritorio) */}
        <div className="navbar-menu desktop">
          {user && (
            <div className="navbar-links">
              <Link 
                to={getBaseRoute()} 
                className={`navbar-link ${location.pathname === getBaseRoute() ? 'active' : ''}`}
              >
                <FaHome className="navbar-icon" />
                <span>Dashboard</span>
              </Link>
              
              {/* Enlaces específicos según rol */}
              {user.role === 'investor' && (
                <Link 
                  to="/investor/portfolio" 
                  className={`navbar-link ${location.pathname.includes('/portfolio') ? 'active' : ''}`}
                >
                  <FaChartLine className="navbar-icon" />
                  <span>Portfolio</span>
                </Link>
              )}
              
              {(user.role === 'admin' || user.role === 'super_admin') && (
                <Link 
                  to={`${user.role === 'admin' ? '/admin' : '/super-admin'}/settings`}
                  className={`navbar-link ${location.pathname.includes('/settings') ? 'active' : ''}`}
                >
                  <FaCog className="navbar-icon" />
                  <span>Configuración</span>
                </Link>
              )}
            </div>
          )}

          {/* Perfil/Login */}
          <div className="navbar-auth">
            {user ? (
              <>
                <div className="navbar-user">
                  <FaUserAlt className="navbar-icon" />
                  <span>{user.username}</span>
                </div>
                <button className="navbar-logout" onClick={handleLogout}>
                  <FaSignOutAlt className="navbar-icon" />
                  <span>Salir</span>
                </button>
              </>
            ) : (
              <Link to="/login" className="navbar-login">
                <FaUserAlt className="navbar-icon" />
                <span>Ingresar</span>
              </Link>
            )}
          </div>
        </div>

        {/* Menú móvil */}
        <AnimatePresence>
          {menuOpen && (
            <motion.div
              className="navbar-menu mobile"
              initial="closed"
              animate="open"
              exit="closed"
              variants={menuVariants}
            >
              {user && (
                <div className="navbar-mobile-user">
                  <FaUserAlt className="navbar-icon" />
                  <span>{user.username}</span>
                  <div className="navbar-mobile-role">{user.role}</div>
                </div>
              )}

              <div className="navbar-mobile-links">
                {user && (
                  <motion.div
                    custom={0}
                    variants={itemVariants}
                  >
                    <Link 
                      to={getBaseRoute()} 
                      className={`navbar-mobile-link ${location.pathname === getBaseRoute() ? 'active' : ''}`}
                      onClick={closeMenu}
                    >
                      <FaHome className="navbar-icon" />
                      <span>Dashboard</span>
                    </Link>
                  </motion.div>
                )}

                {/* Enlaces específicos según rol (móvil) */}
                {user && user.role === 'investor' && (
                  <motion.div
                    custom={1}
                    variants={itemVariants}
                  >
                    <Link 
                      to="/investor/portfolio" 
                      className={`navbar-mobile-link ${location.pathname.includes('/portfolio') ? 'active' : ''}`}
                      onClick={closeMenu}
                    >
                      <FaChartLine className="navbar-icon" />
                      <span>Portfolio</span>
                    </Link>
                  </motion.div>
                )}

                {user && (user.role === 'admin' || user.role === 'super_admin') && (
                  <motion.div
                    custom={1}
                    variants={itemVariants}
                  >
                    <Link 
                      to={`${user.role === 'admin' ? '/admin' : '/super-admin'}/settings`}
                      className={`navbar-mobile-link ${location.pathname.includes('/settings') ? 'active' : ''}`}
                      onClick={closeMenu}
                    >
                      <FaCog className="navbar-icon" />
                      <span>Configuración</span>
                    </Link>
                  </motion.div>
                )}

                {/* Botón de login/logout (móvil) */}
                <motion.div
                  custom={user ? 2 : 0}
                  variants={itemVariants}
                  className="navbar-mobile-auth"
                >
                  {user ? (
                    <button className="navbar-mobile-logout" onClick={handleLogout}>
                      <FaSignOutAlt className="navbar-icon" />
                      <span>Cerrar Sesión</span>
                    </button>
                  ) : (
                    <Link to="/login" className="navbar-mobile-login" onClick={closeMenu}>
                      <FaUserAlt className="navbar-icon" />
                      <span>Ingresar</span>
                    </Link>
                  )}
                </motion.div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </nav>
  );
};

export default NavBar;

// Estilos específicos del componente
const styles = `
  .navbar {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: 70px;
    background-color: rgba(12, 29, 59, 0.8);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid rgba(12, 198, 222, 0.3);
    box-shadow: 0 5px 20px rgba(0, 0, 0, 0.2);
    z-index: var(--z-floating);
  }
  
  .navbar-container {
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 var(--spacing-lg);
    max-width: 1400px;
    margin: 0 auto;
  }
  
  .navbar-logo {
    display: flex;
    align-items: center;
    color: var(--color-primary);
    text-decoration: none;
    gap: var(--spacing-sm);
  }
  
  .navbar-logo-icon {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    border: 2px solid var(--color-primary);
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: var(--font-display);
    font-weight: bold;
    font-size: 1.5rem;
    box-shadow: 0 0 10px rgba(12, 198, 222, 0.5);
  }
  
  .navbar-logo-text {
    font-family: var(--font-display);
    font-size: 1.2rem;
    font-weight: bold;
    letter-spacing: 2px;
  }
  
  .navbar-menu {
    display: flex;
    align-items: center;
    gap: var(--spacing-lg);
  }
  
  .navbar-links {
    display: flex;
    gap: var(--spacing-md);
  }
  
  .navbar-link, .navbar-login, .navbar-logout {
    display: flex;
    align-items: center;
    color: var(--color-text);
    text-decoration: none;
    gap: var(--spacing-xs);
    padding: var(--spacing-xs) var(--spacing-sm);
    border-radius: var(--border-radius-md);
    transition: all var(--transition-fast);
    font-family: var(--font-secondary);
    font-weight: 500;
  }
  
  .navbar-link:hover, .navbar-login:hover, .navbar-logout:hover {
    background-color: rgba(12, 198, 222, 0.1);
    transform: translateY(-2px);
  }
  
  .navbar-link.active {
    color: var(--color-primary);
    background-color: rgba(12, 198, 222, 0.1);
    box-shadow: 0 0 5px rgba(12, 198, 222, 0.3);
  }
  
  .navbar-icon {
    font-size: 1rem;
  }
  
  .navbar-user {
    display: flex;
    align-items: center;
    gap: var(--spacing-xs);
    color: var(--color-primary);
    font-family: var(--font-secondary);
    font-weight: 500;
    padding: var(--spacing-xs) var(--spacing-sm);
    border-radius: var(--border-radius-md);
    background-color: rgba(12, 198, 222, 0.1);
  }
  
  .navbar-logout {
    background: none;
    border: none;
    cursor: pointer;
    color: var(--color-text);
  }
  
  .navbar-logout:hover {
    color: var(--color-danger);
  }
  
  .navbar-menu-toggle {
    display: none;
    background: none;
    border: none;
    color: var(--color-primary);
    font-size: 1.5rem;
    cursor: pointer;
  }
  
  .navbar-menu.mobile {
    display: none;
  }
  
  /* Estilos para móvil */
  @media (max-width: 768px) {
    .navbar-menu.desktop {
      display: none;
    }
    
    .navbar-menu-toggle {
      display: block;
    }
    
    .navbar-menu.mobile {
      display: flex;
      flex-direction: column;
      position: fixed;
      top: 70px;
      right: 0;
      bottom: 0;
      width: 80%;
      max-width: 300px;
      background-color: var(--color-background);
      border-left: 1px solid var(--color-border);
      box-shadow: -5px 0 20px rgba(0, 0, 0, 0.3);
      padding: var(--spacing-lg);
      z-index: var(--z-overlay);
    }
    
    .navbar-mobile-user {
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: var(--spacing-lg) 0;
      border-bottom: 1px solid var(--color-border);
      margin-bottom: var(--spacing-lg);
    }
    
    .navbar-mobile-user .navbar-icon {
      font-size: 2rem;
      margin-bottom: var(--spacing-sm);
      color: var(--color-primary);
    }
    
    .navbar-mobile-user span {
      font-family: var(--font-secondary);
      font-weight: 600;
      font-size: 1.2rem;
      color: var(--color-text);
    }
    
    .navbar-mobile-role {
      margin-top: var(--spacing-xs);
      font-size: 0.9rem;
      color: var(--color-text-secondary);
      text-transform: capitalize;
      background-color: rgba(12, 198, 222, 0.1);
      padding: 2px var(--spacing-sm);
      border-radius: var(--border-radius-md);
    }
    
    .navbar-mobile-links {
      display: flex;
      flex-direction: column;
      gap: var(--spacing-md);
    }
    
    .navbar-mobile-link {
      display: flex;
      align-items: center;
      color: var(--color-text);
      text-decoration: none;
      gap: var(--spacing-sm);
      padding: var(--spacing-sm);
      border-radius: var(--border-radius-md);
      transition: all var(--transition-fast);
      font-family: var(--font-secondary);
      font-weight: 500;
      font-size: 1.1rem;
    }
    
    .navbar-mobile-link:hover {
      background-color: rgba(12, 198, 222, 0.1);
    }
    
    .navbar-mobile-link.active {
      color: var(--color-primary);
      background-color: rgba(12, 198, 222, 0.1);
      box-shadow: 0 0 5px rgba(12, 198, 222, 0.3);
    }
    
    .navbar-mobile-icon {
      font-size: 1.2rem;
    }
    
    .navbar-mobile-auth {
      margin-top: auto;
      padding-top: var(--spacing-lg);
      border-top: 1px solid var(--color-border);
    }
    
    .navbar-mobile-logout {
      display: flex;
      align-items: center;
      gap: var(--spacing-sm);
      width: 100%;
      padding: var(--spacing-sm);
      color: var(--color-danger);
      background: none;
      border: 1px solid var(--color-danger);
      border-radius: var(--border-radius-md);
      font-family: var(--font-secondary);
      font-size: 1.1rem;
      cursor: pointer;
      transition: all var(--transition-fast);
    }
    
    .navbar-mobile-logout:hover {
      background-color: rgba(255, 83, 113, 0.1);
    }
    
    .navbar-mobile-login {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: var(--spacing-sm);
      width: 100%;
      padding: var(--spacing-sm);
      color: var(--color-primary);
      background: none;
      border: 1px solid var(--color-primary);
      border-radius: var(--border-radius-md);
      font-family: var(--font-secondary);
      font-size: 1.1rem;
      text-decoration: none;
      transition: all var(--transition-fast);
    }
    
    .navbar-mobile-login:hover {
      background-color: rgba(12, 198, 222, 0.1);
    }
  }
`;

// Insertar estilos en el documento si no existen ya
if (typeof document !== 'undefined') {
  const styleId = 'navbar-component-styles';
  
  if (!document.getElementById(styleId)) {
    const styleElement = document.createElement('style');
    styleElement.id = styleId;
    styleElement.textContent = styles;
    document.head.appendChild(styleElement);
  }
}