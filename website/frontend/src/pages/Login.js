import React, { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { FaUserAlt, FaLock, FaEye, FaEyeSlash, FaExclamationTriangle } from 'react-icons/fa';
import { useAuth } from '../utils/AuthContext';

/**
 * Página de inicio de sesión con animaciones y toggle de visibilidad de contraseña
 */
const Login = () => {
  // Estados para formulario
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  
  // Hook de navegación y autenticación
  const navigate = useNavigate();
  const { login, user } = useAuth();
  
  // Redirigir si ya está autenticado
  useEffect(() => {
    if (user) {
      // Determinar ruta según rol
      let redirectPath = '/';
      
      if (user.role === 'investor') {
        redirectPath = '/investor';
      } else if (user.role === 'admin') {
        redirectPath = '/admin';
      } else if (user.role === 'super_admin') {
        redirectPath = '/super-admin';
      }
      
      navigate(redirectPath);
    }
  }, [user, navigate]);

  // Manejar envío del formulario
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Validar campos
    if (!username.trim() || !password.trim()) {
      setError('Por favor completa todos los campos');
      return;
    }
    
    // Resetear error
    setError('');
    setIsLoading(true);
    
    try {
      // Intentar inicio de sesión
      const result = await login(username, password);
      
      if (!result.success) {
        setError(result.message || 'Credenciales incorrectas');
      }
    } catch (err) {
      console.error('Error al iniciar sesión:', err);
      setError('Error al conectar con el servidor');
    } finally {
      setIsLoading(false);
    }
  };

  // Variantes de animación
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        duration: 0.5
      }
    },
    exit: {
      opacity: 0,
      transition: {
        duration: 0.3
      }
    }
  };
  
  // Animación para campos del formulario
  const inputVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: (custom) => ({
      y: 0,
      opacity: 1,
      transition: {
        delay: custom * 0.1,
        duration: 0.4
      }
    })
  };

  return (
    <div className="login-page">
      {/* Contenedor principal */}
      <motion.div 
        className="login-container"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        exit="exit"
      >
        {/* Logo */}
        <motion.div
          className="login-logo"
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.5 }}
        >
          <span>G</span>
        </motion.div>
        
        <motion.h1
          className="login-title"
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.2, duration: 0.4 }}
        >
          Acceso al Sistema
        </motion.h1>
        
        {/* Mensaje de error */}
        {error && (
          <motion.div
            className="login-error"
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            <FaExclamationTriangle />
            <span>{error}</span>
          </motion.div>
        )}
        
        {/* Formulario */}
        <form className="login-form" onSubmit={handleSubmit}>
          {/* Campo de usuario */}
          <motion.div
            className="login-field"
            variants={inputVariants}
            custom={1}
            initial="hidden"
            animate="visible"
          >
            <div className="login-icon">
              <FaUserAlt />
            </div>
            <div className="login-input-container">
              <input
                type="text"
                id="username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Usuario"
                required
                autoComplete="username"
                className="login-input"
                disabled={isLoading}
              />
              <label htmlFor="username" className="login-label">Usuario</label>
            </div>
          </motion.div>
          
          {/* Campo de contraseña */}
          <motion.div
            className="login-field"
            variants={inputVariants}
            custom={2}
            initial="hidden"
            animate="visible"
          >
            <div className="login-icon">
              <FaLock />
            </div>
            <div className="login-input-container">
              <input
                type={showPassword ? "text" : "password"}
                id="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Contraseña"
                required
                autoComplete="current-password"
                className="login-input"
                disabled={isLoading}
              />
              <label htmlFor="password" className="login-label">Contraseña</label>
              <button
                type="button"
                className="login-password-toggle"
                onClick={() => setShowPassword(!showPassword)}
                aria-label={showPassword ? "Ocultar contraseña" : "Mostrar contraseña"}
              >
                {showPassword ? <FaEyeSlash /> : <FaEye />}
              </button>
            </div>
          </motion.div>
          
          {/* Botón de envío */}
          <motion.button
            className={`login-button ${isLoading ? 'loading' : ''}`}
            variants={inputVariants}
            custom={3}
            initial="hidden"
            animate="visible"
            type="submit"
            disabled={isLoading}
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.98 }}
          >
            {isLoading ? (
              <div className="login-spinner"></div>
            ) : 'Ingresar'}
          </motion.button>
        </form>
        
        {/* Credenciales de prueba */}
        <motion.div
          className="login-demo-accounts"
          variants={inputVariants}
          custom={4}
          initial="hidden"
          animate="visible"
        >
          <h3>Cuentas de Prueba</h3>
          <div className="demo-account">
            <strong>Inversor:</strong> usuario: investor, contraseña: investor_password
          </div>
          <div className="demo-account">
            <strong>Admin:</strong> usuario: admin, contraseña: admin_password
          </div>
          <div className="demo-account">
            <strong>Super Admin:</strong> usuario: super_admin, contraseña: super_admin_password
          </div>
        </motion.div>
        
        {/* Enlace a la página principal */}
        <motion.div
          className="login-home-link"
          variants={inputVariants}
          custom={5}
          initial="hidden"
          animate="visible"
        >
          <Link to="/">Volver a la página principal</Link>
        </motion.div>
      </motion.div>
    </div>
  );
};

export default Login;

// Estilos específicos para la página Login
const styles = `
  .login-page {
    min-height: 100vh;
    display: flex;
    justify-content: center;
    align-items: center;
    padding: var(--spacing-lg);
    background-color: var(--color-background);
    position: relative;
    overflow: hidden;
  }
  
  .login-page::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: radial-gradient(circle at 30% 30%, rgba(12, 198, 222, 0.1), transparent 70%),
                radial-gradient(circle at 70% 70%, rgba(146, 112, 255, 0.1), transparent 70%);
    z-index: 0;
  }
  
  .login-container {
    width: 100%;
    max-width: 500px;
    padding: var(--spacing-xl);
    background: rgba(22, 43, 77, 0.7);
    backdrop-filter: blur(10px);
    border-radius: var(--border-radius-lg);
    border: 1px solid var(--color-border);
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3);
    position: relative;
    z-index: 1;
  }
  
  .login-logo {
    width: 80px;
    height: 80px;
    border-radius: 50%;
    border: 3px solid var(--color-primary);
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto var(--spacing-lg);
    box-shadow: 0 0 20px rgba(12, 198, 222, 0.5);
    position: relative;
    overflow: hidden;
  }
  
  .login-logo::after {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 200%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
    animation: shimmer 3s infinite;
  }
  
  .login-logo span {
    font-family: var(--font-display);
    font-size: 40px;
    font-weight: bold;
    color: var(--color-primary);
  }
  
  .login-title {
    text-align: center;
    font-size: 2rem;
    margin-bottom: var(--spacing-lg);
    color: var(--color-primary);
    font-family: var(--font-display);
  }
  
  .login-form {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-md);
    margin-bottom: var(--spacing-lg);
  }
  
  .login-field {
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
  }
  
  .login-icon {
    width: 40px;
    height: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(12, 198, 222, 0.1);
    border-radius: var(--border-radius-sm);
    color: var(--color-primary);
  }
  
  .login-input-container {
    flex: 1;
    position: relative;
  }
  
  .login-input {
    width: 100%;
    height: 50px;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid var(--color-border);
    border-radius: var(--border-radius-sm);
    padding: 0 var(--spacing-xl) 0 var(--spacing-md);
    color: var(--color-text);
    font-size: 1rem;
    transition: all var(--transition-normal);
  }
  
  .login-input:focus {
    outline: none;
    border-color: var(--color-primary);
    box-shadow: 0 0 10px rgba(12, 198, 222, 0.3);
  }
  
  .login-input:focus + .login-label,
  .login-input:not(:placeholder-shown) + .login-label {
    transform: translateY(-30px) scale(0.8);
    color: var(--color-primary);
  }
  
  .login-label {
    position: absolute;
    left: var(--spacing-md);
    top: 15px;
    color: var(--color-text-secondary);
    transition: all var(--transition-normal);
    pointer-events: none;
    font-family: var(--font-secondary);
  }
  
  .login-password-toggle {
    position: absolute;
    right: var(--spacing-sm);
    top: 50%;
    transform: translateY(-50%);
    background: none;
    border: none;
    color: var(--color-text-secondary);
    cursor: pointer;
    padding: var(--spacing-xs);
    display: flex;
    align-items: center;
    justify-content: center;
    transition: color var(--transition-fast);
  }
  
  .login-password-toggle:hover {
    color: var(--color-primary);
  }
  
  .login-button {
    height: 50px;
    background: linear-gradient(45deg, var(--color-primary), var(--color-secondary));
    border: none;
    border-radius: var(--border-radius-sm);
    color: var(--color-background);
    font-family: var(--font-display);
    font-size: 1.1rem;
    font-weight: 500;
    text-transform: uppercase;
    cursor: pointer;
    margin-top: var(--spacing-sm);
    position: relative;
    overflow: hidden;
    transition: all var(--transition-normal);
  }
  
  .login-button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
    transition: all var(--transition-normal);
  }
  
  .login-button:hover::before {
    animation: shimmer 2s infinite;
  }
  
  .login-button:disabled {
    opacity: 0.7;
    cursor: not-allowed;
  }
  
  .login-button.loading {
    color: transparent;
  }
  
  .login-spinner {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 24px;
    height: 24px;
    border: 3px solid rgba(255, 255, 255, 0.3);
    border-radius: 50%;
    border-top-color: #fff;
    animation: spin 1s ease-in-out infinite;
  }
  
  @keyframes spin {
    to {
      transform: translate(-50%, -50%) rotate(360deg);
    }
  }
  
  .login-error {
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
    padding: var(--spacing-sm) var(--spacing-md);
    background: rgba(255, 83, 113, 0.1);
    border-left: 3px solid var(--color-danger);
    border-radius: var(--border-radius-sm);
    color: var(--color-danger);
    margin-bottom: var(--spacing-md);
  }
  
  .login-home-link {
    text-align: center;
    margin-top: var(--spacing-md);
  }
  
  .login-home-link a {
    color: var(--color-text-secondary);
    text-decoration: none;
    transition: all var(--transition-fast);
  }
  
  .login-home-link a:hover {
    color: var(--color-primary);
    text-decoration: underline;
  }
  
  .login-demo-accounts {
    margin-top: var(--spacing-lg);
    padding: var(--spacing-md);
    background: rgba(12, 198, 222, 0.05);
    border-radius: var(--border-radius-md);
    border: 1px dashed rgba(12, 198, 222, 0.3);
  }
  
  .login-demo-accounts h3 {
    text-align: center;
    font-size: 1rem;
    margin-bottom: var(--spacing-sm);
    color: var(--color-primary);
  }
  
  .demo-account {
    font-size: 0.8rem;
    margin-bottom: var(--spacing-xs);
    color: var(--color-text-secondary);
  }
  
  .demo-account strong {
    color: var(--color-text);
  }
  
  /* Responsive */
  @media (max-width: 480px) {
    .login-container {
      padding: var(--spacing-md);
    }
    
    .login-title {
      font-size: 1.5rem;
    }
    
    .login-input {
      height: 45px;
    }
    
    .login-button {
      height: 45px;
    }
  }
`;

// Insertar estilos en el documento si no existen ya
if (typeof document !== 'undefined') {
  const styleId = 'login-page-styles';
  
  if (!document.getElementById(styleId)) {
    const styleElement = document.createElement('style');
    styleElement.id = styleId;
    styleElement.textContent = styles;
    document.head.appendChild(styleElement);
  }
}