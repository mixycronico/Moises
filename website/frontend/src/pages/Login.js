import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import axios from 'axios';

const Login = ({ onLogin }) => {
  const [formData, setFormData] = useState({
    username: '',
    password: ''
  });
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  useEffect(() => {
    document.title = 'Iniciar Sesión | Genesis';
  }, []);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    
    // Limpiar error cuando el usuario corrige
    if (error) setError('');
  };

  const togglePasswordVisibility = () => {
    setShowPassword(prev => !prev);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!formData.username || !formData.password) {
      setError('Por favor complete todos los campos');
      return;
    }
    
    setLoading(true);
    setError('');
    
    try {
      const response = await axios.post('/api/login', formData);
      
      if (response.data.success) {
        // Llamar a la función onLogin del componente padre
        onLogin(response.data.user);
        
        // Redirigir según el rol
        const { role } = response.data.user;
        if (role === 'super_admin') {
          navigate('/super-admin');
        } else if (role === 'admin') {
          navigate('/admin');
        } else {
          navigate('/investor');
        }
      }
    } catch (err) {
      if (err.response && err.response.data && err.response.data.message) {
        setError(err.response.data.message);
      } else {
        setError('Error al conectar con el servidor. Por favor, inténtelo de nuevo.');
      }
    } finally {
      setLoading(false);
    }
  };

  // Variantes para animaciones
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: { 
      opacity: 1,
      transition: { 
        when: "beforeChildren",
        staggerChildren: 0.1,
        duration: 0.5
      }
    }
  };
  
  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: { 
      y: 0, 
      opacity: 1,
      transition: { type: "spring", stiffness: 100 }
    }
  };

  // Generar estrellas aleatorias para el fondo
  const generateStars = () => {
    const stars = [];
    for (let i = 0; i < 100; i++) {
      const size = Math.random() * 2;
      stars.push({
        id: i,
        size: size,
        x: Math.random() * 100,
        y: Math.random() * 100,
        animationDuration: 1 + Math.random() * 3
      });
    }
    return stars;
  };

  const stars = generateStars();

  return (
    <div style={{
      minHeight: '100vh',
      overflow: 'hidden',
      position: 'relative',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      backgroundImage: 'linear-gradient(135deg, #0a0e17 0%, #121a29 100%)',
    }}>
      {/* Estrellas de fondo */}
      {stars.map(star => (
        <motion.div
          key={star.id}
          initial={{ opacity: 0.4 }}
          animate={{ opacity: [0.4, 1, 0.4] }}
          transition={{
            repeat: Infinity,
            duration: star.animationDuration,
            ease: "easeInOut"
          }}
          style={{
            position: 'absolute',
            top: `${star.y}%`,
            left: `${star.x}%`,
            width: `${star.size}px`,
            height: `${star.size}px`,
            backgroundColor: '#ffffff',
            borderRadius: '50%',
            boxShadow: '0 0 4px rgba(255, 255, 255, 0.8)',
          }}
        />
      ))}
      
      {/* Logo en la esquina superior izquierda */}
      <Link 
        to="/" 
        style={{
          position: 'absolute',
          top: '2rem',
          left: '2rem',
          fontFamily: 'var(--font-title)',
          fontSize: '1.5rem',
          color: 'var(--primary-color)',
          textDecoration: 'none',
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
          textShadow: 'var(--glow-primary)',
        }}
      >
        <svg 
          width="28" 
          height="28" 
          viewBox="0 0 24 24" 
          fill="none" 
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2ZM12 20C7.59 20 4 16.41 4 12C4 7.59 7.59 4 12 4C16.41 4 20 7.59 20 12C20 16.41 16.41 20 12 20Z"
            fill="url(#gradient)"
          />
          <path
            d="M12 11C13.1046 11 14 10.1046 14 9C14 7.89543 13.1046 7 12 7C10.8954 7 10 7.89543 10 9C10 10.1046 10.8954 11 12 11Z"
            fill="url(#gradient)"
          />
          <path
            d="M12 13C9.79 13 8 14.79 8 17H16C16 14.79 14.21 13 12 13Z"
            fill="url(#gradient)"
          />
          <defs>
            <linearGradient id="gradient" x1="2" y1="12" x2="22" y2="12" gradientUnits="userSpaceOnUse">
              <stop stopColor="#05b2dc" />
              <stop offset="1" stopColor="#8a2be2" />
            </linearGradient>
          </defs>
        </svg>
        GENESIS
      </Link>

      {/* Formulario de inicio de sesión */}
      <motion.div
        className="login-container"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        style={{
          backgroundColor: 'rgba(20, 26, 40, 0.8)',
          backdropFilter: 'blur(10px)',
          borderRadius: 'var(--border-radius-lg)',
          padding: '2.5rem',
          width: '90%',
          maxWidth: '400px',
          boxShadow: '0 10px 30px rgba(0, 0, 0, 0.3)',
          zIndex: 10
        }}
      >
        <motion.div
          variants={itemVariants}
          style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            marginBottom: '2rem'
          }}
        >
          <svg 
            width="60" 
            height="60" 
            viewBox="0 0 24 24" 
            fill="none" 
            xmlns="http://www.w3.org/2000/svg"
            style={{ marginBottom: '1rem' }}
          >
            <path
              d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2ZM12 20C7.59 20 4 16.41 4 12C4 7.59 7.59 4 12 4C16.41 4 20 7.59 20 12C20 16.41 16.41 20 12 20Z"
              fill="url(#gradient-large)"
            />
            <path
              d="M12 11C13.1046 11 14 10.1046 14 9C14 7.89543 13.1046 7 12 7C10.8954 7 10 7.89543 10 9C10 10.1046 10.8954 11 12 11Z"
              fill="url(#gradient-large)"
            />
            <path
              d="M12 13C9.79 13 8 14.79 8 17H16C16 14.79 14.21 13 12 13Z"
              fill="url(#gradient-large)"
            />
            <circle cx="12" cy="12" r="9" stroke="url(#gradient-large)" strokeWidth="0.5" strokeDasharray="1 1" fill="none" />
            <defs>
              <linearGradient id="gradient-large" x1="2" y1="12" x2="22" y2="12" gradientUnits="userSpaceOnUse">
                <stop stopColor="#05b2dc" />
                <stop offset="1" stopColor="#8a2be2" />
              </linearGradient>
            </defs>
          </svg>
          <h1 style={{ 
            textAlign: 'center',
            fontSize: '1.8rem',
            marginBottom: '0.5rem',
            background: 'linear-gradient(135deg, var(--primary-color), var(--secondary-color))',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent'
          }}>
            Iniciar Sesión
          </h1>
          <p style={{ 
            color: 'var(--text-secondary)',
            textAlign: 'center',
            fontSize: '0.9rem'
          }}>
            Accede a tu cuenta Genesis para continuar
          </p>
        </motion.div>

        <motion.form 
          variants={containerVariants}
          onSubmit={handleSubmit}
        >
          {/* Campo de usuario */}
          <motion.div 
            variants={itemVariants}
            className="form-group"
          >
            <label 
              htmlFor="username"
              className="form-label"
            >
              Usuario
            </label>
            <input
              type="text"
              id="username"
              name="username"
              className="form-input"
              value={formData.username}
              onChange={handleChange}
              autoComplete="username"
              placeholder="Ingresa tu nombre de usuario"
              style={{
                backgroundColor: 'rgba(10, 14, 23, 0.5)',
                border: '1px solid rgba(5, 178, 220, 0.3)',
                borderRadius: 'var(--border-radius-sm)',
                padding: '0.8rem',
                color: 'var(--text-color)',
                width: '100%',
                transition: 'all 0.3s ease'
              }}
            />
          </motion.div>

          {/* Campo de contraseña */}
          <motion.div 
            variants={itemVariants}
            className="form-group"
            style={{ position: 'relative' }}
          >
            <label 
              htmlFor="password"
              className="form-label"
            >
              Contraseña
            </label>
            <div style={{ position: 'relative' }}>
              <input
                type={showPassword ? "text" : "password"}
                id="password"
                name="password"
                className="form-input"
                value={formData.password}
                onChange={handleChange}
                autoComplete="current-password"
                placeholder="Ingresa tu contraseña"
                style={{
                  backgroundColor: 'rgba(10, 14, 23, 0.5)',
                  border: '1px solid rgba(5, 178, 220, 0.3)',
                  borderRadius: 'var(--border-radius-sm)',
                  padding: '0.8rem',
                  color: 'var(--text-color)',
                  width: '100%',
                  paddingRight: '2.5rem',
                  transition: 'all 0.3s ease'
                }}
              />
              <button
                type="button"
                onClick={togglePasswordVisibility}
                style={{
                  position: 'absolute',
                  right: '0.5rem',
                  top: '50%',
                  transform: 'translateY(-50%)',
                  background: 'transparent',
                  border: 'none',
                  cursor: 'pointer',
                  color: 'var(--text-secondary)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  padding: '0.5rem'
                }}
              >
                {showPassword ? (
                  <svg width="20" height="20" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12 4.5C7 4.5 2.73 7.61 1 12c1.73 4.39 6 7.5 11 7.5s9.27-3.11 11-7.5c-1.73-4.39-6-7.5-11-7.5zM12 17c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5zm0-8c-1.66 0-3 1.34-3 3s1.34 3 3 3 3-1.34 3-3-1.34-3-3-3z" />
                  </svg>
                ) : (
                  <svg width="20" height="20" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12 7c2.76 0 5 2.24 5 5 0 .65-.13 1.26-.36 1.83l2.92 2.92c1.51-1.26 2.7-2.89 3.43-4.75-1.73-4.39-6-7.5-11-7.5-1.4 0-2.74.25-3.98.7l2.16 2.16C10.74 7.13 11.35 7 12 7zM2 4.27l2.28 2.28.46.46C3.08 8.3 1.78 10.02 1 12c1.73 4.39 6 7.5 11 7.5 1.55 0 3.03-.3 4.38-.84l.42.42L19.73 22 21 20.73 3.27 3 2 4.27zM7.53 9.8l1.55 1.55c-.05.21-.08.43-.08.65 0 1.66 1.34 3 3 3 .22 0 .44-.03.65-.08l1.55 1.55c-.67.33-1.41.53-2.2.53-2.76 0-5-2.24-5-5 0-.79.2-1.53.53-2.2zm4.31-.78l3.15 3.15.02-.16c0-1.66-1.34-3-3-3l-.17.01z" />
                  </svg>
                )}
              </button>
            </div>
          </motion.div>

          {/* Mensaje de error */}
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              style={{
                color: 'var(--danger-color)',
                textAlign: 'center',
                marginBottom: '1rem',
                padding: '0.5rem',
                borderRadius: 'var(--border-radius-sm)',
                backgroundColor: 'rgba(255, 82, 82, 0.1)',
                border: '1px solid rgba(255, 82, 82, 0.3)'
              }}
            >
              {error}
            </motion.div>
          )}

          {/* Botón de inicio de sesión */}
          <motion.button
            variants={itemVariants}
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.97 }}
            type="submit"
            disabled={loading}
            style={{
              width: '100%',
              padding: '0.8rem',
              backgroundColor: 'var(--primary-color)',
              color: 'var(--text-color)',
              border: 'none',
              borderRadius: 'var(--border-radius-sm)',
              fontFamily: 'var(--font-title)',
              fontSize: '1rem',
              fontWeight: 'bold',
              cursor: loading ? 'not-allowed' : 'pointer',
              opacity: loading ? 0.8 : 1,
              transition: 'all 0.3s ease',
              marginTop: '1rem',
              boxShadow: 'var(--glow-primary)',
              position: 'relative',
              overflow: 'hidden'
            }}
          >
            {loading ? 'Iniciando sesión...' : 'Iniciar Sesión'}
            
            {/* Efecto de carga */}
            {loading && (
              <motion.div
                style={{
                  position: 'absolute',
                  bottom: 0,
                  left: 0,
                  height: '4px',
                  backgroundColor: 'rgba(255, 255, 255, 0.5)',
                  width: '100%',
                  transformOrigin: 'left'
                }}
                animate={{
                  scaleX: [0, 1, 0],
                  x: ['0%', '0%', '100%']
                }}
                transition={{
                  repeat: Infinity,
                  duration: 1.5,
                  ease: "easeInOut"
                }}
              />
            )}
          </motion.button>
        </motion.form>
        
        {/* Información adicional */}
        <motion.div
          variants={itemVariants}
          style={{
            marginTop: '2rem',
            textAlign: 'center'
          }}
        >
          <p style={{ 
            color: 'var(--text-secondary)',
            fontSize: '0.85rem',
            marginBottom: '0.5rem'
          }}>
            Credenciales de prueba:
          </p>
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: '0.3rem',
            fontSize: '0.8rem',
            color: 'var(--text-secondary)',
            backgroundColor: 'rgba(10, 14, 23, 0.5)',
            borderRadius: 'var(--border-radius-sm)',
            padding: '0.8rem',
            marginTop: '0.5rem'
          }}>
            <div>
              <span style={{ color: 'var(--primary-color)' }}>Inversor:</span> investor / investor_password
            </div>
            <div>
              <span style={{ color: 'var(--primary-color)' }}>Admin:</span> admin / admin_password
            </div>
            <div>
              <span style={{ color: 'var(--primary-color)' }}>Super Admin:</span> super_admin / super_admin_password
            </div>
          </div>
        </motion.div>
      </motion.div>
    </div>
  );
};

export default Login;