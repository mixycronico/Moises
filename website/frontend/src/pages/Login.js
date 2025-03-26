import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import axios from 'axios';
import { FaEye, FaEyeSlash, FaUser, FaLock } from 'react-icons/fa';

const Login = ({ setUser }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!username || !password) {
      setError('Por favor, ingresa tu nombre de usuario y contraseña');
      return;
    }
    
    setLoading(true);
    setError('');
    
    try {
      const response = await axios.post('/api/login', { 
        username, 
        password 
      }, {
        withCredentials: true
      });
      
      setUser(response.data.user);
      
      // Redirigir según el rol
      if (response.data.user.role === 'super_admin') {
        navigate('/super-admin');
      } else if (response.data.user.role === 'admin') {
        navigate('/admin');
      } else {
        navigate('/investor');
      }
    } catch (err) {
      console.error('Error de inicio de sesión:', err);
      setError(err.response?.data?.message || 'Error al iniciar sesión. Inténtalo de nuevo.');
    } finally {
      setLoading(false);
    }
  };

  const togglePasswordVisibility = () => {
    setShowPassword(!showPassword);
  };

  return (
    <div className="login-container">
      <motion.div 
        className="login-card"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <div className="logo-container">
          <motion.div 
            className="logo"
            animate={{ 
              rotate: 360,
            }}
            transition={{ 
              duration: 20, 
              repeat: Infinity, 
              ease: "linear" 
            }}
          >
            G
          </motion.div>
        </div>
        
        <h2>Acceso al Sistema Genesis</h2>
        
        <form onSubmit={handleSubmit}>
          <div className="input-group">
            <div className="icon-container">
              <FaUser />
            </div>
            <input
              type="text"
              placeholder="Nombre de Usuario"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
            />
          </div>
          
          <div className="input-group">
            <div className="icon-container">
              <FaLock />
            </div>
            <input
              type={showPassword ? "text" : "password"}
              placeholder="Contraseña"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
            <button 
              type="button" 
              className="toggle-password"
              onClick={togglePasswordVisibility}
            >
              {showPassword ? <FaEyeSlash /> : <FaEye />}
            </button>
          </div>
          
          {error && <div className="error-message">{error}</div>}
          
          <motion.button
            type="submit"
            className="login-button"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            disabled={loading}
          >
            {loading ? 'Iniciando sesión...' : 'Iniciar Sesión'}
          </motion.button>
        </form>
      </motion.div>
      
      <div className="cosmic-background">
        {/* Elementos visuales de fondo (estrellas, nebulosas, etc.) */}
      </div>
    </div>
  );
};

export default Login;