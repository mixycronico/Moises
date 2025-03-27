import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import axios from 'axios';

const Login = ({ setUser }) => {
  const [formData, setFormData] = useState({
    username: '',
    password: '',
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const response = await axios.post('/api/auth/login', formData);
      
      if (response.data.success) {
        setUser(response.data.user);
        navigate('/dashboard');
      } else {
        setError(response.data.message || 'Ha ocurrido un error durante el inicio de sesión.');
      }
    } catch (error) {
      console.error('Login error:', error);
      setError(
        error.response?.data?.message || 
        'No se pudo conectar con el servidor. Por favor, intenta de nuevo más tarde.'
      );
    } finally {
      setLoading(false);
    }
  };

  // Animaciones
  const pageVariants = {
    initial: { opacity: 0 },
    in: { 
      opacity: 1,
      transition: { duration: 0.6, ease: "easeOut" }
    },
    out: { 
      opacity: 0,
      transition: { duration: 0.4, ease: "easeIn" }
    }
  };

  const formVariants = {
    initial: { y: 20, opacity: 0 },
    in: { 
      y: 0, 
      opacity: 1,
      transition: { delay: 0.2, duration: 0.5, ease: "easeOut" }
    }
  };

  const starryBackground = {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    overflow: 'hidden',
    zIndex: -1,
  };

  return (
    <motion.div 
      className="flex flex-col min-h-screen bg-cosmic-gradient"
      initial="initial"
      animate="in"
      exit="out"
      variants={pageVariants}
    >
      {/* Fondo con estrellas */}
      <div style={starryBackground}>
        <div className="stars-container">
          {/* Las estrellas se generarán con CSS */}
        </div>
      </div>

      {/* Contenido de la página */}
      <div className="flex-1 flex flex-col justify-center items-center p-4 relative">
        <motion.div
          className="w-full max-w-md"
          variants={formVariants}
        >
          {/* Logo y nombre del sistema */}
          <div className="text-center mb-8">
            <h1 className="text-4xl font-display cosmic-gradient-text mb-2">
              Sistema Genesis
            </h1>
            <p className="text-cosmic-glow">
              Plataforma avanzada de trading con IA consciente
            </p>
          </div>

          {/* Tarjeta de login */}
          <div className="cosmic-card p-6 backdrop-blur-xl">
            <h2 className="text-xl font-semibold mb-6 text-center">
              Iniciar Sesión
            </h2>
            
            {/* Mensaje de error */}
            {error && (
              <div className="bg-cosmic-red/10 border border-cosmic-red/30 text-cosmic-red rounded-md p-3 mb-4 text-sm">
                {error}
              </div>
            )}
            
            {/* Formulario */}
            <form onSubmit={handleSubmit}>
              <div className="mb-4">
                <label htmlFor="username" className="block text-sm font-medium mb-1">
                  Usuario
                </label>
                <input
                  id="username"
                  name="username"
                  type="text"
                  required
                  value={formData.username}
                  onChange={handleChange}
                  className="cosmic-input"
                  placeholder="Ingresa tu nombre de usuario"
                  disabled={loading}
                />
              </div>
              
              <div className="mb-6">
                <label htmlFor="password" className="block text-sm font-medium mb-1">
                  Contraseña
                </label>
                <input
                  id="password"
                  name="password"
                  type="password"
                  required
                  value={formData.password}
                  onChange={handleChange}
                  className="cosmic-input"
                  placeholder="Ingresa tu contraseña"
                  disabled={loading}
                />
              </div>
              
              <button
                type="submit"
                className="w-full cosmic-button py-2.5"
                disabled={loading}
              >
                {loading ? (
                  <div className="flex justify-center items-center">
                    <div className="spinner h-5 w-5 border-2 border-transparent border-t-white rounded-full animate-spin"></div>
                    <span className="ml-2">Iniciando sesión...</span>
                  </div>
                ) : (
                  'Iniciar Sesión'
                )}
              </button>
              
              <div className="mt-4 text-center">
                <Link to="/forgot-password" className="text-sm text-cosmic-blue hover:text-cosmic-glow transition-colors">
                  ¿Olvidaste tu contraseña?
                </Link>
              </div>
            </form>
          </div>
          
          {/* Enlace para crear cuenta */}
          <div className="text-center mt-4">
            <p className="text-sm text-gray-400">
              ¿No tienes una cuenta?{' '}
              <Link to="/register" className="text-cosmic-blue hover:text-cosmic-glow transition-colors">
                Contacta al administrador
              </Link>
            </p>
          </div>
        </motion.div>
      </div>
      
      {/* Footer */}
      <footer className="py-4 text-center text-sm text-gray-500">
        <p>Sistema Genesis v4.4 — Quantum Ultra Divino</p>
        <p className="text-xs mt-1">© 2025 Todos los derechos reservados</p>
      </footer>
    </motion.div>
  );
};

export default Login;