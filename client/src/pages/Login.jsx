import { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import gsap from 'gsap';
import logoGenesis from '../assets/logo-genesis.svg';

const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  // Comprobar si ya hay sesión activa
  useEffect(() => {
    const token = localStorage.getItem('userToken');
    if (token) {
      navigate('/dashboard');
    }
  }, [navigate]);

  // Efectos visuales y animaciones
  useEffect(() => {
    // Animación del logo
    gsap.fromTo(
      '.login-logo',
      { rotation: 0 },
      { rotation: 360, duration: 20, repeat: -1, ease: 'linear' }
    );
    
    // Animación del formulario
    gsap.fromTo(
      '.login-form',
      { opacity: 0, y: 20 },
      { opacity: 1, y: 0, duration: 0.8, ease: 'power2.out' }
    );
    
    // Animación de partículas
    const particles = document.querySelectorAll('.particle');
    particles.forEach((particle) => {
      gsap.to(particle, {
        x: `random(-100, 100)`,
        y: `random(-100, 100)`,
        opacity: `random(0.2, 0.8)`,
        duration: `random(5, 15)`,
        repeat: -1,
        yoyo: true,
        ease: 'none',
      });
    });
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);
    
    try {
      // Intentar iniciar sesión
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email, password }),
      });
      
      const data = await response.json();
      
      if (data.success) {
        // Guardar token y datos de usuario
        localStorage.setItem('userToken', data.token);
        localStorage.setItem('userData', JSON.stringify(data.user));
        
        // Animación de éxito antes de redirigir
        gsap.to('.login-form', {
          y: -20,
          opacity: 0,
          duration: 0.5,
          onComplete: () => {
            navigate('/dashboard');
          }
        });
      } else {
        setError(data.message || 'Credenciales inválidas');
        
        // Animación de error
        gsap.fromTo(
          '.error-message',
          { opacity: 0, y: -10 },
          { opacity: 1, y: 0, duration: 0.3 }
        );
      }
    } catch (error) {
      console.error('Error de inicio de sesión:', error);
      setError('Error de conexión. Intenta de nuevo más tarde.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-light to-primary-dark flex items-center justify-center p-4 relative overflow-hidden">
      {/* Partículas de fondo */}
      {Array.from({ length: 20 }).map((_, i) => (
        <div
          key={i}
          className="particle absolute w-2 h-2 bg-secondary rounded-full opacity-50"
          style={{
            top: `${Math.random() * 100}%`,
            left: `${Math.random() * 100}%`,
          }}
        />
      ))}
      
      <div className="login-container max-w-md w-full p-6 bg-white dark:bg-gray-800 bg-opacity-10 dark:bg-opacity-10 backdrop-blur-lg rounded-xl shadow-xl border border-white border-opacity-20">
        <div className="text-center mb-8">
          <img
            src={logoGenesis}
            alt="Genesis Logo"
            className="login-logo w-24 h-24 mx-auto mb-4"
          />
          <h1 className="text-3xl font-bold text-white mb-2">Genesis</h1>
          <p className="text-gray-300">
            Sistema avanzado de trading con inteligencia cósmica
          </p>
        </div>
        
        <form onSubmit={handleSubmit} className="login-form space-y-6">
          {error && (
            <div className="error-message bg-red-500 bg-opacity-20 text-red-100 p-3 rounded-lg text-center">
              {error}
            </div>
          )}
          
          <div>
            <label htmlFor="email" className="block text-gray-300 mb-2">
              Correo electrónico
            </label>
            <input
              id="email"
              type="email"
              className="w-full px-4 py-3 bg-white bg-opacity-10 rounded-lg focus:outline-none focus:ring-2 focus:ring-secondary border border-gray-300 border-opacity-10 text-white"
              placeholder="correo@ejemplo.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </div>
          
          <div>
            <label htmlFor="password" className="block text-gray-300 mb-2">
              Contraseña
            </label>
            <input
              id="password"
              type="password"
              className="w-full px-4 py-3 bg-white bg-opacity-10 rounded-lg focus:outline-none focus:ring-2 focus:ring-secondary border border-gray-300 border-opacity-10 text-white"
              placeholder="••••••••"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>
          
          <div className="text-right">
            <a href="#" className="text-secondary hover:text-secondary-light text-sm">
              ¿Olvidaste tu contraseña?
            </a>
          </div>
          
          <button
            type="submit"
            className={`w-full py-3 bg-secondary hover:bg-secondary-light text-white rounded-lg font-medium transition-colors duration-300 focus:outline-none focus:ring-2 focus:ring-secondary focus:ring-offset-2 ${
              isLoading ? 'opacity-75 cursor-not-allowed' : ''
            }`}
            disabled={isLoading}
          >
            {isLoading ? (
              <span className="flex items-center justify-center">
                <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Iniciando sesión...
              </span>
            ) : (
              'Iniciar Sesión'
            )}
          </button>
        </form>
        
        <div className="mt-8 text-center text-gray-300 text-sm">
          Sistema Genesis v4.0 - Trascendental
          <br />
          <span className="opacity-70">© 2025 Genesis Trading System</span>
        </div>
      </div>
    </div>
  );
};

export default Login;