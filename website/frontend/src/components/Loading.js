import React from 'react';
import { motion } from 'framer-motion';

/**
 * Componente de carga con animación y estilo cósmico
 * 
 * @param {Object} props - Propiedades del componente
 * @param {string} [props.message='Cargando...'] - Mensaje a mostrar durante la carga
 * @returns {JSX.Element}
 */
const Loading = ({ message = 'Cargando...' }) => {
  return (
    <div className="loading-container">
      <motion.div
        className="loading-content"
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5 }}
      >
        <div className="logo-container">
          <svg 
            className="genesis-logo" 
            viewBox="0 0 100 100" 
            xmlns="http://www.w3.org/2000/svg"
          >
            <defs>
              <linearGradient id="logoGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#0cc6de" />
                <stop offset="100%" stopColor="#9270ff" />
              </linearGradient>
              <filter id="glow">
                <feGaussianBlur stdDeviation="2.5" result="blur" />
                <feComposite in="SourceGraphic" in2="blur" operator="over" />
              </filter>
            </defs>
            <g filter="url(#glow)">
              <path 
                d="M50,10 A40,40 0 1,0 50,90 A40,40 0 1,0 50,10 Z M50,20 A30,30 0 1,1 50,80 A30,30 0 1,1 50,20 Z" 
                fill="none" 
                stroke="url(#logoGradient)" 
                strokeWidth="2"
              />
              <path 
                d="M50,10 L50,20 M50,80 L50,90 M10,50 L20,50 M80,50 L90,50" 
                stroke="url(#logoGradient)" 
                strokeWidth="2"
              />
              <path 
                d="M30,50 L70,50 M50,30 L50,70" 
                stroke="url(#logoGradient)" 
                strokeWidth="4"
              />
              <circle 
                cx="50" 
                cy="50" 
                r="8" 
                fill="url(#logoGradient)"
              />
            </g>
          </svg>
        </div>
        
        <div className="spinner">
          <motion.div 
            className="spinner-ring"
            animate={{ 
              rotate: 360,
            }}
            transition={{ 
              duration: 2,
              ease: "linear",
              repeat: Infinity,
              repeatType: "loop"
            }}
          ></motion.div>
          <motion.div 
            className="spinner-ring"
            animate={{ 
              rotate: -360,
            }}
            transition={{ 
              duration: 3,
              ease: "linear",
              repeat: Infinity,
              repeatType: "loop"
            }}
          ></motion.div>
        </div>
        
        <motion.div
          className="loading-message"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3, duration: 0.5 }}
        >
          {message}
        </motion.div>
        
        <div className="loading-particles">
          {[...Array(20)].map((_, i) => (
            <motion.div
              key={i}
              className="loading-particle"
              initial={{ 
                x: 0,
                y: 0,
                scale: Math.random() * 0.5 + 0.5,
                opacity: 0
              }}
              animate={{ 
                x: (Math.random() - 0.5) * 200,
                y: (Math.random() - 0.5) * 200,
                opacity: [0, Math.random() * 0.5 + 0.3, 0]
              }}
              transition={{ 
                duration: Math.random() * 3 + 2,
                repeat: Infinity,
                repeatType: "loop",
                delay: Math.random() * 2
              }}
            />
          ))}
        </div>
      </motion.div>
    </div>
  );
};

// Estilos específicos para el componente de carga
const styles = `
  .loading-container {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100vh;
    display: flex;
    justify-content: center;
    align-items: center;
    background-color: var(--color-background);
    z-index: 1000;
  }
  
  .loading-content {
    position: relative;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: var(--spacing-xl);
    max-width: 400px;
  }
  
  .logo-container {
    margin-bottom: var(--spacing-lg);
  }
  
  .genesis-logo {
    width: 80px;
    height: 80px;
    animation: pulse 2s infinite;
  }
  
  .spinner {
    position: relative;
    width: 100px;
    height: 100px;
    margin-bottom: var(--spacing-lg);
  }
  
  .spinner-ring {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    border: 2px solid transparent;
    border-radius: 50%;
  }
  
  .spinner-ring:nth-child(1) {
    border-top-color: var(--color-primary);
    border-left-color: var(--color-primary);
  }
  
  .spinner-ring:nth-child(2) {
    width: 80%;
    height: 80%;
    top: 10%;
    left: 10%;
    border-bottom-color: var(--color-secondary);
    border-right-color: var(--color-secondary);
  }
  
  .loading-message {
    font-family: var(--font-display);
    color: var(--color-primary);
    font-size: 1.2rem;
    text-align: center;
    margin-top: var(--spacing-md);
  }
  
  .loading-particles {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    overflow: hidden;
    z-index: -1;
  }
  
  .loading-particle {
    position: absolute;
    top: 50%;
    left: 50%;
    width: 6px;
    height: 6px;
    background: linear-gradient(to right, var(--color-primary), var(--color-secondary));
    border-radius: var(--border-radius-circle);
    box-shadow: 0 0 10px rgba(12, 198, 222, 0.8);
  }
  
  @keyframes pulse {
    0%, 100% {
      transform: scale(1);
      opacity: 1;
    }
    50% {
      transform: scale(1.05);
      opacity: 0.8;
    }
  }
`;

// Insertar estilos en el documento si no existen ya
if (typeof document !== 'undefined') {
  const styleId = 'loading-component-styles';
  
  if (!document.getElementById(styleId)) {
    const styleElement = document.createElement('style');
    styleElement.id = styleId;
    styleElement.textContent = styles;
    document.head.appendChild(styleElement);
  }
}

export default Loading;