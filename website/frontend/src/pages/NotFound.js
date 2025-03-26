import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { FaHome, FaExclamationTriangle } from 'react-icons/fa';

/**
 * Página 404 con animación y estilo cósmico
 */
const NotFound = () => {
  return (
    <div className="not-found-page">
      <div className="not-found-container">
        <motion.div
          className="not-found-content"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <div className="not-found-icon">
            <FaExclamationTriangle />
          </div>
          
          <motion.h1
            className="not-found-title"
            initial={{ scale: 0.8 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.2, duration: 0.5 }}
          >
            404
          </motion.h1>
          
          <motion.h2
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3, duration: 0.5 }}
          >
            Anomalía Dimensional Detectada
          </motion.h2>
          
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.4, duration: 0.5 }}
          >
            La ruta que intentas explorar parece existir en otra dimensión. 
            Nuestros sensores no pueden localizar este recurso en el continuo espacio-tiempo.
          </motion.p>
          
          <motion.div
            className="not-found-actions"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5, duration: 0.5 }}
          >
            <Link to="/" className="not-found-button">
              <FaHome />
              <span>Volver al Origen</span>
            </Link>
          </motion.div>
        </motion.div>
      </div>
      
      {/* Estrellas flotantes para efecto cósmico */}
      <div className="floating-stars">
        {[...Array(20)].map((_, i) => (
          <motion.div
            key={i}
            className="floating-star"
            initial={{ 
              x: Math.random() * window.innerWidth, 
              y: Math.random() * window.innerHeight, 
              scale: Math.random() * 0.5 + 0.5 
            }}
            animate={{ 
              x: Math.random() * window.innerWidth, 
              y: Math.random() * window.innerHeight,
              opacity: [0.2, 0.8, 0.2],
            }}
            transition={{ 
              duration: Math.random() * 20 + 10, 
              repeat: Infinity, 
              repeatType: "reverse"
            }}
          />
        ))}
      </div>
    </div>
  );
};

export default NotFound;

// Estilos específicos para la página NotFound
const styles = `
  .not-found-page {
    min-height: 100vh;
    display: flex;
    justify-content: center;
    align-items: center;
    background-color: var(--color-background);
    position: relative;
    overflow: hidden;
  }
  
  .not-found-container {
    width: 100%;
    max-width: 800px;
    padding: var(--spacing-xl);
    z-index: 1;
  }
  
  .not-found-content {
    background: rgba(22, 43, 77, 0.7);
    backdrop-filter: blur(10px);
    border-radius: var(--border-radius-lg);
    border: 1px solid var(--color-border);
    padding: var(--spacing-xl);
    text-align: center;
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3);
  }
  
  .not-found-icon {
    font-size: 4rem;
    color: var(--color-danger);
    margin-bottom: var(--spacing-md);
    animation: pulse 2s infinite;
  }
  
  .not-found-title {
    font-size: 8rem;
    font-family: var(--font-display);
    background: linear-gradient(45deg, var(--color-primary), var(--color-secondary));
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    margin-bottom: var(--spacing-sm);
    line-height: 1;
    text-shadow: 0 0 10px rgba(12, 198, 222, 0.5);
  }
  
  .not-found-content h2 {
    color: var(--color-primary);
    font-family: var(--font-display);
    font-size: 2rem;
    margin-bottom: var(--spacing-md);
  }
  
  .not-found-content p {
    color: var(--color-text);
    font-size: 1.2rem;
    margin-bottom: var(--spacing-lg);
    max-width: 600px;
    margin-left: auto;
    margin-right: auto;
  }
  
  .not-found-actions {
    margin-top: var(--spacing-lg);
  }
  
  .not-found-button {
    display: inline-flex;
    align-items: center;
    gap: var(--spacing-sm);
    padding: var(--spacing-sm) var(--spacing-lg);
    background: linear-gradient(45deg, var(--color-primary), var(--color-secondary));
    color: var(--color-background);
    border-radius: var(--border-radius-md);
    text-decoration: none;
    font-family: var(--font-display);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1px;
    box-shadow: 0 4px 15px rgba(12, 198, 222, 0.4);
    transition: all var(--transition-normal);
  }
  
  .not-found-button:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 25px rgba(12, 198, 222, 0.6);
  }
  
  .floating-stars {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: 0;
  }
  
  .floating-star {
    position: absolute;
    width: 3px;
    height: 3px;
    background-color: white;
    border-radius: 50%;
    box-shadow: 0 0 10px rgba(255, 255, 255, 0.8);
  }
  
  @keyframes pulse {
    0%, 100% {
      transform: scale(1);
      opacity: 1;
    }
    50% {
      transform: scale(1.1);
      opacity: 0.8;
    }
  }
  
  @media (max-width: 768px) {
    .not-found-title {
      font-size: 6rem;
    }
    
    .not-found-content h2 {
      font-size: 1.5rem;
    }
    
    .not-found-content p {
      font-size: 1rem;
    }
  }
  
  @media (max-width: 480px) {
    .not-found-title {
      font-size: 4rem;
    }
    
    .not-found-content h2 {
      font-size: 1.2rem;
    }
  }
`;

// Insertar estilos en el documento si no existen ya
if (typeof document !== 'undefined') {
  const styleId = 'not-found-page-styles';
  
  if (!document.getElementById(styleId)) {
    const styleElement = document.createElement('style');
    styleElement.id = styleId;
    styleElement.textContent = styles;
    document.head.appendChild(styleElement);
  }
}