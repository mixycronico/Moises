import React from 'react';
import { motion } from 'framer-motion';

/**
 * Componente de carga con animación del logo G
 */
const Loading = () => {
  return (
    <motion.div
      className="loading-container"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.3 }}
    >
      <div className="loading-content">
        <motion.div
          className="g-logo"
          animate={{
            scale: [1, 1.1, 1],
            opacity: [0.7, 1, 0.7]
          }}
          transition={{
            duration: 2,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        />
        <motion.div
          className="loading-text"
          animate={{
            opacity: [0.5, 1, 0.5]
          }}
          transition={{
            duration: 2,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        >
          Cargando Sistema Genesis...
        </motion.div>
      </div>
    </motion.div>
  );
};

export default Loading;

// Estilos específicos del componente
const styles = `
  .loading-container {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    display: flex;
    justify-content: center;
    align-items: center;
    background-color: var(--color-background);
    z-index: var(--z-overlay);
  }
  
  .loading-content {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: var(--spacing-md);
  }
  
  .g-logo {
    width: 120px;
    height: 120px;
    border-radius: 50%;
    border: 4px solid var(--color-primary);
    position: relative;
    box-shadow: 0 0 20px rgba(12, 198, 222, 0.7);
  }
  
  .g-logo::before {
    content: "G";
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-family: var(--font-display);
    font-size: 60px;
    font-weight: bold;
    color: var(--color-primary);
  }
  
  .loading-text {
    font-family: var(--font-display);
    font-size: 1.2rem;
    color: var(--color-primary);
    letter-spacing: 2px;
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