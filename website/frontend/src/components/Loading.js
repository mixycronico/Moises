import React from 'react';
import { motion } from 'framer-motion';

const Loading = ({ message = 'Cargando...' }) => {
  // Variantes para animaciones
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: { 
      opacity: 1,
      transition: { 
        when: "beforeChildren",
        staggerChildren: 0.2
      }
    }
  };
  
  const circleVariants = {
    hidden: { scale: 0, opacity: 0 },
    visible: {
      scale: 1,
      opacity: 1,
      transition: {
        repeat: Infinity,
        repeatType: "reverse",
        duration: 1
      }
    }
  };
  
  const messageVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: { 
      y: 0, 
      opacity: 1,
      transition: {
        delay: 0.5,
        ease: "easeOut"
      }
    }
  };
  
  // Colores para los círculos
  const colors = [
    'var(--primary-color)', 
    'var(--secondary-color)', 
    'var(--accent-color)'
  ];
  
  return (
    <motion.div
      className="loading-container"
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: '100vh',
        background: 'var(--background-color)',
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        zIndex: 1000
      }}
    >
      <div style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        gap: '12px',
        marginBottom: '20px'
      }}>
        {colors.map((color, index) => (
          <motion.div
            key={index}
            variants={circleVariants}
            style={{
              width: '20px',
              height: '20px',
              borderRadius: '50%',
              backgroundColor: color,
              boxShadow: `0 0 15px ${color}`
            }}
            transition={{
              repeat: Infinity,
              repeatType: "reverse",
              duration: 1,
              delay: index * 0.2
            }}
          />
        ))}
      </div>
      
      <motion.div
        variants={messageVariants}
        style={{
          fontFamily: 'var(--font-title)',
          color: 'var(--primary-color)',
          textShadow: 'var(--glow-primary)',
          fontSize: '1.2rem',
          letterSpacing: '1px'
        }}
      >
        {message}
      </motion.div>
    </motion.div>
  );
};

export default Loading;