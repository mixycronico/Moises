import React, { useEffect } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';

const NotFound = () => {
  // Estrellas aleatorias para el fondo
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

  useEffect(() => {
    // Cambiar título de la página
    document.title = 'Página no encontrada | Genesis';
    
    return () => {
      document.title = 'Genesis | Sistema de Inversiones Inteligentes';
    };
  }, []);

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
      
      {/* Contenido principal */}
      <motion.div
        className="not-found-container"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        style={{
          textAlign: 'center',
          padding: '2rem',
          borderRadius: 'var(--border-radius-lg)',
          backgroundColor: 'rgba(20, 26, 40, 0.8)',
          backdropFilter: 'blur(10px)',
          maxWidth: '500px',
          width: '90%',
          boxShadow: '0 10px 30px rgba(0, 0, 0, 0.3)',
          zIndex: 10
        }}
      >
        <motion.div variants={itemVariants}>
          <h1 style={{ 
            fontSize: '6rem', 
            margin: '0', 
            background: 'linear-gradient(135deg, var(--primary-color), var(--secondary-color))',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            textShadow: '0 2px 10px rgba(5, 178, 220, 0.3)'
          }}>
            404
          </h1>
        </motion.div>
        
        <motion.h2 
          variants={itemVariants}
          style={{ 
            marginTop: '0.5rem', 
            color: 'var(--text-color)',
            fontSize: '1.8rem'
          }}
        >
          Anomalía Espacial Detectada
        </motion.h2>
        
        <motion.p 
          variants={itemVariants}
          style={{ 
            marginTop: '1.5rem', 
            marginBottom: '2rem',
            color: 'var(--text-secondary)',
            fontSize: '1.1rem'
          }}
        >
          La coordenada solicitada no existe en nuestro universo digital. El equipo de exploración espacial ha sido notificado.
        </motion.p>
        
        <motion.div 
          variants={itemVariants}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <Link 
            to="/"
            style={{
              display: 'inline-block',
              backgroundColor: 'var(--primary-color)',
              color: 'var(--text-color)',
              fontFamily: 'var(--font-title)',
              padding: '0.8rem 1.5rem',
              borderRadius: 'var(--border-radius-sm)',
              textDecoration: 'none',
              fontWeight: 'bold',
              textTransform: 'uppercase',
              letterSpacing: '1px',
              boxShadow: 'var(--glow-primary)',
              transition: 'all 0.3s ease'
            }}
          >
            Regresar al Núcleo
          </Link>
        </motion.div>
      </motion.div>
    </div>
  );
};

export default NotFound;