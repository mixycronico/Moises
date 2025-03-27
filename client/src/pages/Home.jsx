import { useEffect, useState, useRef } from 'react';
import { Link } from 'react-router-dom';
import { motion, useAnimation } from 'framer-motion';
import { FiArrowRight, FiLock } from 'react-icons/fi';
import logoImage from '../assets/logo-genesis.png';

const Home = () => {
  const controlsHero = useAnimation();
  const [particles, setParticles] = useState([]);
  const networkRef = useRef(null);
  
  // Generar partículas aleatorias para la red neuronal de fondo
  useEffect(() => {
    const generateParticles = () => {
      const newParticles = [];
      const particleCount = 50; // reducido para mejor rendimiento
      
      for (let i = 0; i < particleCount; i++) {
        newParticles.push({
          id: i,
          x: Math.random() * 100,
          y: Math.random() * 100,
          size: Math.random() * 3 + 1,
          color: Math.random() > 0.7 ? '#9e6bdb' : (Math.random() > 0.5 ? '#5b8af7' : '#42c9a0'),
          opacity: Math.random() * 0.5 + 0.1
        });
      }
      
      setParticles(newParticles);
    };
    
    generateParticles();
    controlsHero.start('visible');
    
    // Función para dibujar líneas de conexión en el canvas
    const drawNetworkLines = () => {
      const canvas = networkRef.current;
      if (!canvas) return;
      
      const ctx = canvas.getContext('2d');
      const width = canvas.width;
      const height = canvas.height;
      
      // Limpiar canvas
      ctx.clearRect(0, 0, width, height);
      
      // Calcular posiciones de partículas
      const points = particles.map(p => ({
        x: p.x * width / 100,
        y: p.y * height / 100,
        size: p.size,
        color: p.color,
        opacity: p.opacity
      }));
      
      // Dibujar líneas entre puntos cercanos
      for (let i = 0; i < points.length; i++) {
        for (let j = i + 1; j < points.length; j++) {
          const dx = points[i].x - points[j].x;
          const dy = points[i].y - points[j].y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          
          // Conectar solo puntos relativamente cercanos
          if (distance < 150) {
            // Opacidad basada en la distancia
            const opacity = (1 - distance / 150) * 0.2;
            
            ctx.beginPath();
            ctx.moveTo(points[i].x, points[i].y);
            ctx.lineTo(points[j].x, points[j].y);
            ctx.strokeStyle = `rgba(158, 107, 219, ${opacity})`;
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
        }
      }
      
      // Dibujar partículas
      points.forEach(point => {
        ctx.beginPath();
        ctx.arc(point.x, point.y, point.size, 0, Math.PI * 2);
        ctx.fillStyle = point.color + Math.floor(point.opacity * 255).toString(16).padStart(2, '0');
        ctx.fill();
      });
      
      // Animar
      requestAnimationFrame(drawNetworkLines);
    };
    
    // Ajustar tamaño del canvas
    const handleResize = () => {
      if (networkRef.current) {
        networkRef.current.width = window.innerWidth;
        networkRef.current.height = window.innerHeight;
      }
    };
    
    handleResize();
    window.addEventListener('resize', handleResize);
    
    // Iniciar animación
    const animationId = requestAnimationFrame(drawNetworkLines);
    
    return () => {
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(animationId);
    };
  }, [controlsHero, particles]);
  
  // Variantes de animación
  const heroVariants = {
    hidden: { opacity: 0 },
    visible: { 
      opacity: 1,
      transition: { 
        duration: 1.2, 
        ease: "easeOut",
        when: "beforeChildren",
        staggerChildren: 0.3
      }
    }
  };
  
  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: { duration: 0.7, ease: "easeOut" }
    }
  };

  return (
    <div className="min-h-screen bg-cosmic-dark text-white overflow-hidden">
      {/* Red neuronal de fondo */}
      <canvas 
        ref={networkRef} 
        className="fixed inset-0 z-0 w-full h-full" 
      />
      
      {/* Header */}
      <header className="fixed top-0 left-0 right-0 z-50 bg-cosmic-dark bg-opacity-30 backdrop-blur-md">
        <div className="container mx-auto px-4 py-2 flex justify-between items-center">
          <div className="flex items-center">
            <img src={logoImage} alt="Genesis Logo" className="h-16 w-auto" />
          </div>
          
          <div>
            <Link 
              to="/login" 
              className="cosmic-button-secondary flex items-center"
            >
              <FiLock className="mr-2" />
              Iniciar Sesión
            </Link>
          </div>
        </div>
      </header>
      
      {/* Hero Section */}
      <motion.section 
        className="pt-32 pb-24 md:pt-40 md:pb-32 relative flex flex-col items-center justify-center min-h-screen"
        initial="hidden"
        animate={controlsHero}
        variants={heroVariants}
      >
        <div className="container mx-auto px-4 relative z-10 text-center">
          {/* Logo principal grande */}
          <motion.div
            className="mx-auto mb-12 relative"
            variants={itemVariants}
          >
            <img 
              src={logoImage} 
              alt="Genesis Logo" 
              className="h-32 w-auto mx-auto animate-float"
            />
          </motion.div>
          
          <motion.h1 
            className="text-5xl md:text-6xl lg:text-7xl font-display font-bold mb-6 text-center cosmic-gradient-text"
            variants={itemVariants}
          >
            GENESIS
          </motion.h1>
          
          <motion.div
            variants={itemVariants}
            className="mb-8"
          >
            <h2 className="text-2xl md:text-3xl lg:text-4xl font-medium text-white">
              Sistema de inversiones trascendental con inteligencia cósmica
            </h2>
          </motion.div>
          
          <motion.p 
            className="text-lg md:text-xl text-center text-cosmic-glow mt-6 max-w-2xl mx-auto"
            variants={itemVariants}
          >
            Descubre un nuevo nivel de inversión guiado por IA evolutiva
          </motion.p>
          
          <motion.div 
            className="mt-12 flex justify-center"
            variants={itemVariants}
          >
            <Link 
              to="/login" 
              className="cosmic-button text-lg px-8 py-3 flex items-center"
            >
              Acceder al Sistema
              <FiArrowRight className="ml-2" />
            </Link>
          </motion.div>
        </div>
      </motion.section>
    </div>
  );
};

export default Home;