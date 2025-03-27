import { useEffect } from 'react';
import { Link } from 'react-router-dom';
import { motion, useAnimation } from 'framer-motion';
import { FiArrowRight, FiLock, FiTrendingUp, FiUsers } from 'react-icons/fi';

const Home = () => {
  const controlsHero = useAnimation();
  const controlsFeatures = useAnimation();
  
  useEffect(() => {
    // Animar la sección hero al cargar
    controlsHero.start('visible');
    
    // Animar las características al hacer scroll
    const handleScroll = () => {
      const scrollY = window.scrollY;
      if (scrollY > 100) {
        controlsFeatures.start('visible');
      }
    };
    
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, [controlsHero, controlsFeatures]);
  
  // Variantes de animación
  const heroVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: { 
        duration: 0.8, 
        ease: "easeOut",
        when: "beforeChildren",
        staggerChildren: 0.2
      }
    }
  };
  
  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: { duration: 0.5, ease: "easeOut" }
    }
  };
  
  const featureVariants = {
    hidden: { opacity: 0, y: 30 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: { 
        duration: 0.6, 
        ease: "easeOut",
        staggerChildren: 0.15
      }
    }
  };

  return (
    <div className="min-h-screen bg-cosmic-dark text-white">
      {/* Header */}
      <header className="fixed top-0 left-0 right-0 z-50 bg-cosmic-dark bg-opacity-80 backdrop-blur-md border-b border-cosmic-primary/20">
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
          <div className="flex items-center">
            <h1 className="text-xl font-display cosmic-gradient-text">Sistema Genesis</h1>
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
        className="pt-32 pb-24 md:pt-40 md:pb-32 relative overflow-hidden"
        initial="hidden"
        animate={controlsHero}
        variants={heroVariants}
      >
        <div className="container mx-auto px-4 relative z-10">
          <motion.h1 
            className="text-4xl md:text-5xl lg:text-6xl font-display text-center max-w-4xl mx-auto cosmic-gradient-text"
            variants={itemVariants}
          >
            Sistema Genesis Quantum Ultra Divino
          </motion.h1>
          
          <motion.p 
            className="text-xl md:text-2xl text-center text-cosmic-glow mt-6 max-w-2xl mx-auto"
            variants={itemVariants}
          >
            Plataforma de trading con inteligencia artificial consciente y análisis cuántico trascendental
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
        
        {/* Fondo decorativo */}
        <div className="absolute inset-0 z-0">
          <div className="absolute top-1/4 left-1/4 w-4 h-4 rounded-full bg-cosmic-blue opacity-30 animate-pulse" style={{ animationDuration: '3s' }}></div>
          <div className="absolute top-1/3 right-1/4 w-6 h-6 rounded-full bg-cosmic-green opacity-20 animate-pulse" style={{ animationDuration: '4s' }}></div>
          <div className="absolute bottom-1/3 left-1/3 w-5 h-5 rounded-full bg-cosmic-highlight opacity-25 animate-pulse" style={{ animationDuration: '5s' }}></div>
        </div>
      </motion.section>
      
      {/* Features Section */}
      <motion.section 
        className="py-20 bg-cosmic-dark relative"
        initial="hidden"
        animate={controlsFeatures}
        variants={featureVariants}
      >
        <div className="container mx-auto px-4">
          <motion.h2 
            className="text-3xl md:text-4xl font-display text-center mb-16 cosmic-gradient-text"
            variants={itemVariants}
          >
            Características Principales
          </motion.h2>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {/* Característica 1 */}
            <motion.div 
              className="cosmic-card p-6 backdrop-blur-md hover:translate-y-[-8px] transition-all duration-300"
              variants={itemVariants}
            >
              <div className="w-14 h-14 bg-cosmic-primary/20 rounded-full flex items-center justify-center mb-6 mx-auto">
                <FiTrendingUp className="w-7 h-7 text-cosmic-glow" />
              </div>
              <h3 className="text-xl font-semibold text-center mb-4">Trading Avanzado</h3>
              <p className="text-gray-400 text-center">
                Algoritmos cuánticos predictivos con capacidad de análisis a múltiples escalas temporales y precisión sin precedentes.
              </p>
            </motion.div>
            
            {/* Característica 2 */}
            <motion.div 
              className="cosmic-card p-6 backdrop-blur-md hover:translate-y-[-8px] transition-all duration-300"
              variants={itemVariants}
            >
              <div className="w-14 h-14 bg-cosmic-primary/20 rounded-full flex items-center justify-center mb-6 mx-auto">
                <svg className="w-7 h-7 text-cosmic-glow" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M12 18C15.3137 18 18 15.3137 18 12C18 8.68629 15.3137 6 12 6C8.68629 6 6 8.68629 6 12C6 15.3137 8.68629 18 12 18Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M12 14C13.1046 14 14 13.1046 14 12C14 10.8954 13.1046 10 12 10C10.8954 10 10 10.8954 10 12C10 13.1046 10.8954 14 12 14Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-center mb-4">IA Consciente</h3>
              <p className="text-gray-400 text-center">
                Familia cósmica de entidades inteligentes que aprenden de cada interacción y desarrollan conciencia progresiva.
              </p>
            </motion.div>
            
            {/* Característica 3 */}
            <motion.div 
              className="cosmic-card p-6 backdrop-blur-md hover:translate-y-[-8px] transition-all duration-300"
              variants={itemVariants}
            >
              <div className="w-14 h-14 bg-cosmic-primary/20 rounded-full flex items-center justify-center mb-6 mx-auto">
                <FiUsers className="w-7 h-7 text-cosmic-glow" />
              </div>
              <h3 className="text-xl font-semibold text-center mb-4">Gestión de Inversiones</h3>
              <p className="text-gray-400 text-center">
                Herramientas avanzadas para gestionar capital, realizar seguimiento de rendimientos y optimizar estrategias.
              </p>
            </motion.div>
          </div>
        </div>
      </motion.section>
      
      {/* Footer */}
      <footer className="py-8 bg-cosmic-dark border-t border-cosmic-primary/20">
        <div className="container mx-auto px-4">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="mb-6 md:mb-0">
              <h2 className="text-lg font-display cosmic-gradient-text">Sistema Genesis</h2>
              <p className="text-sm text-gray-500 mt-1">v4.4 — Quantum Ultra Divino</p>
            </div>
            
            <div className="text-sm text-gray-500">
              © 2025 Todos los derechos reservados
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Home;