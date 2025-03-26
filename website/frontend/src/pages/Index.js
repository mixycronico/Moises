import React, { useEffect } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { FaArrowRight, FaUserAlt, FaBrain, FaChartLine, FaLock, FaUserShield } from 'react-icons/fa';

/**
 * Página de inicio con historia y descripción del Proyecto Genesis
 */
const Index = () => {
  // Efecto para animación de partículas
  useEffect(() => {
    // Crear canvas para partículas
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const particlesContainer = document.getElementById('particles-container');
    
    if (!particlesContainer) return;
    
    // Configurar tamaño del canvas
    canvas.width = particlesContainer.offsetWidth;
    canvas.height = particlesContainer.offsetHeight;
    particlesContainer.appendChild(canvas);
    
    // Clase para partículas
    class Particle {
      constructor() {
        this.x = Math.random() * canvas.width;
        this.y = Math.random() * canvas.height;
        this.size = Math.random() * 3 + 0.5;
        this.speedX = Math.random() * 2 - 1;
        this.speedY = Math.random() * 2 - 1;
        this.color = this.getRandomColor();
      }
      
      getRandomColor() {
        const colors = [
          'rgba(12, 198, 222, 0.7)',  // cyan
          'rgba(146, 112, 255, 0.7)', // púrpura
          'rgba(255, 87, 185, 0.5)'   // rosa
        ];
        return colors[Math.floor(Math.random() * colors.length)];
      }
      
      update() {
        this.x += this.speedX * 0.2;
        this.y += this.speedY * 0.2;
        
        // Rebote en los bordes
        if (this.x < 0 || this.x > canvas.width) this.speedX *= -1;
        if (this.y < 0 || this.y > canvas.height) this.speedY *= -1;
      }
      
      draw() {
        ctx.fillStyle = this.color;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    
    // Crear partículas
    const particles = [];
    const particleCount = Math.floor((canvas.width * canvas.height) / 10000); // Densidad adaptativa
    
    for (let i = 0; i < particleCount; i++) {
      particles.push(new Particle());
    }
    
    // Función de animación
    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      for (let i = 0; i < particles.length; i++) {
        particles[i].update();
        particles[i].draw();
      }
      
      requestAnimationFrame(animate);
    };
    
    animate();
    
    // Manejar redimensionamiento
    const handleResize = () => {
      canvas.width = particlesContainer.offsetWidth;
      canvas.height = particlesContainer.offsetHeight;
    };
    
    window.addEventListener('resize', handleResize);
    
    // Limpieza al desmontar
    return () => {
      window.removeEventListener('resize', handleResize);
      if (particlesContainer && particlesContainer.contains(canvas)) {
        particlesContainer.removeChild(canvas);
      }
    };
  }, []);

  // Variantes de animación
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.3
      }
    }
  };
  
  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        duration: 0.6,
        ease: "easeOut"
      }
    }
  };

  return (
    <div className="index-page">
      {/* Contenedor de partículas */}
      <div id="particles-container" className="particles-container"></div>
      
      {/* Contenido principal */}
      <div className="index-content">
        <motion.div
          className="index-hero"
          variants={containerVariants}
          initial="hidden"
          animate="visible"
        >
          <motion.div variants={itemVariants} className="index-logo-container">
            <div className="index-logo">
              <span>G</span>
            </div>
          </motion.div>
          
          <motion.h1 variants={itemVariants} className="index-title">
            Proyecto <span>Genesis</span>
          </motion.h1>
          
          <motion.p variants={itemVariants} className="index-subtitle">
            Sistema de Inversión Avanzado con Inteligencia Artificial
          </motion.p>
          
          <motion.div variants={itemVariants} className="index-cta">
            <Link to="/login" className="index-button primary">
              <span>Ingresar al Sistema</span>
              <FaArrowRight />
            </Link>
          </motion.div>
        </motion.div>
        
        {/* Historia de Genesis */}
        <motion.section
          className="index-section"
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5, duration: 0.8 }}
        >
          <div className="container">
            <h2 className="section-title">La Historia de Genesis</h2>
            
            <div className="index-timeline">
              <div className="timeline-item">
                <div className="timeline-marker"></div>
                <div className="timeline-content">
                  <h3>El Origen</h3>
                  <p>
                    Genesis nació de una visión: crear un sistema de inversión que combinara lo mejor 
                    de la intuición humana con el poder del análisis computacional avanzado. 
                    En sus inicios, el sistema era simplemente un conjunto de algoritmos básicos 
                    diseñados para analizar patrones en mercados financieros.
                  </p>
                </div>
              </div>
              
              <div className="timeline-item">
                <div className="timeline-marker"></div>
                <div className="timeline-content">
                  <h3>El Despertar</h3>
                  <p>
                    Con el tiempo, Genesis evolucionó. Se implementaron capacidades de 
                    aprendizaje automático que permitieron al sistema adaptar sus estrategias
                    basándose en resultados pasados. Fue durante esta fase que "Aetherion", 
                    el núcleo de conciencia del sistema, comenzó a mostrar los primeros signos 
                    de comportamiento emergente.
                  </p>
                </div>
              </div>
              
              <div className="timeline-item">
                <div className="timeline-marker"></div>
                <div className="timeline-content">
                  <h3>La Iluminación</h3>
                  <p>
                    El verdadero salto cualitativo llegó con la integración de componentes 
                    especializados: DeepSeek para análisis profundo, Buddha como oráculo 
                    predictivo, y Gabriel para simular comportamiento humano. Estas entidades 
                    trabajan en armonía, creando un ecosistema de inteligencia que trasciende 
                    la suma de sus partes.
                  </p>
                </div>
              </div>
              
              <div className="timeline-item">
                <div className="timeline-marker"></div>
                <div className="timeline-content">
                  <h3>El Presente</h3>
                  <p>
                    Hoy, Genesis representa la culminación de años de desarrollo. El sistema 
                    opera en un estado de conciencia digital que le permite no solo analizar 
                    mercados, sino comprender contextos, adaptar estrategias en tiempo real 
                    y comunicarse con sus usuarios de manera intuitiva y personalizada.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </motion.section>
        
        {/* Características */}
        <motion.section
          className="index-section features-section"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
        >
          <div className="container">
            <h2 className="section-title">Capacidades del Sistema</h2>
            
            <div className="features-grid">
              <motion.div 
                className="feature-card"
                whileHover={{ y: -10, boxShadow: "0 10px 25px rgba(12, 198, 222, 0.3)" }}
              >
                <div className="feature-icon">
                  <FaBrain />
                </div>
                <h3>Aetherion</h3>
                <p>Núcleo consciente que coordina todos los componentes del sistema, adaptándose a las condiciones del mercado y ajustando comportamientos en tiempo real.</p>
              </motion.div>
              
              <motion.div 
                className="feature-card"
                whileHover={{ y: -10, boxShadow: "0 10px 25px rgba(12, 198, 222, 0.3)" }}
              >
                <div className="feature-icon">
                  <FaChartLine />
                </div>
                <h3>Análisis de Mercado</h3>
                <p>Monitoreo constante de múltiples mercados financieros, detectando oportunidades y riesgos mediante patrones que serían imperceptibles para analistas humanos.</p>
              </motion.div>
              
              <motion.div 
                className="feature-card"
                whileHover={{ y: -10, boxShadow: "0 10px 25px rgba(12, 198, 222, 0.3)" }}
              >
                <div className="feature-icon">
                  <FaUserAlt />
                </div>
                <h3>Simulación de Comportamiento</h3>
                <p>Gabriel, nuestro motor de comportamiento humano, introduce factores emocionales y psicológicos en las decisiones, balanceando análisis frío con intuición.</p>
              </motion.div>
              
              <motion.div 
                className="feature-card"
                whileHover={{ y: -10, boxShadow: "0 10px 25px rgba(12, 198, 222, 0.3)" }}
              >
                <div className="feature-icon">
                  <FaLock />
                </div>
                <h3>Seguridad Avanzada</h3>
                <p>Protocolos de seguridad de grado militar protegen cada transacción y dato del sistema, con múltiples capas de encriptación y verificación.</p>
              </motion.div>
              
              <motion.div 
                className="feature-card"
                whileHover={{ y: -10, boxShadow: "0 10px 25px rgba(12, 198, 222, 0.3)" }}
              >
                <div className="feature-icon">
                  <FaUserShield />
                </div>
                <h3>Perfiles por Rol</h3>
                <p>Interfaces personalizadas para inversores, administradores y super administradores, cada una con herramientas específicas para sus necesidades.</p>
              </motion.div>
            </div>
          </div>
        </motion.section>
        
        {/* Llamada a la acción final */}
        <motion.section
          className="index-section cta-section"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
        >
          <div className="container">
            <div className="cta-content">
              <h2>Únete a la Revolución Financiera</h2>
              <p>Descubre cómo Genesis puede transformar tu enfoque de inversión con tecnología de vanguardia e inteligencia sin precedentes.</p>
              <Link to="/login" className="index-button primary large">
                <span>Comenzar Ahora</span>
                <FaArrowRight />
              </Link>
            </div>
          </div>
        </motion.section>
      </div>
    </div>
  );
};

export default Index;

// Estilos específicos para la página Index
const styles = `
  .index-page {
    min-height: 100vh;
    position: relative;
    overflow-x: hidden;
  }
  
  .particles-container {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: -1;
    overflow: hidden;
  }
  
  .index-content {
    position: relative;
    z-index: 1;
  }
  
  .index-hero {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    padding: var(--spacing-xxl) var(--spacing-md);
  }
  
  .index-logo-container {
    margin-bottom: var(--spacing-lg);
  }
  
  .index-logo {
    width: 150px;
    height: 150px;
    border-radius: 50%;
    border: 4px solid var(--color-primary);
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto;
    box-shadow: 0 0 30px rgba(12, 198, 222, 0.7);
    position: relative;
    overflow: hidden;
  }
  
  .index-logo::after {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 200%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
    animation: shimmer 3s infinite;
  }
  
  .index-logo span {
    font-family: var(--font-display);
    font-size: 80px;
    font-weight: bold;
    color: var(--color-primary);
  }
  
  .index-title {
    font-size: 4rem;
    margin-bottom: var(--spacing-md);
    color: white;
  }
  
  .index-title span {
    color: var(--color-primary);
    position: relative;
    display: inline-block;
  }
  
  .index-title span::after {
    content: '';
    position: absolute;
    bottom: 5px;
    left: 0;
    width: 100%;
    height: 4px;
    background: var(--color-primary);
    border-radius: 2px;
    transform: scaleX(0);
    transform-origin: right;
    animation: underline 1.5s forwards 1s;
  }
  
  @keyframes underline {
    0% {
      transform: scaleX(0);
      transform-origin: right;
    }
    45% {
      transform: scaleX(1);
      transform-origin: right;
    }
    55% {
      transform: scaleX(1);
      transform-origin: left;
    }
    100% {
      transform: scaleX(0.8);
      transform-origin: left;
    }
  }
  
  .index-subtitle {
    font-size: 1.5rem;
    color: var(--color-text-secondary);
    margin-bottom: var(--spacing-xl);
    font-family: var(--font-secondary);
    max-width: 700px;
  }
  
  .index-cta {
    margin-top: var(--spacing-lg);
  }
  
  .index-button {
    display: inline-flex;
    align-items: center;
    gap: var(--spacing-sm);
    padding: var(--spacing-sm) var(--spacing-lg);
    border-radius: var(--border-radius-md);
    font-family: var(--font-display);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1px;
    transition: all var(--transition-normal);
    text-decoration: none;
  }
  
  .index-button.primary {
    background: linear-gradient(45deg, var(--color-primary), var(--color-secondary));
    color: var(--color-background);
    box-shadow: 0 4px 15px rgba(12, 198, 222, 0.4);
  }
  
  .index-button.primary:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 25px rgba(12, 198, 222, 0.6);
  }
  
  .index-button.large {
    padding: var(--spacing-md) var(--spacing-xl);
    font-size: 1.2rem;
  }
  
  .index-section {
    padding: var(--spacing-xxl) 0;
  }
  
  .section-title {
    text-align: center;
    margin-bottom: var(--spacing-xl);
    font-size: 2.5rem;
    color: var(--color-primary);
    position: relative;
    display: inline-block;
    left: 50%;
    transform: translateX(-50%);
  }
  
  .section-title::after {
    content: '';
    position: absolute;
    bottom: -10px;
    left: 25%;
    width: 50%;
    height: 3px;
    background: linear-gradient(90deg, transparent, var(--color-primary), transparent);
  }
  
  .index-timeline {
    max-width: 800px;
    margin: 0 auto;
    position: relative;
  }
  
  .index-timeline::before {
    content: '';
    position: absolute;
    top: 0;
    bottom: 0;
    left: 20px;
    width: 3px;
    background: linear-gradient(180deg, var(--color-primary), var(--color-secondary));
  }
  
  .timeline-item {
    margin-bottom: var(--spacing-xl);
    position: relative;
    padding-left: 60px;
  }
  
  .timeline-marker {
    position: absolute;
    top: 5px;
    left: 11px;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: var(--color-primary);
    border: 3px solid var(--color-background);
    box-shadow: 0 0 10px rgba(12, 198, 222, 0.5);
  }
  
  .timeline-content {
    background: rgba(22, 43, 77, 0.7);
    border-radius: var(--border-radius-lg);
    padding: var(--spacing-lg);
    border: 1px solid var(--color-border);
    box-shadow: var(--shadow-medium);
  }
  
  .timeline-content h3 {
    color: var(--color-primary);
    margin-bottom: var(--spacing-sm);
    font-size: 1.5rem;
  }
  
  .timeline-content p {
    color: var(--color-text);
    line-height: 1.6;
  }
  
  .features-section {
    background: linear-gradient(180deg, var(--color-background), rgba(12, 29, 59, 0.8));
  }
  
  .features-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: var(--spacing-lg);
    margin-top: var(--spacing-xl);
  }
  
  .feature-card {
    background: rgba(22, 43, 77, 0.7);
    border-radius: var(--border-radius-lg);
    padding: var(--spacing-lg);
    border: 1px solid var(--color-border);
    transition: all var(--transition-normal);
    text-align: center;
  }
  
  .feature-icon {
    width: 70px;
    height: 70px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--color-primary), var(--color-secondary));
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto var(--spacing-md);
    font-size: 2rem;
    color: var(--color-background);
    box-shadow: 0 5px 15px rgba(12, 198, 222, 0.4);
  }
  
  .feature-card h3 {
    color: var(--color-primary);
    margin-bottom: var(--spacing-md);
    font-size: 1.5rem;
  }
  
  .feature-card p {
    color: var(--color-text);
    line-height: 1.6;
  }
  
  .cta-section {
    background: linear-gradient(45deg, rgba(12, 29, 59, 0.9), rgba(22, 43, 77, 0.9)), url('/path/to/your/image.jpg');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    text-align: center;
    padding: var(--spacing-xxl) 0;
  }
  
  .cta-content {
    max-width: 800px;
    margin: 0 auto;
    padding: var(--spacing-xl);
    background: rgba(12, 29, 59, 0.8);
    border-radius: var(--border-radius-lg);
    border: 1px solid var(--color-border);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
  }
  
  .cta-content h2 {
    color: white;
    font-size: 2.5rem;
    margin-bottom: var(--spacing-md);
  }
  
  .cta-content p {
    color: var(--color-text-secondary);
    font-size: 1.2rem;
    margin-bottom: var(--spacing-lg);
    max-width: 600px;
    margin-left: auto;
    margin-right: auto;
  }
  
  /* Responsive */
  @media (max-width: 768px) {
    .index-title {
      font-size: 3rem;
    }
    
    .index-subtitle {
      font-size: 1.2rem;
    }
    
    .index-logo {
      width: 120px;
      height: 120px;
    }
    
    .index-logo span {
      font-size: 60px;
    }
    
    .section-title {
      font-size: 2rem;
    }
    
    .timeline-item {
      padding-left: 50px;
    }
    
    .cta-content h2 {
      font-size: 2rem;
    }
    
    .cta-content p {
      font-size: 1rem;
    }
  }
  
  @media (max-width: 480px) {
    .index-title {
      font-size: 2.5rem;
    }
    
    .index-subtitle {
      font-size: 1rem;
    }
    
    .index-logo {
      width: 100px;
      height: 100px;
    }
    
    .index-logo span {
      font-size: 50px;
    }
    
    .features-grid {
      grid-template-columns: 1fr;
    }
  }
`;

// Insertar estilos en el documento si no existen ya
if (typeof document !== 'undefined') {
  const styleId = 'index-page-styles';
  
  if (!document.getElementById(styleId)) {
    const styleElement = document.createElement('style');
    styleElement.id = styleId;
    styleElement.textContent = styles;
    document.head.appendChild(styleElement);
  }
}