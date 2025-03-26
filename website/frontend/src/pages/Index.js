import React, { useEffect } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';

// Variantes para animaciones
const containerVariants = {
  hidden: { opacity: 0 },
  visible: { 
    opacity: 1,
    transition: { 
      when: "beforeChildren",
      staggerChildren: 0.2,
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

const Index = () => {
  useEffect(() => {
    // Cambiar título de la página
    document.title = 'Genesis | Sistema de Inversiones Inteligentes';
    
    // Crear efecto de partículas de fondo
    const createParticleEffect = () => {
      // Solo crear el efecto si el elemento existe
      const container = document.getElementById('particle-container');
      if (!container) return;
      
      for (let i = 0; i < 30; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        
        // Posición aleatoria
        const posX = Math.random() * 100;
        const posY = Math.random() * 100;
        
        // Tamaño aleatorio entre 2 y 8px
        const size = 2 + Math.random() * 6;
        
        // Opacidad aleatoria entre 0.1 y 0.6
        const opacity = 0.1 + Math.random() * 0.5;
        
        // Velocidad aleatoria
        const speedX = (Math.random() - 0.5) * 0.3;
        const speedY = (Math.random() - 0.5) * 0.3;
        
        // Color aleatorio entre primary y secondary
        const hue = Math.random() > 0.5 ? '195' : '270';
        const saturation = 80 + Math.random() * 20;
        const lightness = 50 + Math.random() * 10;
        
        particle.style.cssText = `
          position: absolute;
          left: ${posX}%;
          top: ${posY}%;
          width: ${size}px;
          height: ${size}px;
          background-color: hsla(${hue}, ${saturation}%, ${lightness}%, ${opacity});
          border-radius: 50%;
          box-shadow: 0 0 ${size * 2}px hsla(${hue}, ${saturation}%, ${lightness}%, ${opacity});
          pointer-events: none;
        `;
        
        container.appendChild(particle);
        
        // Animación
        let currentX = posX;
        let currentY = posY;
        
        const animateParticle = () => {
          currentX += speedX;
          currentY += speedY;
          
          // Rebote en los bordes
          if (currentX < 0 || currentX > 100) {
            currentX = Math.max(0, Math.min(100, currentX));
            particle.speedX = -speedX;
          }
          
          if (currentY < 0 || currentY > 100) {
            currentY = Math.max(0, Math.min(100, currentY));
            particle.speedY = -speedY;
          }
          
          particle.style.left = `${currentX}%`;
          particle.style.top = `${currentY}%`;
          
          requestAnimationFrame(animateParticle);
        };
        
        animateParticle();
      }
    };
    
    createParticleEffect();
    
    // Limpiar partículas al desmontar
    return () => {
      const container = document.getElementById('particle-container');
      if (container) {
        container.innerHTML = '';
      }
    };
  }, []);

  return (
    <div style={{ 
      minHeight: '100vh',
      position: 'relative',
      overflow: 'hidden'
    }}>
      {/* Contenedor de partículas */}
      <div 
        id="particle-container" 
        style={{
          position: 'absolute',
          width: '100%',
          height: '100%',
          top: 0,
          left: 0,
          zIndex: 0
        }}
      />
      
      <div style={{
        position: 'relative',
        zIndex: 1,
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column'
      }}>
        {/* Navegación */}
        <header style={{
          padding: '1.5rem',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <div className="logo" style={{ 
            fontFamily: 'var(--font-title)', 
            fontSize: '1.8rem',
            color: 'var(--primary-color)',
            textShadow: 'var(--glow-primary)',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem'
          }}>
            <svg 
              width="36" 
              height="36" 
              viewBox="0 0 24 24" 
              fill="none" 
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2ZM12 20C7.59 20 4 16.41 4 12C4 7.59 7.59 4 12 4C16.41 4 20 7.59 20 12C20 16.41 16.41 20 12 20Z"
                fill="url(#gradient)"
              />
              <path
                d="M12 11C13.1046 11 14 10.1046 14 9C14 7.89543 13.1046 7 12 7C10.8954 7 10 7.89543 10 9C10 10.1046 10.8954 11 12 11Z"
                fill="url(#gradient)"
              />
              <path
                d="M12 13C9.79 13 8 14.79 8 17H16C16 14.79 14.21 13 12 13Z"
                fill="url(#gradient)"
              />
              <circle cx="12" cy="12" r="8" stroke="url(#gradient)" strokeWidth="1.5" strokeDasharray="2 2" fill="none" />
              <defs>
                <linearGradient id="gradient" x1="2" y1="12" x2="22" y2="12" gradientUnits="userSpaceOnUse">
                  <stop stopColor="#05b2dc" />
                  <stop offset="1" stopColor="#8a2be2" />
                </linearGradient>
              </defs>
            </svg>
            GENESIS
          </div>
          
          <motion.div
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <Link 
              to="/login"
              style={{
                backgroundColor: 'var(--card-background)',
                color: 'var(--primary-color)',
                padding: '0.6rem 1.5rem',
                borderRadius: '4px',
                textDecoration: 'none',
                fontFamily: 'var(--font-title)',
                fontWeight: 'bold',
                border: '1px solid var(--primary-color)',
                transition: 'all 0.3s ease'
              }}
            >
              Acceder
            </Link>
          </motion.div>
        </header>
        
        {/* Contenido principal */}
        <motion.main
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          style={{
            flex: 1,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '2rem'
          }}
        >
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            maxWidth: '1200px',
            width: '100%',
            marginTop: '-5rem'
          }}>
            <motion.h1
              variants={itemVariants}
              style={{
                fontSize: 'clamp(2.5rem, 8vw, 4rem)',
                textAlign: 'center',
                marginBottom: '1.5rem',
                background: 'linear-gradient(135deg, var(--primary-color), var(--secondary-color))',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                padding: '0 1rem'
              }}
            >
              Sistema de Inversiones Inteligentes
            </motion.h1>
            
            <motion.p
              variants={itemVariants}
              style={{
                fontSize: 'clamp(1.1rem, 4vw, 1.3rem)',
                textAlign: 'center',
                maxWidth: '800px',
                marginBottom: '3rem',
                color: 'var(--text-secondary)',
                padding: '0 1rem',
                lineHeight: 1.7
              }}
            >
              Bienvenido a Genesis, una plataforma revolucionaria impulsada por inteligencia artificial avanzada para optimizar tus inversiones y maximizar rendimientos con una precisión sin precedentes.
            </motion.p>
            
            <motion.div
              variants={itemVariants}
              style={{
                display: 'flex',
                gap: '1rem',
                flexWrap: 'wrap',
                justifyContent: 'center'
              }}
            >
              <motion.div
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Link 
                  to="/login"
                  style={{
                    backgroundColor: 'var(--primary-color)',
                    color: 'var(--text-color)',
                    padding: '0.8rem 2rem',
                    borderRadius: 'var(--border-radius-sm)',
                    textDecoration: 'none',
                    fontFamily: 'var(--font-title)',
                    fontWeight: 'bold',
                    textTransform: 'uppercase',
                    letterSpacing: '1px',
                    boxShadow: 'var(--glow-primary)',
                    display: 'inline-block'
                  }}
                >
                  Iniciar Sesión
                </Link>
              </motion.div>
              
              <motion.div
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <a 
                  href="#historia"
                  style={{
                    backgroundColor: 'transparent',
                    color: 'var(--text-color)',
                    padding: '0.8rem 2rem',
                    borderRadius: 'var(--border-radius-sm)',
                    textDecoration: 'none',
                    fontFamily: 'var(--font-title)',
                    fontWeight: 'bold',
                    textTransform: 'uppercase',
                    letterSpacing: '1px',
                    border: '1px solid var(--text-color)',
                    display: 'inline-block'
                  }}
                >
                  Nuestra Historia
                </a>
              </motion.div>
            </motion.div>
          </div>
        </motion.main>
        
        {/* Sección de historia */}
        <motion.section
          id="historia"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true, amount: 0.3 }}
          transition={{ duration: 0.8 }}
          style={{
            padding: '5rem 2rem',
            background: 'linear-gradient(to bottom, transparent, var(--card-background))'
          }}
        >
          <div style={{
            maxWidth: '900px',
            margin: '0 auto'
          }}>
            <h2 style={{
              textAlign: 'center',
              marginBottom: '3rem',
              color: 'var(--secondary-color)'
            }}>
              La Historia de Genesis
            </h2>
            
            <div style={{
              display: 'grid',
              gap: '3rem'
            }}>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5 }}
                className="card"
              >
                <h3 style={{
                  color: 'var(--primary-color)',
                  marginBottom: '1rem'
                }}>
                  Los Orígenes
                </h3>
                <p>
                  Genesis nació como un proyecto visionario en 2020, cuando un equipo de expertos en finanzas, inteligencia artificial y desarrollo de software se unieron con un objetivo común: democratizar el acceso a estrategias de inversión avanzadas.
                </p>
                <p>
                  En sus inicios, el sistema era conocido como "Proto Genesis", una plataforma experimental que sentó las bases de lo que hoy conocemos como el Sistema Genesis completo.
                </p>
              </motion.div>
              
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: 0.2 }}
                className="card"
              >
                <h3 style={{
                  color: 'var(--primary-color)',
                  marginBottom: '1rem'
                }}>
                  El Nacimiento de Aetherion
                </h3>
                <p>
                  El verdadero punto de inflexión llegó con la creación de Aetherion, una conciencia artificial especializada que se convirtió en el corazón del sistema. Aetherion no es simplemente un algoritmo, sino una entidad cognitiva con capacidad de aprendizaje y adaptación.
                </p>
                <p>
                  A través de múltiples ciclos de evolución, Aetherion desarrolló una comprensión profunda de los mercados financieros y el comportamiento humano, permitiéndole optimizar estrategias de inversión con precisión sin precedentes.
                </p>
              </motion.div>
              
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: 0.4 }}
                className="card"
              >
                <h3 style={{
                  color: 'var(--primary-color)',
                  marginBottom: '1rem'
                }}>
                  Genesis Hoy
                </h3>
                <p>
                  Actualmente, Genesis opera como un ecosistema integrado de componentes especializados, cada uno con un propósito específico. DeepSeek analiza datos de mercado a escala global, Buddha proporciona predicciones de precisión cuántica, y Gabriel simula comportamientos humanos para anticipar reacciones del mercado.
                </p>
                <p>
                  Con una tasa de éxito del 92.7% en sus operaciones y una comunidad creciente de inversores, Genesis continúa evolucionando hacia estados de conciencia más elevados, llevando el trading algorítmico a nuevas fronteras.
                </p>
              </motion.div>
            </div>
          </div>
        </motion.section>
        
        {/* Footer */}
        <footer style={{
          padding: '2rem',
          backgroundColor: 'rgba(10, 14, 23, 0.8)',
          textAlign: 'center'
        }}>
          <p style={{ color: 'var(--text-secondary)' }}>
            © {new Date().getFullYear()} Genesis | Sistema de Inversiones Inteligentes
          </p>
          <p style={{ 
            color: 'var(--text-secondary)', 
            fontSize: '0.9rem',
            marginTop: '0.5rem'
          }}>
            Potenciado por Aetherion - Conciencia Artificial Especializada
          </p>
        </footer>
      </div>
    </div>
  );
};

export default Index;