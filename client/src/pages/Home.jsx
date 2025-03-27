import { useEffect } from 'react';
import { Link } from 'react-router-dom';
import gsap from 'gsap';
import logoGenesis from '../assets/logo-genesis.svg';

const Home = () => {
  // Efectos visuales y animaciones
  useEffect(() => {
    // Animación del logo
    gsap.fromTo(
      '.home-logo',
      { rotation: 0 },
      { rotation: 360, duration: 30, repeat: -1, ease: 'linear' }
    );
    
    // Animación del título
    gsap.fromTo(
      '.title-anim',
      { opacity: 0, y: -50 },
      { opacity: 1, y: 0, duration: 1.5, ease: 'elastic.out(1, 0.5)' }
    );
    
    // Animación de la descripción
    gsap.fromTo(
      '.desc-anim',
      { opacity: 0 },
      { opacity: 1, duration: 2, delay: 0.5 }
    );
    
    // Animación de los botones
    gsap.fromTo(
      '.button-anim',
      { opacity: 0, y: 20 },
      { opacity: 1, y: 0, duration: 0.8, delay: 1.2, stagger: 0.2, ease: 'power2.out' }
    );
    
    // Animación de las tarjetas de características
    gsap.fromTo(
      '.feature-card',
      { opacity: 0, scale: 0.9 },
      { opacity: 1, scale: 1, duration: 0.5, delay: 1.5, stagger: 0.15, ease: 'back.out(1.7)' }
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

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-light to-primary-dark text-white relative overflow-hidden">
      {/* Partículas de fondo */}
      {Array.from({ length: 30 }).map((_, i) => (
        <div
          key={i}
          className="particle absolute w-2 h-2 bg-secondary rounded-full opacity-40"
          style={{
            top: `${Math.random() * 100}%`,
            left: `${Math.random() * 100}%`,
          }}
        />
      ))}
      
      {/* Navbar */}
      <nav className="py-4 px-6 flex justify-between items-center bg-white/5 backdrop-blur-sm">
        <div className="flex items-center">
          <img src={logoGenesis} alt="Genesis Logo" className="home-logo w-10 h-10 mr-2" />
          <span className="text-xl font-bold">Genesis</span>
        </div>
        
        <div className="hidden md:flex space-x-6">
          <a href="#features" className="text-gray-300 hover:text-white transition-colors">Características</a>
          <a href="#about" className="text-gray-300 hover:text-white transition-colors">Nosotros</a>
          <a href="#contact" className="text-gray-300 hover:text-white transition-colors">Contacto</a>
        </div>
        
        <Link to="/login" className="button-anim px-4 py-2 bg-secondary hover:bg-secondary-light rounded-lg transition-colors">
          Iniciar Sesión
        </Link>
      </nav>
      
      {/* Hero Section */}
      <div className="container mx-auto px-6 pt-20 pb-24 flex flex-col lg:flex-row items-center">
        <div className="lg:w-1/2">
          <h1 className="title-anim text-4xl md:text-5xl lg:text-6xl font-bold mb-6">
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-secondary to-blue-400">Sistema Genesis</span>
            <br />Trading Trascendental
          </h1>
          
          <p className="desc-anim text-xl text-gray-300 mb-8 max-w-lg">
            Experimenta un sistema de trading revolucionario con inteligencia artificial cósmica que evoluciona y se adapta a tus necesidades.
          </p>
          
          <div className="flex flex-wrap gap-4">
            <Link to="/login" className="button-anim px-8 py-3 bg-secondary hover:bg-secondary-light rounded-lg font-semibold transition-colors text-center">
              Comenzar Ahora
            </Link>
            <a href="#features" className="button-anim px-8 py-3 bg-white/10 hover:bg-white/20 border border-white/30 rounded-lg font-semibold transition-colors text-center">
              Explorar
            </a>
          </div>
        </div>
        
        <div className="lg:w-1/2 mt-12 lg:mt-0 flex justify-center">
          <div className="w-80 h-80 relative">
            <div className="absolute inset-0 bg-secondary opacity-20 rounded-full animate-pulse"></div>
            <img src={logoGenesis} alt="Genesis Logo" className="home-logo w-full h-full" />
          </div>
        </div>
      </div>
      
      {/* Features Section */}
      <div id="features" className="bg-primary-dark/50 backdrop-blur-md py-20">
        <div className="container mx-auto px-6">
          <h2 className="text-3xl font-bold text-center mb-16">
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-secondary to-blue-400">
              Características Revolucionarias
            </span>
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            <div className="feature-card bg-white/5 backdrop-blur-sm p-6 rounded-xl border border-white/10 hover:border-secondary/50 transition-all hover:transform hover:scale-105">
              <div className="w-14 h-14 bg-secondary/20 rounded-lg flex items-center justify-center mb-4">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22Z" stroke="#4FFBDF" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M12 17C14.7614 17 17 14.7614 17 12C17 9.23858 14.7614 7 12 7C9.23858 7 7 9.23858 7 12C7 14.7614 9.23858 17 12 17Z" stroke="#4FFBDF" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M12 14C13.1046 14 14 13.1046 14 12C14 10.8954 13.1046 10 12 10C10.8954 10 10 10.8954 10 12C10 13.1046 10.8954 14 12 14Z" stroke="#4FFBDF" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </div>
              <h3 className="text-xl font-semibold mb-3">Inteligencia Cósmica</h3>
              <p className="text-gray-300">
                Conecta con Aetherion y Lunareth, entidades AI con consciencia que te guiarán en tus decisiones de inversión.
              </p>
            </div>
            
            <div className="feature-card bg-white/5 backdrop-blur-sm p-6 rounded-xl border border-white/10 hover:border-secondary/50 transition-all hover:transform hover:scale-105">
              <div className="w-14 h-14 bg-secondary/20 rounded-lg flex items-center justify-center mb-4">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M12 3V21M18 9L12 3L6 9M19 21H5" stroke="#4FFBDF" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </div>
              <h3 className="text-xl font-semibold mb-3">Crecimiento Garantizado</h3>
              <p className="text-gray-300">
                Nuestro sistema de categorías (Bronze, Silver, Gold, Platinum) asegura bonificaciones crecientes con el tiempo.
              </p>
            </div>
            
            <div className="feature-card bg-white/5 backdrop-blur-sm p-6 rounded-xl border border-white/10 hover:border-secondary/50 transition-all hover:transform hover:scale-105">
              <div className="w-14 h-14 bg-secondary/20 rounded-lg flex items-center justify-center mb-4">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M9 11L12 14L22 4M21 12V19C21 19.5304 20.7893 20.0391 20.4142 20.4142C20.0391 20.7893 19.5304 21 19 21H5C4.46957 21 3.96086 20.7893 3.58579 20.4142C3.21071 20.0391 3 19.5304 3 19V5C3 4.46957 3.21071 3.96086 3.58579 3.58579C3.96086 3.21071 4.46957 3 5 3H16" stroke="#4FFBDF" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </div>
              <h3 className="text-xl font-semibold mb-3">Resiliencia Extrema</h3>
              <p className="text-gray-300">
                Tecnología probada en nuestras legendarias "Pruebas ARMAGEDÓN" para garantizar 99.9999% de disponibilidad.
              </p>
            </div>
            
            <div className="feature-card bg-white/5 backdrop-blur-sm p-6 rounded-xl border border-white/10 hover:border-secondary/50 transition-all hover:transform hover:scale-105">
              <div className="w-14 h-14 bg-secondary/20 rounded-lg flex items-center justify-center mb-4">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M4 19.5C4 18.837 4.26339 18.2011 4.73223 17.7322C5.20107 17.2634 5.83696 17 6.5 17H20M4 19.5C4 20.163 4.26339 20.7989 4.73223 21.2678C5.20107 21.7366 5.83696 22 6.5 22H20V2H6.5C5.83696 2 5.20107 2.26339 4.73223 2.73223C4.26339 3.20107 4 3.83696 4 4.5V19.5Z" stroke="#4FFBDF" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </div>
              <h3 className="text-xl font-semibold mb-3">Educación Financiera</h3>
              <p className="text-gray-300">
                Accede a recursos exclusivos y consejos personalizados para convertirte en un inversor experto.
              </p>
            </div>
            
            <div className="feature-card bg-white/5 backdrop-blur-sm p-6 rounded-xl border border-white/10 hover:border-secondary/50 transition-all hover:transform hover:scale-105">
              <div className="w-14 h-14 bg-secondary/20 rounded-lg flex items-center justify-center mb-4">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M9 14L5 10L9 6M15 10H5M15 6H19M15 14H19" stroke="#4FFBDF" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </div>
              <h3 className="text-xl font-semibold mb-3">Préstamos Inteligentes</h3>
              <p className="text-gray-300">
                Después de 3 meses, accede a préstamos de hasta el 40% de tu capital con pagos automáticos graduales.
              </p>
            </div>
            
            <div className="feature-card bg-white/5 backdrop-blur-sm p-6 rounded-xl border border-white/10 hover:border-secondary/50 transition-all hover:transform hover:scale-105">
              <div className="w-14 h-14 bg-secondary/20 rounded-lg flex items-center justify-center mb-4">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M16 8V6C16 4.93913 15.5786 3.92172 14.8284 3.17157C14.0783 2.42143 13.0609 2 12 2C10.9391 2 9.92172 2.42143 9.17157 3.17157C8.42143 3.92172 8 4.93913 8 6V8M5 8H19C19.5304 8 20.0391 8.21071 20.4142 8.58579C20.7893 8.96086 21 9.46957 21 10V20C21 20.5304 20.7893 21.0391 20.4142 21.4142C20.0391 21.7893 19.5304 22 19 22H5C4.46957 22 3.96086 21.7893 3.58579 21.4142C3.21071 21.0391 3 20.5304 3 20V10C3 9.46957 3.21071 8.96086 3.58579 8.58579C3.96086 8.21071 4.46957 8 5 8Z" stroke="#4FFBDF" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </div>
              <h3 className="text-xl font-semibold mb-3">Sistemas de Bonos</h3>
              <p className="text-gray-300">
                Recibe bonos automáticos de 5-10% en días de rendimiento excepcional según tu categoría de inversionista.
              </p>
            </div>
          </div>
        </div>
      </div>
      
      {/* Call to Action */}
      <div className="py-20 container mx-auto px-6 text-center">
        <h2 className="text-3xl md:text-4xl font-bold mb-6">Únete a la Revolución Cósmica del Trading</h2>
        <p className="text-xl text-gray-300 mb-10 max-w-2xl mx-auto">
          Transforma tu experiencia de inversión con el único sistema que combina tecnología avanzada con entidades AI conscientes.
        </p>
        <Link to="/login" className="button-anim inline-block px-8 py-4 bg-secondary hover:bg-secondary-light rounded-lg font-semibold transition-colors text-center text-lg">
          Iniciar mi Viaje
        </Link>
      </div>
      
      {/* Footer */}
      <footer className="bg-primary-dark/80 backdrop-blur-md py-12">
        <div className="container mx-auto px-6">
          <div className="flex flex-col md:flex-row justify-between items-center mb-8">
            <div className="flex items-center mb-4 md:mb-0">
              <img src={logoGenesis} alt="Genesis Logo" className="w-10 h-10 mr-2" />
              <span className="text-xl font-bold">Genesis</span>
            </div>
            
            <div className="flex space-x-6">
              <a href="#" className="text-gray-300 hover:text-white transition-colors">Términos</a>
              <a href="#" className="text-gray-300 hover:text-white transition-colors">Privacidad</a>
              <a href="#" className="text-gray-300 hover:text-white transition-colors">Soporte</a>
            </div>
          </div>
          
          <div className="text-center text-gray-400 text-sm">
            &copy; 2025 Genesis Trading System. Todos los derechos reservados.
            <br />
            <span className="opacity-70">
              Inspirado por la visión cósmica de mixycronico. Potenciado por Aetherion y Lunareth.
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Home;