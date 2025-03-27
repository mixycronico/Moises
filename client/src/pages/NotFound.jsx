import { Link } from 'react-router-dom';
import { useEffect } from 'react';
import gsap from 'gsap';
import logoGenesis from '../assets/logo-genesis.svg';

const NotFound = () => {
  useEffect(() => {
    // Animación del contenido
    gsap.fromTo(
      '.content',
      { opacity: 0, y: 20 },
      { opacity: 1, y: 0, duration: 0.8, ease: 'power2.out' }
    );
    
    // Animación del logo
    gsap.fromTo(
      '.logo',
      { rotation: 0 },
      { rotation: 360, duration: 20, repeat: -1, ease: 'linear' }
    );
    
    // Animación de partículas
    const particles = document.querySelectorAll('.particle');
    particles.forEach((particle) => {
      gsap.to(particle, {
        x: `random(-100, 100)`,
        y: `random(-100, 100)`,
        opacity: `random(0.3, 0.8)`,
        duration: `random(3, 6)`,
        repeat: -1,
        yoyo: true,
        ease: 'none',
      });
    });
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-light to-blue-900 flex items-center justify-center relative px-4">
      {/* Partículas de fondo */}
      {Array.from({ length: 15 }).map((_, i) => (
        <div
          key={i}
          className="particle absolute w-2 h-2 bg-secondary rounded-full"
          style={{
            top: `${Math.random() * 100}%`,
            left: `${Math.random() * 100}%`,
            opacity: 0.5,
          }}
        />
      ))}

      <div className="content text-center">
        <img src={logoGenesis} alt="Genesis Logo" className="logo w-32 mx-auto mb-8" />
        
        <h1 className="text-6xl font-bold text-white mb-4">404</h1>
        <h2 className="text-2xl text-gray-300 mb-6">Página no encontrada</h2>
        <p className="text-gray-400 mb-8 max-w-md">
          La ruta que buscas no existe en esta dimensión del sistema Genesis. 
          Aetherion y Lunareth no pudieron encontrar este recurso.
        </p>
        
        <Link to="/" className="neon-button inline-block">
          Volver al Inicio
        </Link>
      </div>
    </div>
  );
};

export default NotFound;