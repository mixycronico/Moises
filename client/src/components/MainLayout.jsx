import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Sidebar from './Sidebar';
import Header from './Header';
import CosmicChat from './CosmicChat';
import gsap from 'gsap';

const MainLayout = ({ children, title = 'Dashboard' }) => {
  const navigate = useNavigate();

  // Comprobar autenticación al cargar
  useEffect(() => {
    const token = localStorage.getItem('userToken');
    if (!token) {
      // Redirigir a login si no hay token
      navigate('/login');
    }
  }, [navigate]);

  // Animaciones GSAP para el contenido
  useEffect(() => {
    // Animación del contenido principal
    gsap.fromTo(
      '.main-content',
      { opacity: 0 },
      { opacity: 1, duration: 0.8, ease: 'power2.out' }
    );
    
    // Animación de las partículas de fondo
    const particles = document.querySelectorAll('.bg-particle');
    particles.forEach((particle) => {
      gsap.to(particle, {
        x: 'random(-100, 100)',
        y: 'random(-100, 100)',
        opacity: 'random(0.1, 0.5)',
        duration: 'random(15, 30)',
        repeat: -1,
        yoyo: true,
        ease: 'none',
      });
    });
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-primary-dark flex">
      {/* Partículas de fondo */}
      {Array.from({ length: 15 }).map((_, i) => (
        <div
          key={i}
          className="bg-particle absolute opacity-20 dark:opacity-30"
          style={{
            top: `${Math.random() * 100}%`,
            left: `${Math.random() * 100}%`,
            width: `${Math.random() * 10 + 5}px`,
            height: `${Math.random() * 10 + 5}px`,
            borderRadius: '50%',
            background: `radial-gradient(circle, rgba(140,82,255,1) 0%, rgba(99,102,241,0) 70%)`,
          }}
        />
      ))}
      
      {/* Barra lateral */}
      <Sidebar />
      
      {/* Contenido principal */}
      <div className="flex-1 ml-64 flex flex-col relative">
        <Header title={title} />
        
        <main className="main-content flex-1 p-4 overflow-auto">
          {children}
        </main>
        
        {/* Componente de chat flotante */}
        <CosmicChat />
      </div>
    </div>
  );
};

export default MainLayout;