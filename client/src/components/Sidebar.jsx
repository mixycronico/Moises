import { useEffect, useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useNavigate } from 'react-router-dom';
import logoGenesis from '../assets/logo-genesis.svg';
import { FiHome, FiPieChart, FiDollarSign, FiMessageCircle, FiSettings, FiLogOut, FiMenu, FiX } from 'react-icons/fi';
import gsap from 'gsap';

const Sidebar = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [isOpen, setIsOpen] = useState(true);
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);

  // Detectar cambio de tamaño de ventana para modo responsive
  useEffect(() => {
    const handleResize = () => {
      setIsMobile(window.innerWidth < 768);
      if (window.innerWidth >= 768) {
        setIsOpen(true);
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Manejo de animaciones GSAP
  useEffect(() => {
    // Animación del logo
    gsap.fromTo(
      '.sidebar-logo',
      { rotation: 0 },
      { rotation: 360, duration: 30, repeat: -1, ease: 'linear' }
    );
    
    // Animación de los enlaces
    gsap.fromTo(
      '.nav-link',
      { opacity: 0, x: -20 },
      { opacity: 1, x: 0, duration: 0.5, stagger: 0.1, ease: 'power2.out' }
    );
  }, [isOpen]);

  // Navegación
  const handleLogout = () => {
    // Eliminar token y datos de usuario
    localStorage.removeItem('userToken');
    localStorage.removeItem('userData');
    // Redirigir al login
    navigate('/login');
  };

  const toggleSidebar = () => {
    setIsOpen(!isOpen);
  };

  // Array de enlaces para el menú
  const navItems = [
    { path: '/dashboard', icon: <FiHome size={20} />, label: 'Inicio' },
    { path: '/stats', icon: <FiPieChart size={20} />, label: 'Estadísticas' },
    { path: '/investments', icon: <FiDollarSign size={20} />, label: 'Inversiones' },
    { path: '/chat', icon: <FiMessageCircle size={20} />, label: 'Chat Cósmico' },
    { path: '/settings', icon: <FiSettings size={20} />, label: 'Configuración' },
  ];

  return (
    <>
      {/* Botón de hamburguesa para móvil */}
      {isMobile && (
        <button
          className="fixed top-4 left-4 z-30 p-2 rounded-full bg-primary-dark text-white"
          onClick={toggleSidebar}
        >
          {isOpen ? <FiX size={24} /> : <FiMenu size={24} />}
        </button>
      )}

      {/* Barra lateral */}
      <div
        className={`fixed inset-y-0 left-0 z-20 w-64 bg-primary-dark text-white transform transition-transform duration-300 ease-in-out ${
          isOpen ? 'translate-x-0' : '-translate-x-full'
        } ${isMobile ? 'shadow-lg' : ''}`}
      >
        <div className="p-4 space-y-6">
          {/* Logo y título */}
          <div className="flex items-center justify-center py-2">
            <img
              src={logoGenesis}
              alt="Genesis Logo"
              className="sidebar-logo w-12 h-12 mr-2"
            />
            <span className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-500">
              Genesis
            </span>
          </div>

          {/* Menú de navegación */}
          <nav className="mt-8 space-y-2">
            {navItems.map((item) => (
              <Link
                key={item.path}
                to={item.path}
                className={`nav-link flex items-center p-2 rounded-lg transition-colors duration-200 ${
                  location.pathname === item.path
                    ? 'bg-secondary/30 text-white'
                    : 'text-gray-300 hover:bg-secondary/20'
                }`}
              >
                <span className="mr-3">{item.icon}</span>
                <span>{item.label}</span>
                
                {/* Indicador de selección activa */}
                {location.pathname === item.path && (
                  <span className="ml-auto w-1.5 h-5 rounded-full bg-secondary"></span>
                )}
              </Link>
            ))}
          </nav>

          {/* Cerrar sesión */}
          <div className="pt-8">
            <button
              onClick={handleLogout}
              className="flex items-center w-full p-2 rounded-lg text-gray-300 hover:bg-red-800/30 transition-colors duration-200"
            >
              <span className="mr-3">
                <FiLogOut size={20} />
              </span>
              <span>Cerrar Sesión</span>
            </button>
          </div>
        </div>

        {/* Sección de información del usuario */}
        <div className="absolute bottom-0 left-0 right-0 p-4">
          <div className="flex items-center bg-primary-light/30 rounded-lg p-2">
            <div className="w-8 h-8 rounded-full bg-secondary flex items-center justify-center text-white font-bold">
              U
            </div>
            <div className="ml-2">
              <div className="text-sm font-medium">Usuario</div>
              <div className="text-xs text-gray-400">Inversionista</div>
            </div>
          </div>
        </div>
      </div>

      {/* Overlay para móvil cuando el sidebar está abierto */}
      {isMobile && isOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-10"
          onClick={toggleSidebar}
        ></div>
      )}
    </>
  );
};

export default Sidebar;