import { useState, useEffect } from 'react';
import { FiBell, FiSearch, FiMoon, FiSun } from 'react-icons/fi';
import gsap from 'gsap';

const Header = ({ title }) => {
  const [darkMode, setDarkMode] = useState(localStorage.getItem('darkMode') === 'true');
  const [notifications, setNotifications] = useState([
    { id: 1, text: 'Bono aplicado: +$75.50', time: '10 min', read: false },
    { id: 2, text: 'Nuevo rendimiento registrado', time: '1 hora', read: false },
    { id: 3, text: 'Actualización de categoría disponible', time: '2 días', read: true }
  ]);
  const [showNotifications, setShowNotifications] = useState(false);

  // Manejo del tema oscuro/claro
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
    localStorage.setItem('darkMode', darkMode);
  }, [darkMode]);

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
  };

  // Animación del título
  useEffect(() => {
    gsap.fromTo(
      '.page-title',
      { opacity: 0, y: -20 },
      { opacity: 1, y: 0, duration: 0.8, ease: 'power3.out' }
    );
  }, [title]);

  // Manejo de notificaciones
  const toggleNotifications = () => {
    setShowNotifications(!showNotifications);
    
    if (!showNotifications) {
      // Animar entrada de notificaciones
      gsap.fromTo(
        '.notification-panel',
        { opacity: 0, y: -10 },
        { opacity: 1, y: 0, duration: 0.3, ease: 'power2.out' }
      );
    }
  };

  const markAsRead = (id) => {
    setNotifications(
      notifications.map(notification =>
        notification.id === id
          ? { ...notification, read: true }
          : notification
      )
    );
  };

  const unreadCount = notifications.filter(n => !n.read).length;

  return (
    <div className="z-10 px-4 py-3 flex justify-between items-center bg-white/10 dark:bg-primary-dark/20 backdrop-blur-md shadow-sm">
      {/* Título de la página */}
      <h1 className="page-title text-2xl font-bold text-primary-dark dark:text-white">{title}</h1>
      
      {/* Grupo derecho */}
      <div className="flex items-center space-x-4">
        {/* Barra de búsqueda */}
        <div className="hidden md:flex items-center relative">
          <input
            type="text"
            placeholder="Buscar..."
            className="pl-9 pr-4 py-2 rounded-full bg-gray-100 dark:bg-primary-light/20 text-gray-700 dark:text-gray-300 focus:outline-none focus:ring-2 focus:ring-secondary/50 w-48"
          />
          <FiSearch className="absolute left-3 text-gray-500 dark:text-gray-400" />
        </div>
        
        {/* Toggle de tema oscuro/claro */}
        <button
          onClick={toggleDarkMode}
          className="p-2 rounded-full hover:bg-gray-200 dark:hover:bg-primary-light/20 transition-colors duration-200"
        >
          {darkMode ? (
            <FiSun className="text-yellow-400" size={20} />
          ) : (
            <FiMoon className="text-primary-dark" size={20} />
          )}
        </button>
        
        {/* Notificaciones */}
        <div className="relative">
          <button
            onClick={toggleNotifications}
            className="p-2 rounded-full hover:bg-gray-200 dark:hover:bg-primary-light/20 transition-colors duration-200 relative"
          >
            <FiBell className="text-primary-dark dark:text-white" size={20} />
            {unreadCount > 0 && (
              <span className="absolute top-0 right-0 w-4 h-4 bg-red-500 rounded-full text-white text-xs flex items-center justify-center">
                {unreadCount}
              </span>
            )}
          </button>
          
          {/* Panel de notificaciones */}
          {showNotifications && (
            <div className="notification-panel absolute right-0 mt-2 w-72 bg-white dark:bg-primary-dark shadow-lg rounded-lg overflow-hidden border border-gray-200 dark:border-primary-light/20 z-30">
              <div className="p-3 border-b border-gray-200 dark:border-primary-light/20 flex justify-between items-center">
                <h3 className="font-medium text-primary-dark dark:text-white">Notificaciones</h3>
                <span className="text-xs text-blue-500 cursor-pointer hover:text-blue-700">
                  Marcar todas como leídas
                </span>
              </div>
              
              <div className="max-h-96 overflow-y-auto">
                {notifications.length > 0 ? (
                  notifications.map(notification => (
                    <div
                      key={notification.id}
                      className={`p-3 border-b border-gray-100 dark:border-primary-light/10 hover:bg-gray-50 dark:hover:bg-primary-light/10 transition-colors duration-150 cursor-pointer ${
                        !notification.read ? 'bg-blue-50 dark:bg-blue-900/20' : ''
                      }`}
                      onClick={() => markAsRead(notification.id)}
                    >
                      <div className="flex justify-between items-start">
                        <p className="text-sm text-gray-800 dark:text-gray-200">
                          {notification.text}
                        </p>
                        {!notification.read && (
                          <span className="w-2 h-2 rounded-full bg-blue-500 mt-1"></span>
                        )}
                      </div>
                      <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                        Hace {notification.time}
                      </p>
                    </div>
                  ))
                ) : (
                  <div className="p-4 text-center text-gray-500 dark:text-gray-400">
                    No hay notificaciones
                  </div>
                )}
              </div>
              
              <div className="p-2 text-center border-t border-gray-200 dark:border-primary-light/20">
                <button className="text-sm text-blue-500 hover:text-blue-700">
                  Ver todas
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Header;