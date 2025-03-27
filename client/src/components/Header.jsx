import { useState, useRef, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  FiMenu, FiX, FiUser, FiSettings, FiLogOut, 
  FiBell, FiHelpCircle, FiChevronDown 
} from 'react-icons/fi';
import axios from 'axios';

const Header = ({ toggleSidebar, sidebarOpen, user }) => {
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const [notificationsOpen, setNotificationsOpen] = useState(false);
  const [notifications, setNotifications] = useState([]);
  const userMenuRef = useRef(null);
  const notificationRef = useRef(null);
  const navigate = useNavigate();
  
  // Cargar notificaciones (simulado)
  useEffect(() => {
    const fetchNotifications = async () => {
      try {
        const response = await axios.get('/api/notifications');
        if (response.data.success) {
          setNotifications(response.data.notifications);
        }
      } catch (error) {
        console.error('Error loading notifications:', error);
        // Notificaciones de prueba en caso de error
        setNotifications([
          {
            id: 1,
            type: 'info',
            message: 'El sistema ha completado el análisis predictivo',
            timestamp: '2025-03-27T10:30:00Z',
            read: false
          },
          {
            id: 2,
            type: 'success',
            message: 'Tu operación de BTC/USDT ha generado +$125.32',
            timestamp: '2025-03-27T09:15:00Z',
            read: true
          },
          {
            id: 3,
            type: 'warning',
            message: 'Actualización de sistema programada para mañana',
            timestamp: '2025-03-26T16:45:00Z',
            read: true
          }
        ]);
      }
    };
    
    fetchNotifications();
  }, []);
  
  // Cerrar menús al hacer clic fuera
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (userMenuRef.current && !userMenuRef.current.contains(event.target)) {
        setUserMenuOpen(false);
      }
      if (notificationRef.current && !notificationRef.current.contains(event.target)) {
        setNotificationsOpen(false);
      }
    };
    
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);
  
  // Logout
  const handleLogout = async () => {
    try {
      await axios.post('/api/auth/logout');
      navigate('/login');
    } catch (error) {
      console.error('Error during logout:', error);
      // Navegación de emergencia
      navigate('/login');
    }
  };
  
  // Marcar notificación como leída
  const markAsRead = async (id) => {
    try {
      await axios.post(`/api/notifications/${id}/read`);
      setNotifications(prev => 
        prev.map(notif => 
          notif.id === id ? { ...notif, read: true } : notif
        )
      );
    } catch (error) {
      console.error('Error marking notification as read:', error);
      // Actualización optimista en caso de error
      setNotifications(prev => 
        prev.map(notif => 
          notif.id === id ? { ...notif, read: true } : notif
        )
      );
    }
  };
  
  // Variantes para animaciones
  const menuVariants = {
    hidden: { opacity: 0, y: -5 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: { duration: 0.2 }
    },
    exit: { 
      opacity: 0,
      y: -5,
      transition: { duration: 0.2 }
    }
  };
  
  // Formatear fecha de notificaciones
  const formatNotificationTime = (timestamp) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMinutes = Math.floor((now - date) / (1000 * 60));
    
    if (diffMinutes < 60) {
      return `Hace ${diffMinutes} min`;
    } else if (diffMinutes < 24 * 60) {
      const hours = Math.floor(diffMinutes / 60);
      return `Hace ${hours} ${hours === 1 ? 'hora' : 'horas'}`;
    } else {
      return date.toLocaleDateString('es-ES', { 
        day: '2-digit', 
        month: '2-digit'
      });
    }
  };
  
  // Obtener color según tipo de notificación
  const getNotificationColor = (type) => {
    switch (type) {
      case 'success':
        return 'bg-cosmic-green/20 text-cosmic-green border-cosmic-green/30';
      case 'warning':
        return 'bg-cosmic-yellow/20 text-cosmic-yellow border-cosmic-yellow/30';
      case 'error':
        return 'bg-cosmic-red/20 text-cosmic-red border-cosmic-red/30';
      case 'info':
      default:
        return 'bg-cosmic-blue/20 text-cosmic-blue border-cosmic-blue/30';
    }
  };
  
  // Contar notificaciones no leídas
  const unreadCount = notifications.filter(notif => !notif.read).length;

  return (
    <header className="h-16 bg-cosmic-primary/20 backdrop-blur-md border-b border-cosmic-primary/30 flex items-center justify-between px-4 relative z-10">
      {/* Left section with toggle and title */}
      <div className="flex items-center">
        <button 
          onClick={toggleSidebar}
          className="p-2 mr-4 rounded-full hover:bg-cosmic-primary/30 text-cosmic-glow transition-colors"
          aria-label={sidebarOpen ? 'Ocultar menú' : 'Mostrar menú'}
        >
          {sidebarOpen ? <FiX className="h-5 w-5" /> : <FiMenu className="h-5 w-5" />}
        </button>
        
        <h1 className="text-lg font-semibold cosmic-gradient-text hidden sm:block">
          Sistema Genesis
        </h1>
      </div>
      
      {/* Right section with user info and notifications */}
      <div className="flex items-center space-x-3">
        {/* Notification bell */}
        <div className="relative" ref={notificationRef}>
          <button 
            className="p-2 rounded-full hover:bg-cosmic-primary/30 relative"
            onClick={() => setNotificationsOpen(!notificationsOpen)}
          >
            <FiBell className="h-5 w-5 text-cosmic-glow" />
            {unreadCount > 0 && (
              <span className="absolute top-1 right-1 w-2 h-2 bg-cosmic-red rounded-full"></span>
            )}
          </button>
          
          {/* Notifications dropdown */}
          <AnimatePresence>
            {notificationsOpen && (
              <motion.div
                className="absolute right-0 mt-2 w-80 cosmic-card border border-cosmic-primary/30 shadow-xl z-30"
                variants={menuVariants}
                initial="hidden"
                animate="visible"
                exit="exit"
              >
                <div className="p-3 border-b border-cosmic-primary/30 flex justify-between items-center">
                  <h3 className="font-medium">Notificaciones</h3>
                  {unreadCount > 0 && (
                    <span className="text-xs bg-cosmic-primary/20 px-2 py-1 rounded-full">
                      {unreadCount} no {unreadCount === 1 ? 'leída' : 'leídas'}
                    </span>
                  )}
                </div>
                
                <div className="max-h-80 overflow-y-auto p-1">
                  {notifications.length > 0 ? (
                    notifications.map((notification) => (
                      <div 
                        key={notification.id} 
                        className={`p-3 mb-1 rounded border ${getNotificationColor(notification.type)} ${!notification.read ? 'bg-opacity-30' : 'opacity-75'}`}
                        onClick={() => markAsRead(notification.id)}
                      >
                        <div className="flex justify-between items-start mb-1">
                          <p className="font-medium text-sm">
                            {notification.message}
                          </p>
                          {!notification.read && (
                            <span className="w-2 h-2 rounded-full bg-current"></span>
                          )}
                        </div>
                        <div className="text-xs opacity-80">
                          {formatNotificationTime(notification.timestamp)}
                        </div>
                      </div>
                    ))
                  ) : (
                    <div className="p-4 text-center text-gray-400">
                      No hay notificaciones
                    </div>
                  )}
                </div>
                
                <div className="p-2 border-t border-cosmic-primary/30 text-center">
                  <button className="text-sm text-cosmic-blue hover:text-cosmic-glow transition-colors">
                    Ver todas las notificaciones
                  </button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
        
        {/* User menu */}
        <div className="relative" ref={userMenuRef}>
          <button 
            className="flex items-center space-x-2 p-1 rounded-full hover:bg-cosmic-primary/30 transition-colors"
            onClick={() => setUserMenuOpen(!userMenuOpen)}
          >
            <div className="w-8 h-8 bg-cosmic-primary/30 rounded-full flex items-center justify-center text-cosmic-glow">
              {user?.username?.charAt(0).toUpperCase() || 'U'}
            </div>
            <span className="hidden md:block text-sm">{user?.username || 'Usuario'}</span>
            <FiChevronDown className={`h-4 w-4 transition-transform ${userMenuOpen ? 'rotate-180' : ''}`} />
          </button>
          
          {/* User dropdown menu */}
          <AnimatePresence>
            {userMenuOpen && (
              <motion.div
                className="absolute right-0 mt-2 w-48 cosmic-card border border-cosmic-primary/30 shadow-xl z-30"
                variants={menuVariants}
                initial="hidden"
                animate="visible"
                exit="exit"
              >
                <div className="p-3 border-b border-cosmic-primary/30">
                  <p className="font-medium">{user?.username || 'Usuario'}</p>
                  <p className="text-xs text-gray-400">{user?.email || 'usuario@email.com'}</p>
                </div>
                
                <div className="py-1">
                  <Link 
                    to="/profile" 
                    className="flex items-center px-4 py-2 hover:bg-cosmic-primary/20 transition-colors"
                    onClick={() => setUserMenuOpen(false)}
                  >
                    <FiUser className="mr-2 h-4 w-4" />
                    <span>Mi Perfil</span>
                  </Link>
                  
                  <Link 
                    to="/settings" 
                    className="flex items-center px-4 py-2 hover:bg-cosmic-primary/20 transition-colors"
                    onClick={() => setUserMenuOpen(false)}
                  >
                    <FiSettings className="mr-2 h-4 w-4" />
                    <span>Configuración</span>
                  </Link>
                  
                  <Link 
                    to="/help" 
                    className="flex items-center px-4 py-2 hover:bg-cosmic-primary/20 transition-colors"
                    onClick={() => setUserMenuOpen(false)}
                  >
                    <FiHelpCircle className="mr-2 h-4 w-4" />
                    <span>Ayuda</span>
                  </Link>
                </div>
                
                <div className="py-1 border-t border-cosmic-primary/30">
                  <button 
                    className="flex items-center w-full text-left px-4 py-2 text-cosmic-red hover:bg-cosmic-primary/20 transition-colors"
                    onClick={handleLogout}
                  >
                    <FiLogOut className="mr-2 h-4 w-4" />
                    <span>Cerrar Sesión</span>
                  </button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </header>
  );
};

export default Header;