import { useState, useEffect } from 'react';
import { Outlet } from 'react-router-dom';
import Header from './Header';
import Sidebar from './Sidebar';
import MobileNavbar from './MobileNavbar';
import CosmicChat from './CosmicChat';
import { FiMessageCircle } from 'react-icons/fi';

const MainLayout = ({ user }) => {
  const [sidebarOpen, setSidebarOpen] = useState(window.innerWidth >= 768);
  const [chatOpen, setChatOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);
  
  // Detectar cambios de tamaño de pantalla para ajustar sidebar automáticamente
  useEffect(() => {
    const handleResize = () => {
      const mobile = window.innerWidth < 768;
      setIsMobile(mobile);
      
      // En pantalla pequeña, cerrar sidebar automáticamente
      if (mobile && sidebarOpen) {
        setSidebarOpen(false);
      } 
      // En pantalla grande, abrir sidebar automáticamente si estaba cerrado
      else if (!mobile && !sidebarOpen) {
        setSidebarOpen(true);
      }
    };
    
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [sidebarOpen]);
  
  const toggleSidebar = () => {
    setSidebarOpen(prev => !prev);
  };
  
  const toggleChat = () => {
    setChatOpen(prev => !prev);
  };

  return (
    <div className="flex h-screen bg-cosmic-dark text-white overflow-hidden">
      {/* Overlay para móvil cuando sidebar está abierto */}
      {isMobile && sidebarOpen && (
        <div 
          className="fixed inset-0 bg-cosmic-darkest bg-opacity-70 z-10 transition-opacity"
          onClick={toggleSidebar}
        />
      )}
      
      {/* Sidebar */}
      <Sidebar open={sidebarOpen} user={user} isMobile={isMobile} />
      
      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header toggleSidebar={toggleSidebar} sidebarOpen={sidebarOpen} user={user} />
        
        {/* Main Content Area */}
        <main className={`flex-1 overflow-auto bg-cosmic-darkest ${isMobile ? 'pb-20' : ''}`}>
          <Outlet />
        </main>
      </div>
      
      {/* Chat Button - ajustado para que no se superponga con la barra de navegación móvil */}
      <button
        className={`fixed right-4 z-30 cosmic-button-floating w-12 h-12 flex items-center justify-center text-xl shadow-lg ${isMobile ? 'bottom-20' : 'bottom-4'}`}
        onClick={toggleChat}
        aria-label="Chat Cósmico"
      >
        <FiMessageCircle />
      </button>
      
      {/* Cosmic Chat */}
      <CosmicChat open={chatOpen} toggleChat={toggleChat} isMobile={isMobile} />
      
      {/* Mobile Navigation Bar */}
      {isMobile && <MobileNavbar user={user} />}
      
      {/* Radial gradients for cosmic effect */}
      <div className="fixed top-0 left-0 w-full h-full pointer-events-none overflow-hidden z-0">
        <div className="absolute top-0 left-0 w-1/3 h-1/3 bg-cosmic-blue opacity-5 rounded-full filter blur-3xl"></div>
        <div className="absolute bottom-0 right-0 w-1/2 h-1/2 bg-cosmic-highlight opacity-5 rounded-full filter blur-3xl"></div>
        <div className="absolute top-1/3 right-1/4 w-1/4 h-1/4 bg-cosmic-green opacity-5 rounded-full filter blur-3xl"></div>
      </div>
    </div>
  );
};

export default MainLayout;