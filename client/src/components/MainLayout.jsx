import { useState } from 'react';
import { Outlet } from 'react-router-dom';
import Header from './Header';
import Sidebar from './Sidebar';
import CosmicChat from './CosmicChat';
import { FiMessageCircle } from 'react-icons/fi';

const MainLayout = ({ user }) => {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [chatOpen, setChatOpen] = useState(false);
  
  const toggleSidebar = () => {
    setSidebarOpen(prev => !prev);
  };
  
  const toggleChat = () => {
    setChatOpen(prev => !prev);
  };

  return (
    <div className="flex h-screen bg-cosmic-dark text-white overflow-hidden">
      {/* Sidebar */}
      <Sidebar open={sidebarOpen} user={user} />
      
      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header toggleSidebar={toggleSidebar} sidebarOpen={sidebarOpen} user={user} />
        
        {/* Main Content Area */}
        <main className="flex-1 overflow-auto bg-cosmic-darkest">
          <Outlet />
        </main>
      </div>
      
      {/* Chat Button */}
      <button
        className="fixed bottom-4 right-4 z-30 cosmic-button-floating w-12 h-12 flex items-center justify-center text-xl shadow-lg"
        onClick={toggleChat}
      >
        <FiMessageCircle />
      </button>
      
      {/* Cosmic Chat */}
      <CosmicChat open={chatOpen} toggleChat={toggleChat} />
      
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