import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { FaSignOutAlt, FaUser, FaBell } from 'react-icons/fa';

const Navbar = ({ user, onLogout }) => {
  return (
    <nav className="navbar">
      <div className="navbar-brand">
        <motion.div 
          className="navbar-logo"
          animate={{ 
            rotate: 360,
          }}
          transition={{ 
            duration: 20, 
            repeat: Infinity, 
            ease: "linear" 
          }}
        >
          G
        </motion.div>
        <h1>Genesis</h1>
      </div>
      
      <div className="navbar-menu">
        <motion.div 
          className="navbar-notifications"
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
        >
          <FaBell />
          <span className="notification-badge">0</span>
        </motion.div>
        
        <motion.div 
          className="navbar-user"
          whileHover={{ scale: 1.05 }}
        >
          <span className="user-name">{user?.username}</span>
          <div className="user-avatar">
            <FaUser />
          </div>
        </motion.div>
        
        <motion.button 
          className="logout-button"
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          onClick={onLogout}
        >
          <FaSignOutAlt />
          <span>Salir</span>
        </motion.button>
      </div>
    </nav>
  );
};

export default Navbar;