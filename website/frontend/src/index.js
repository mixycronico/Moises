/**
 * Punto de entrada principal para el frontend del Proyecto Genesis
 */
import React from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import App from './App';
import './styles/global.css';

// Crear un fondo de estrellas aleatoriamente
const createStarsBackground = () => {
  const starsContainer = document.createElement('div');
  starsContainer.className = 'stars-background';
  
  // Crear estrellas aleatorias
  const numStars = Math.floor(window.innerWidth * window.innerHeight / 1000);
  
  for (let i = 0; i < numStars; i++) {
    const star = document.createElement('div');
    star.className = 'star';
    
    // Posición aleatoria
    star.style.left = `${Math.random() * 100}%`;
    star.style.top = `${Math.random() * 100}%`;
    
    // Tamaño aleatorio
    const size = Math.random() * 3 + 1;
    star.style.width = `${size}px`;
    star.style.height = `${size}px`;
    
    // Brillo aleatorio
    star.style.opacity = Math.random() * 0.7 + 0.3;
    
    // Velocidad de parpadeo aleatoria
    const animationDuration = Math.random() * 4 + 2;
    star.style.animation = `twinkle ${animationDuration}s infinite`;
    
    // Añadir al contenedor
    starsContainer.appendChild(star);
  }
  
  document.body.appendChild(starsContainer);
};

// Renderizar la aplicación
const init = () => {
  const rootElement = document.getElementById('root');
  const root = createRoot(rootElement);
  
  root.render(
    <React.StrictMode>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </React.StrictMode>
  );
  
  // Crear fondo de estrellas después de que React se haya cargado
  createStarsBackground();
};

// Iniciar la aplicación cuando el DOM esté listo
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}