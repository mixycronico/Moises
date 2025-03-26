/**
 * Efectos cósmicos para Proto Genesis
 * Este script añade efectos visuales avanzados para mejorar la experiencia de usuario
 */

document.addEventListener('DOMContentLoaded', () => {
    console.log('✨ Iniciando efectos cósmicos...');
    
    // Crear el fondo de nebulosa cósmica si no existe
    if (!document.querySelector('.cosmic-nebula')) {
        const nebula = document.createElement('div');
        nebula.className = 'cosmic-nebula';
        document.body.prepend(nebula);
    }
    
    // Añadir partículas flotantes
    createFloatingParticles();
    
    // Añadir efectos de resplandor al pasar el mouse
    addHoverGlowEffects();
    
    // Añadir efectos de pulsación
    addPulseEffects();
    
    console.log('✨ Efectos cósmicos iniciados correctamente');
});

/**
 * Crea partículas flotantes que se mueven suavemente por la pantalla
 */
function createFloatingParticles() {
    const container = document.createElement('div');
    container.className = 'particles-container';
    document.body.appendChild(container);
    
    // Crear 25 partículas con diferentes tamaños, colores y velocidades
    for (let i = 0; i < 25; i++) {
        createParticle(container);
    }
}

/**
 * Crea una partícula individual con propiedades aleatorias
 */
function createParticle(container) {
    const particle = document.createElement('div');
    particle.className = 'particle';
    
    // Tamaño aleatorio entre 2 y 5 píxeles
    const size = Math.random() * 3 + 2;
    particle.style.width = `${size}px`;
    particle.style.height = `${size}px`;
    
    // Posición inicial aleatoria
    particle.style.left = `${Math.random() * 100}vw`;
    particle.style.top = `${Math.random() * 100}vh`;
    
    // Colores disponibles
    const colors = [
        'rgba(0, 234, 255, 0.8)',  // Cian
        'rgba(0, 255, 170, 0.8)',  // Verde-cian
        'rgba(255, 238, 0, 0.5)',  // Amarillo
        'rgba(255, 255, 255, 0.8)' // Blanco
    ];
    
    // Color aleatorio de la lista
    const color = colors[Math.floor(Math.random() * colors.length)];
    particle.style.backgroundColor = color;
    particle.style.boxShadow = `0 0 ${size * 2}px ${color}`;
    
    // Duración aleatoria para la animación (entre 15 y 30 segundos)
    const duration = Math.random() * 15 + 15;
    particle.style.animation = `float ${duration}s linear infinite`;
    
    // Retraso aleatorio para la animación
    particle.style.animationDelay = `${Math.random() * 5}s`;
    
    // Añadir al contenedor
    container.appendChild(particle);
}

/**
 * Añade efectos de resplandor al pasar el mouse por elementos interactivos
 */
function addHoverGlowEffects() {
    // Aplicar a botones
    const buttons = document.querySelectorAll('a.cta-button, button, .cosmic-button');
    buttons.forEach(button => {
        button.addEventListener('mouseenter', () => {
            button.style.boxShadow = '0 0 25px var(--accent-color)';
        });
        
        button.addEventListener('mouseleave', () => {
            button.style.boxShadow = '0 0 15px var(--accent-color)';
        });
    });
    
    // Aplicar a tarjetas
    const cards = document.querySelectorAll('.feature-card, .cosmic-card');
    cards.forEach(card => {
        card.addEventListener('mouseenter', () => {
            card.style.transform = 'translateY(-10px)';
            card.style.boxShadow = '0 10px 25px rgba(0, 0, 0, 0.3), 0 0 20px rgba(0, 234, 255, 0.4)';
        });
        
        card.addEventListener('mouseleave', () => {
            card.style.transform = 'translateY(0)';
            card.style.boxShadow = '0 5px 15px rgba(0, 0, 0, 0.2), 0 0 15px rgba(0, 234, 255, 0.2)';
        });
    });
}

/**
 * Añade efectos de pulsación a ciertos elementos
 */
function addPulseEffects() {
    // Aplicar a títulos de sección
    const sectionTitles = document.querySelectorAll('.section-title, .cosmic-section-title');
    sectionTitles.forEach(title => {
        title.classList.add('pulse-text');
    });
    
    // Aplicar a iconos
    const icons = document.querySelectorAll('.feature-icon');
    icons.forEach(icon => {
        icon.classList.add('pulse-opacity');
    });
}

/**
 * Actualiza dinámicamente los fondos de estrellas para mayor movimiento
 */
function enhanceStarBackgrounds() {
    const starsContainers = document.querySelectorAll('.stars, .stars2, .stars3');
    
    starsContainers.forEach((container, index) => {
        // Aumentar el brillo periódicamente
        setInterval(() => {
            const currentOpacity = parseFloat(window.getComputedStyle(container).opacity);
            container.style.opacity = Math.min(1, currentOpacity + 0.1);
            
            setTimeout(() => {
                container.style.opacity = Math.max(0.5, currentOpacity);
            }, 500);
        }, 3000 + (index * 1000)); // Tiempos escalonados para cada capa
    });
}