// main.js - Funcionalidad general del sitio con efectos mejorados

document.addEventListener('DOMContentLoaded', function() {
    // Aplicar efectos de brillo a elementos de navegación y logo
    applyGlowEffects();
    
    // Mejorar estrellas con brillo adicional
    enhanceStarsEffect();
    
    // Animación de parallax para el fondo de estrellas
    document.addEventListener('mousemove', function(e) {
        const stars = document.querySelectorAll('.stars, .stars2, .stars3');
        const x = e.clientX / window.innerWidth;
        const y = e.clientY / window.innerHeight;
        
        stars.forEach((layer, index) => {
            const speed = (index + 1) * 2;
            const offsetX = (x - 0.5) * speed;
            const offsetY = (y - 0.5) * speed;
            
            layer.style.transform = `translate(${offsetX}px, ${offsetY}px)`;
        });
    });
    
    // Activar elementos cuando entran en el viewport con efecto mejorado
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('active');
                
                // Añadir brillo adicional a elementos activos
                if (entry.target.classList.contains('feature-card') || 
                    entry.target.classList.contains('stage')) {
                    entry.target.style.boxShadow = '0 0 25px rgba(0, 234, 255, 0.4)';
                }
            }
        });
    }, { threshold: 0.1 });
    
    // Observar elementos para animación de entrada
    document.querySelectorAll('.feature-card, .stage, .dashboard-card, .stat-item').forEach(el => {
        observer.observe(el);
        el.classList.add('fade-in'); // Añadir clase para animación
    });
});

// Función para aplicar efectos de brillo a elementos interactivos
function applyGlowEffects() {
    // Elementos con efecto de brillo
    const glowElements = document.querySelectorAll('.logo, .logo-glow, .cta-button, .nav-links a');
    
    // Aplicar efectos base permanentes para mayor luminosidad
    glowElements.forEach(element => {
        // Aplicar un brillo sutil por defecto
        if (element.classList.contains('logo') || element.classList.contains('logo-glow')) {
            element.style.textShadow = '0 0 10px var(--highlight-color)';
        }
        
        if (element.classList.contains('cta-button')) {
            element.style.boxShadow = '0 0 10px var(--accent-color)';
        }
        
        // Añadir efecto hover mejorado
        element.addEventListener('mouseover', function() {
            if (element.classList.contains('logo') || element.classList.contains('logo-glow')) {
                this.style.textShadow = '0 0 20px var(--highlight-color), 0 0 30px var(--highlight-color), 0 0 40px var(--highlight-color)';
            } else if (element.classList.contains('cta-button')) {
                this.style.boxShadow = '0 0 25px var(--accent-color)';
                this.style.transform = 'scale(1.05)';
            } else {
                this.style.textShadow = '0 0 10px var(--highlight-color)';
                this.style.color = 'var(--highlight-color)';
            }
        });
        
        element.addEventListener('mouseout', function() {
            if (element.classList.contains('logo') || element.classList.contains('logo-glow')) {
                this.style.textShadow = '0 0 10px var(--highlight-color)';
            } else if (element.classList.contains('cta-button')) {
                this.style.boxShadow = '0 0 10px var(--accent-color)';
                this.style.transform = 'scale(1)';
            } else {
                this.style.textShadow = '';
                this.style.color = '';
            }
        });
    });
    
    // Aplicar efecto de pulso al logo principal
    const logoGlow = document.querySelector('.logo-glow');
    if (logoGlow) {
        setInterval(() => {
            logoGlow.style.textShadow = '0 0 30px var(--highlight-color), 0 0 50px var(--highlight-color)';
            setTimeout(() => {
                logoGlow.style.textShadow = '0 0 10px var(--highlight-color)';
            }, 500);
        }, 3000);
    }
}

// Mejorar el efecto de estrellas para mayor luminosidad
function enhanceStarsEffect() {
    const starsContainer = document.querySelector('.stars-container');
    if (starsContainer) {
        // Aumentar la opacidad para mayor visibilidad
        starsContainer.style.opacity = '0.9';
        
        // Añadir más brillo a las estrellas
        const stars = document.querySelectorAll('.stars, .stars2, .stars3');
        stars.forEach(starLayer => {
            starLayer.style.filter = 'drop-shadow(0 0 3px rgba(0, 234, 255, 0.9))';
        });
    }
}

// Efecto de partículas para el fondo (versión mejorada)
function createParticles() {
    const particlesContainer = document.createElement('div');
    particlesContainer.className = 'particles-container';
    document.body.appendChild(particlesContainer);
    
    // Aumentar número de partículas para un efecto más dramático
    for (let i = 0; i < 75; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        
        // Posición aleatoria
        particle.style.left = Math.random() * 100 + 'vw';
        particle.style.top = Math.random() * 100 + 'vh';
        
        // Tamaño aleatorio (ligeramente más grandes)
        const size = Math.random() * 6 + 2;
        particle.style.width = size + 'px';
        particle.style.height = size + 'px';
        
        // Color aleatorio (cian o blanco para combinar con tema)
        const isHighlight = Math.random() > 0.7;
        particle.style.backgroundColor = isHighlight ? 'var(--highlight-color)' : '#ffffff';
        
        // Brillo adicional para partículas
        particle.style.boxShadow = isHighlight ? '0 0 8px var(--highlight-color)' : '0 0 5px #ffffff';
        
        // Animación aleatoria
        particle.style.animationDuration = (Math.random() * 10 + 8) + 's';
        particle.style.animationDelay = (Math.random() * 5) + 's';
        
        particlesContainer.appendChild(particle);
    }
}

// Añadir efectos de transición entre páginas
function addPageTransitions() {
    document.querySelectorAll('a').forEach(link => {
        // Solo aplicar a enlaces internos
        if (link.hostname === window.location.hostname) {
            link.addEventListener('click', function(e) {
                // No aplicar en enlaces especiales
                if (this.classList.contains('no-transition')) return;
                
                const href = this.getAttribute('href');
                const isInternalLink = href && href.startsWith('/') || href.startsWith(window.location.origin);
                
                if (isInternalLink) {
                    e.preventDefault();
                    
                    // Crear efecto de flash
                    const overlay = document.createElement('div');
                    overlay.style.position = 'fixed';
                    overlay.style.top = 0;
                    overlay.style.left = 0;
                    overlay.style.width = '100%';
                    overlay.style.height = '100%';
                    overlay.style.backgroundColor = 'rgba(0, 234, 255, 0.2)';
                    overlay.style.zIndex = 9999;
                    overlay.style.opacity = 0;
                    overlay.style.transition = 'opacity 0.3s ease';
                    document.body.appendChild(overlay);
                    
                    // Animar efecto
                    setTimeout(() => {
                        overlay.style.opacity = 0.7;
                        setTimeout(() => {
                            window.location.href = href;
                        }, 300);
                    }, 10);
                }
            });
        }
    });
}

// Crear partículas al cargar la página
window.addEventListener('load', function() {
    createParticles();
    addPageTransitions();
});