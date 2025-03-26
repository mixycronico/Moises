// main.js - Funcionalidad general del sitio

document.addEventListener('DOMContentLoaded', function() {
    // Animación adicional para elementos con efecto de brillo
    const glowElements = document.querySelectorAll('.logo, .logo-glow, .cta-button');
    
    glowElements.forEach(element => {
        element.addEventListener('mouseover', function() {
            this.style.textShadow = '0 0 20px var(--highlight-color), 0 0 30px var(--highlight-color)';
            if (this.classList.contains('cta-button')) {
                this.style.boxShadow = '0 0 20px var(--accent-color)';
            }
        });
        
        element.addEventListener('mouseout', function() {
            this.style.textShadow = '';
            if (this.classList.contains('cta-button')) {
                this.style.boxShadow = '';
            }
        });
    });
    
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
    
    // Activar elementos cuando entran en el viewport
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('active');
            }
        });
    }, { threshold: 0.1 });
    
    document.querySelectorAll('.feature-card, .stage').forEach(el => {
        observer.observe(el);
        el.classList.add('fade-in'); // Añadir clase para animación
    });
});

// Efecto de partículas para el fondo (versión simplificada)
function createParticles() {
    const particlesContainer = document.createElement('div');
    particlesContainer.className = 'particles-container';
    document.body.appendChild(particlesContainer);
    
    for (let i = 0; i < 50; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        
        // Posición aleatoria
        particle.style.left = Math.random() * 100 + 'vw';
        particle.style.top = Math.random() * 100 + 'vh';
        
        // Tamaño aleatorio
        const size = Math.random() * 5 + 2;
        particle.style.width = size + 'px';
        particle.style.height = size + 'px';
        
        // Animación aleatoria
        particle.style.animationDuration = (Math.random() * 10 + 10) + 's';
        particle.style.animationDelay = (Math.random() * 5) + 's';
        
        particlesContainer.appendChild(particle);
    }
}

// Crear partículas al cargar la página
window.addEventListener('load', createParticles);