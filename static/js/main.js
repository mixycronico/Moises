/**
 * GENESIS - Sistema de Inversiones
 * Script principal
 */

document.addEventListener('DOMContentLoaded', function() {
    // Inicializar efectos de partículas
    initParticles();
    
    // Inicializar la navegación móvil
    initMobileNav();
    
    // Inicializar efectos de scroll
    initScrollEffects();
    
    // Inicializar GSAP ScrollTrigger si está disponible
    if (window.gsap && window.ScrollTrigger) {
        gsap.registerPlugin(ScrollTrigger);
    }
});

/**
 * Inicializar animación de partículas
 */
function initParticles() {
    const canvas = document.getElementById('particle-canvas');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    let particles = [];
    const particleCount = 100;
    
    // Ajustar el tamaño del canvas al tamaño de la ventana
    function resizeCanvas() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }
    
    // Crear partículas iniciales
    function createParticles() {
        particles = [];
        for (let i = 0; i < particleCount; i++) {
            particles.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                radius: Math.random() * 2 + 1,
                color: getRandomColor(),
                speed: Math.random() * 0.5 + 0.1,
                angle: Math.random() * 360,
                opacity: Math.random() * 0.5 + 0.1
            });
        }
    }
    
    // Obtener un color aleatorio de la paleta
    function getRandomColor() {
        const colors = [
            'rgba(156, 39, 176, 0.7)',  // Púrpura
            'rgba(103, 58, 183, 0.7)',  // Violeta
            'rgba(63, 81, 181, 0.7)',   // Índigo
            'rgba(225, 190, 231, 0.3)'  // Lavanda claro
        ];
        return colors[Math.floor(Math.random() * colors.length)];
    }
    
    // Dibujar partículas
    function drawParticles() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        particles.forEach(particle => {
            ctx.beginPath();
            ctx.arc(particle.x, particle.y, particle.radius, 0, Math.PI * 2);
            ctx.fillStyle = particle.color;
            ctx.fill();
            
            // Actualizar posición
            particle.x += Math.cos(particle.angle * Math.PI / 180) * particle.speed;
            particle.y += Math.sin(particle.angle * Math.PI / 180) * particle.speed;
            
            // Rebote en los bordes
            if (particle.x > canvas.width) particle.x = 0;
            if (particle.x < 0) particle.x = canvas.width;
            if (particle.y > canvas.height) particle.y = 0;
            if (particle.y < 0) particle.y = canvas.height;
        });
        
        // Dibujar líneas entre partículas cercanas
        drawConnections();
        
        requestAnimationFrame(drawParticles);
    }
    
    // Dibujar conexiones entre partículas cercanas
    function drawConnections() {
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                if (distance < 100) {
                    ctx.beginPath();
                    ctx.strokeStyle = `rgba(156, 39, 176, ${0.2 * (1 - distance / 100)})`;
                    ctx.lineWidth = 0.5;
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.stroke();
                }
            }
        }
    }
    
    // Inicializar
    resizeCanvas();
    createParticles();
    drawParticles();
    
    // Ajustar canvas cuando cambia el tamaño de la ventana
    window.addEventListener('resize', function() {
        resizeCanvas();
        createParticles();
    });
}

/**
 * Inicializar navegación móvil
 */
function initMobileNav() {
    const menuToggle = document.querySelector('.menu-toggle');
    const navList = document.querySelector('.nav-list');
    
    if (menuToggle && navList) {
        menuToggle.addEventListener('click', function() {
            navList.classList.toggle('active');
            const isOpen = navList.classList.contains('active');
            
            // Animar las barras del menú
            const bars = menuToggle.querySelectorAll('.bar');
            if (isOpen) {
                bars[0].style.transform = 'rotate(-45deg) translate(-5px, 6px)';
                bars[1].style.opacity = '0';
                bars[2].style.transform = 'rotate(45deg) translate(-5px, -6px)';
            } else {
                bars[0].style.transform = 'none';
                bars[1].style.opacity = '1';
                bars[2].style.transform = 'none';
            }
        });
    }
}

/**
 * Inicializar efectos de scroll
 */
function initScrollEffects() {
    const header = document.querySelector('.site-header');
    
    if (header) {
        window.addEventListener('scroll', function() {
            if (window.scrollY > 50) {
                header.classList.add('scrolled');
            } else {
                header.classList.remove('scrolled');
            }
        });
    }
    
    // Detectar elementos que aparecen en el viewport
    const fadeElements = document.querySelectorAll('.fade-in');
    
    if (fadeElements.length > 0) {
        const fadeInObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('visible');
                }
            });
        }, {
            threshold: 0.1
        });
        
        fadeElements.forEach(element => {
            fadeInObserver.observe(element);
        });
    }
}