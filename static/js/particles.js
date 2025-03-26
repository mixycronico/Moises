/**
 * Genesis - Sistema de Inversiones
 * Script para crear efecto de partículas cósmicas en el fondo
 */

document.addEventListener('DOMContentLoaded', function() {
    // Crear el canvas para las partículas
    const canvas = document.createElement('canvas');
    canvas.className = 'particle-canvas';
    document.body.appendChild(canvas);
    
    const ctx = canvas.getContext('2d');
    
    // Configuración del lienzo
    function resizeCanvas() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }
    
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();
    
    // Configuración de partículas
    const particleCount = 100;
    const particles = [];
    
    // Colores para las partículas (temática cósmica)
    const colors = [
        '#9c27b0', // Púrpura
        '#673ab7', // Violeta oscuro
        '#3f51b5', // Azul índigo
        '#2196f3', // Azul celeste
        '#00bcd4', // Cian
        '#e1bee7', // Lavanda claro
        '#b39ddb', // Lavanda medio
        '#7e57c2'  // Violeta medio
    ];
    
    // Crear partículas iniciales
    function initParticles() {
        for (let i = 0; i < particleCount; i++) {
            particles.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                size: Math.random() * 3 + 0.5,
                speed: Math.random() * 0.5 + 0.1,
                color: colors[Math.floor(Math.random() * colors.length)],
                direction: Math.random() * Math.PI * 2,
                opacity: Math.random() * 0.5 + 0.1,
                pulse: Math.random() * 0.02 + 0.01,
                pulseDirection: Math.random() > 0.5 ? 1 : -1
            });
        }
    }
    
    // Actualizar posición de las partículas
    function updateParticles() {
        particles.forEach(p => {
            // Movimiento
            p.x += Math.cos(p.direction) * p.speed;
            p.y += Math.sin(p.direction) * p.speed;
            
            // Cambio ligero de dirección (movimiento orgánico)
            p.direction += (Math.random() - 0.5) * 0.05;
            
            // Efecto pulso en opacidad
            p.opacity += p.pulse * p.pulseDirection;
            if (p.opacity > 0.8 || p.opacity < 0.1) {
                p.pulseDirection *= -1;
            }
            
            // Reposicionar si sale del canvas
            if (p.x < 0) p.x = canvas.width;
            if (p.x > canvas.width) p.x = 0;
            if (p.y < 0) p.y = canvas.height;
            if (p.y > canvas.height) p.y = 0;
        });
    }
    
    // Dibujar partículas
    function drawParticles() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        particles.forEach(p => {
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            ctx.fillStyle = p.color;
            ctx.globalAlpha = p.opacity;
            ctx.fill();
        });
        
        // Dibujar líneas de conexión entre partículas cercanas
        ctx.globalAlpha = 0.1;
        ctx.strokeStyle = '#9c27b0';
        ctx.lineWidth = 0.5;
        
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                // Conectar partículas cercanas
                if (distance < 150) {
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.stroke();
                }
            }
        }
        
        ctx.globalAlpha = 1;
    }
    
    // Animación principal
    function animate() {
        updateParticles();
        drawParticles();
        requestAnimationFrame(animate);
    }
    
    // Iniciar todo
    initParticles();
    animate();
});