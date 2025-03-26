/**
 * GENESIS - Sistema de Inversiones
 * Efecto de partículas avanzado
 */

class ParticleSystem {
    constructor(options = {}) {
        this.options = Object.assign({
            selector: '#particle-canvas',
            particleCount: 100,
            connectionDistance: 120,
            particleColors: [
                'rgba(156, 39, 176, 0.7)',  // Púrpura
                'rgba(103, 58, 183, 0.7)',  // Violeta
                'rgba(63, 81, 181, 0.7)',   // Índigo
                'rgba(225, 190, 231, 0.3)'  // Lavanda claro
            ],
            minSpeed: 0.1,
            maxSpeed: 0.5,
            minRadius: 1,
            maxRadius: 3,
            responsive: true,
            interactiveMode: true,
            interactionRadius: 150
        }, options);
        
        this.canvas = document.querySelector(this.options.selector);
        if (!this.canvas) {
            console.warn('Particle canvas not found');
            return;
        }
        
        this.ctx = this.canvas.getContext('2d');
        this.particles = [];
        this.animationFrame = null;
        this.mouse = {
            x: null,
            y: null,
            radius: this.options.interactionRadius
        };
        
        this.init();
    }
    
    init() {
        this.resizeCanvas();
        this.createParticles();
        this.setupEventListeners();
        this.animate();
    }
    
    resizeCanvas() {
        const devicePixelRatio = window.devicePixelRatio || 1;
        this.canvas.width = window.innerWidth * devicePixelRatio;
        this.canvas.height = window.innerHeight * devicePixelRatio;
        this.canvas.style.width = window.innerWidth + 'px';
        this.canvas.style.height = window.innerHeight + 'px';
        this.ctx.scale(devicePixelRatio, devicePixelRatio);
    }
    
    createParticles() {
        this.particles = [];
        const { particleCount, minRadius, maxRadius, minSpeed, maxSpeed, particleColors } = this.options;
        
        for (let i = 0; i < particleCount; i++) {
            const radius = Math.random() * (maxRadius - minRadius) + minRadius;
            const x = Math.random() * (this.canvas.width - radius * 2) + radius;
            const y = Math.random() * (this.canvas.height - radius * 2) + radius;
            const speed = Math.random() * (maxSpeed - minSpeed) + minSpeed;
            const angle = Math.random() * 360;
            const color = particleColors[Math.floor(Math.random() * particleColors.length)];
            const opacity = Math.random() * 0.5 + 0.1;
            
            // Verificar que la partícula no se superponga con otras existentes
            let overlapping = false;
            for (let j = 0; j < this.particles.length; j++) {
                const p = this.particles[j];
                const dx = x - p.x;
                const dy = y - p.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                
                if (dist < radius + p.radius) {
                    overlapping = true;
                    break;
                }
            }
            
            if (!overlapping) {
                this.particles.push({
                    x,
                    y,
                    radius,
                    color,
                    speed,
                    angle,
                    opacity,
                    lastX: x,
                    lastY: y
                });
            } else {
                i--; // Intentar de nuevo
            }
        }
    }
    
    setupEventListeners() {
        if (this.options.responsive) {
            window.addEventListener('resize', () => {
                this.resizeCanvas();
                this.createParticles();
            });
        }
        
        if (this.options.interactiveMode) {
            window.addEventListener('mousemove', (e) => {
                this.mouse.x = e.clientX;
                this.mouse.y = e.clientY;
            });
            
            window.addEventListener('mouseout', () => {
                this.mouse.x = null;
                this.mouse.y = null;
            });
            
            // Para dispositivos táctiles
            window.addEventListener('touchmove', (e) => {
                if (e.touches.length > 0) {
                    this.mouse.x = e.touches[0].clientX;
                    this.mouse.y = e.touches[0].clientY;
                }
            });
            
            window.addEventListener('touchend', () => {
                this.mouse.x = null;
                this.mouse.y = null;
            });
        }
    }
    
    animate() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        this.updateParticles();
        this.drawParticles();
        this.drawConnections();
        
        this.animationFrame = requestAnimationFrame(() => this.animate());
    }
    
    updateParticles() {
        const { interactiveMode } = this.options;
        
        this.particles.forEach((p) => {
            // Guardar posición anterior
            p.lastX = p.x;
            p.lastY = p.y;
            
            // Actualizar posición
            p.x += Math.cos(p.angle * Math.PI / 180) * p.speed;
            p.y += Math.sin(p.angle * Math.PI / 180) * p.speed;
            
            // Rebote en los bordes
            if (p.x - p.radius <= 0 || p.x + p.radius >= this.canvas.width) {
                p.angle = 180 - p.angle;
            }
            
            if (p.y - p.radius <= 0 || p.y + p.radius >= this.canvas.height) {
                p.angle = 360 - p.angle;
            }
            
            // Interacción con el ratón
            if (interactiveMode && this.mouse.x !== null && this.mouse.y !== null) {
                const dx = p.x - this.mouse.x;
                const dy = p.y - this.mouse.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                
                if (dist < this.mouse.radius) {
                    const force = (this.mouse.radius - dist) / this.mouse.radius;
                    const forceDirectionX = dx / dist;
                    const forceDirectionY = dy / dist;
                    
                    p.x += forceDirectionX * force * 2;
                    p.y += forceDirectionY * force * 2;
                }
            }
        });
    }
    
    drawParticles() {
        this.particles.forEach((p) => {
            this.ctx.beginPath();
            this.ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
            this.ctx.fillStyle = p.color;
            this.ctx.fill();
        });
    }
    
    drawConnections() {
        const { connectionDistance } = this.options;
        
        for (let i = 0; i < this.particles.length; i++) {
            for (let j = i + 1; j < this.particles.length; j++) {
                const p1 = this.particles[i];
                const p2 = this.particles[j];
                const dx = p1.x - p2.x;
                const dy = p1.y - p2.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                if (distance < connectionDistance) {
                    const opacity = 1 - (distance / connectionDistance);
                    this.ctx.beginPath();
                    this.ctx.strokeStyle = `rgba(156, 39, 176, ${opacity * 0.4})`;
                    this.ctx.lineWidth = p1.radius * 0.2;
                    this.ctx.moveTo(p1.x, p1.y);
                    this.ctx.lineTo(p2.x, p2.y);
                    this.ctx.stroke();
                }
            }
        }
    }
    
    destroy() {
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
        }
        
        window.removeEventListener('resize', this.resizeCanvas);
        window.removeEventListener('mousemove', this.handleMouseMove);
        window.removeEventListener('mouseout', this.handleMouseOut);
        window.removeEventListener('touchmove', this.handleTouchMove);
        window.removeEventListener('touchend', this.handleTouchEnd);
    }
}

// Inicializar sistema de partículas cuando el DOM esté listo
document.addEventListener('DOMContentLoaded', () => {
    new ParticleSystem({
        selector: '#particle-canvas',
        particleCount: 80,
        connectionDistance: 150,
        interactiveMode: true
    });
});