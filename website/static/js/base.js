// Funcionalidad para el menú móvil
document.addEventListener('DOMContentLoaded', function() {
    const mobileMenuToggle = document.querySelector('.mobile-menu-toggle');
    const mobileMenuClose = document.querySelector('.mobile-menu-close');
    const mobileMenu = document.querySelector('.mobile-menu');
    
    if (mobileMenuToggle && mobileMenuClose && mobileMenu) {
        mobileMenuToggle.addEventListener('click', function() {
            mobileMenu.classList.add('active');
            document.body.style.overflow = 'hidden'; // Prevenir scroll
        });
        
        mobileMenuClose.addEventListener('click', function() {
            mobileMenu.classList.remove('active');
            document.body.style.overflow = ''; // Restaurar scroll
        });
    }
    
    // Cerrar alertas
    const alertCloseButtons = document.querySelectorAll('.alert-close');
    alertCloseButtons.forEach(button => {
        button.addEventListener('click', function() {
            const alert = this.closest('.alert');
            if (alert) {
                alert.style.opacity = '0';
                setTimeout(() => {
                    alert.style.display = 'none';
                }, 300);
            }
        });
    });
    
    // Añadir año actual al copyright
    const currentYearElements = document.querySelectorAll('.current-year');
    const currentYear = new Date().getFullYear();
    currentYearElements.forEach(el => {
        el.textContent = currentYear;
    });
    
    // Efecto de revelación al scroll
    const revealElements = document.querySelectorAll('.reveal');
    
    function reveal() {
        const windowHeight = window.innerHeight;
        
        revealElements.forEach(el => {
            const elementTop = el.getBoundingClientRect().top;
            const elementVisible = 150;
            
            if (elementTop < windowHeight - elementVisible) {
                el.classList.add('active');
            }
        });
    }
    
    window.addEventListener('scroll', reveal);
    reveal(); // Ejecutar una vez al cargar
    
    // Efecto de hover parallax en tarjetas
    const parallaxCards = document.querySelectorAll('.parallax-card');
    
    parallaxCards.forEach(card => {
        card.addEventListener('mousemove', function(e) {
            const rect = this.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            
            const offsetX = (x - centerX) / centerX * 10;
            const offsetY = (y - centerY) / centerY * 10;
            
            this.style.transform = `perspective(1000px) rotateY(${offsetX}deg) rotateX(${-offsetY}deg)`;
            
            // Efecto de brillo
            const glare = this.querySelector('.card-glare');
            if (glare) {
                const intensity = (x / rect.width * 100);
                glare.style.background = `linear-gradient(${intensity}deg, rgba(255,255,255,0.25), transparent)`;
            }
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'perspective(1000px) rotateY(0deg) rotateX(0deg)';
            
            const glare = this.querySelector('.card-glare');
            if (glare) {
                glare.style.background = 'transparent';
            }
        });
    });
});