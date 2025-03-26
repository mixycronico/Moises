/**
 * Genesis - Sistema de Inversiones
 * Script principal para funcionalidades del sitio
 */

document.addEventListener('DOMContentLoaded', function() {
    // Menú móvil
    const menuToggle = document.querySelector('.menu-toggle');
    const navList = document.querySelector('.nav-list');
    
    if (menuToggle && navList) {
        menuToggle.addEventListener('click', function() {
            navList.classList.toggle('active');
            menuToggle.classList.toggle('active');
        });
    }
    
    // Efecto header al scroll
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
    
    // Manejador de alertas para cerrarlas
    const alerts = document.querySelectorAll('.alert');
    
    alerts.forEach(alert => {
        const closeBtn = document.createElement('span');
        closeBtn.innerHTML = '&times;';
        closeBtn.className = 'alert-close';
        closeBtn.style.float = 'right';
        closeBtn.style.cursor = 'pointer';
        closeBtn.style.fontWeight = 'bold';
        
        closeBtn.addEventListener('click', function() {
            alert.style.display = 'none';
        });
        
        alert.appendChild(closeBtn);
        
        // Auto-ocultar alertas después de 5 segundos
        setTimeout(() => {
            alert.style.opacity = '0';
            alert.style.transition = 'opacity 0.5s ease';
            
            setTimeout(() => {
                alert.style.display = 'none';
            }, 500);
        }, 5000);
    });
    
    // Ajustar tamaño del texto en logos para dispositivos pequeños
    function adjustLogoTextSize() {
        const logoText = document.querySelectorAll('.logo-text, .footer-logo-text');
        if (window.innerWidth < 576) {
            logoText.forEach(text => {
                text.style.fontSize = '1rem';
            });
        } else {
            logoText.forEach(text => {
                text.style.fontSize = '';
            });
        }
    }
    
    window.addEventListener('resize', adjustLogoTextSize);
    adjustLogoTextSize();
    
    // Efecto de aparición en scroll (solo para elementos con clase 'fade-in')
    const fadeElements = document.querySelectorAll('.fade-in');
    
    function checkFade() {
        const triggerBottom = window.innerHeight * 0.8;
        
        fadeElements.forEach(element => {
            const elementTop = element.getBoundingClientRect().top;
            
            if (elementTop < triggerBottom) {
                element.classList.add('visible');
            }
        });
    }
    
    window.addEventListener('scroll', checkFade);
    checkFade(); // Verificar elementos visibles inicialmente
    
    // Validación de formularios genérica
    const forms = document.querySelectorAll('form[data-validate="true"]');
    
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            let isValid = true;
            const requiredInputs = form.querySelectorAll('[required]');
            
            requiredInputs.forEach(input => {
                // Eliminar mensajes de error previos
                const existingError = input.parentNode.querySelector('.error-message');
                if (existingError) {
                    existingError.remove();
                }
                
                // Validar campo vacío
                if (!input.value.trim()) {
                    isValid = false;
                    const errorMsg = document.createElement('div');
                    errorMsg.className = 'error-message';
                    errorMsg.style.color = '#f44336';
                    errorMsg.style.fontSize = '0.8rem';
                    errorMsg.style.marginTop = '5px';
                    errorMsg.textContent = 'Este campo es obligatorio';
                    
                    input.parentNode.appendChild(errorMsg);
                    input.style.borderColor = '#f44336';
                }
                
                // Validación de email
                if (input.type === 'email' && input.value.trim()) {
                    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
                    if (!emailRegex.test(input.value)) {
                        isValid = false;
                        const errorMsg = document.createElement('div');
                        errorMsg.className = 'error-message';
                        errorMsg.style.color = '#f44336';
                        errorMsg.style.fontSize = '0.8rem';
                        errorMsg.style.marginTop = '5px';
                        errorMsg.textContent = 'Email inválido';
                        
                        input.parentNode.appendChild(errorMsg);
                        input.style.borderColor = '#f44336';
                    }
                }
                
                // Validación de contraseña
                if (input.type === 'password' && input.dataset.minLength && input.value.trim()) {
                    const minLength = parseInt(input.dataset.minLength);
                    if (input.value.length < minLength) {
                        isValid = false;
                        const errorMsg = document.createElement('div');
                        errorMsg.className = 'error-message';
                        errorMsg.style.color = '#f44336';
                        errorMsg.style.fontSize = '0.8rem';
                        errorMsg.style.marginTop = '5px';
                        errorMsg.textContent = `La contraseña debe tener al menos ${minLength} caracteres`;
                        
                        input.parentNode.appendChild(errorMsg);
                        input.style.borderColor = '#f44336';
                    }
                }
            });
            
            if (!isValid) {
                e.preventDefault();
            }
        });
        
        // Resetear estilos de error al escribir
        const inputs = form.querySelectorAll('input, textarea, select');
        inputs.forEach(input => {
            input.addEventListener('input', function() {
                const existingError = input.parentNode.querySelector('.error-message');
                if (existingError) {
                    existingError.remove();
                }
                input.style.borderColor = '';
            });
        });
    });
});