/**
 * Custom Gauges for Genesis Investment Platform
 * 
 * Implementación de medidores personalizados con efectos holográficos
 * para la plataforma de inversiones Genesis.
 */

class GenesisGauge {
    /**
     * Crea un medidor personalizado con efectos holográficos.
     * @param {string} elementId - ID del elemento donde se renderizará el gauge
     * @param {Object} options - Opciones de configuración
     */
    constructor(elementId, options = {}) {
        this.elementId = elementId;
        this.element = document.getElementById(elementId);
        
        if (!this.element) {
            console.error(`Elemento con ID "${elementId}" no encontrado.`);
            return;
        }
        
        // Opciones predeterminadas
        this.options = {
            value: options.value || 0,
            min: options.min || 0,
            max: options.max || 100,
            size: options.size || 200,
            thickness: options.thickness || 20,
            startAngle: options.startAngle || -Math.PI / 2,
            endAngle: options.endAngle || (3 * Math.PI) / 2,
            label: options.label || '',
            valueLabel: options.valueLabel || '',
            primaryColor: options.primaryColor || '#9c27b0',
            secondaryColor: options.secondaryColor || '#673ab7',
            bgColor: options.bgColor || 'rgba(45, 35, 75, 0.2)',
            textColor: options.textColor || '#ffffff',
            animate: options.animate !== undefined ? options.animate : true,
            animationDuration: options.animationDuration || 1000,
            showValue: options.showValue !== undefined ? options.showValue : true,
            gradient: options.gradient !== undefined ? options.gradient : true,
            glowEffect: options.glowEffect !== undefined ? options.glowEffect : true,
            formatValue: options.formatValue || (value => `${value}%`),
            type: options.type || 'arc' // 'arc', 'bar', 'circle'
        };
        
        // Crear el canvas
        this.canvas = document.createElement('canvas');
        this.canvas.width = this.options.size;
        this.canvas.height = this.options.size;
        this.element.appendChild(this.canvas);
        
        this.ctx = this.canvas.getContext('2d');
        
        // Crear elemento para el valor
        if (this.options.showValue) {
            this.valueDisplay = document.createElement('div');
            this.valueDisplay.className = 'gauge-value';
            this.valueDisplay.style.position = 'absolute';
            this.valueDisplay.style.top = '50%';
            this.valueDisplay.style.left = '50%';
            this.valueDisplay.style.transform = 'translate(-50%, -50%)';
            this.valueDisplay.style.fontSize = `${this.options.size / 6}px`;
            this.valueDisplay.style.fontWeight = 'bold';
            this.valueDisplay.style.color = this.options.textColor;
            this.valueDisplay.style.textAlign = 'center';
            this.element.appendChild(this.valueDisplay);
        }
        
        // Crear elemento para la etiqueta
        if (this.options.label) {
            this.labelDisplay = document.createElement('div');
            this.labelDisplay.className = 'gauge-label';
            this.labelDisplay.textContent = this.options.label;
            this.labelDisplay.style.textAlign = 'center';
            this.labelDisplay.style.marginTop = '10px';
            this.labelDisplay.style.color = this.options.textColor;
            this.labelDisplay.style.fontSize = `${this.options.size / 12}px`;
            this.element.appendChild(this.labelDisplay);
        }
        
        // Posición relativa para el contenedor
        this.element.style.position = 'relative';
        
        // Estado de animación
        this.currentValue = 0;
        this.targetValue = this.options.value;
        this.animationStartTime = null;
        
        // Renderizar inicialmente
        this.render();
    }
    
    /**
     * Actualiza el valor del medidor.
     * @param {number} value - Nuevo valor
     * @param {boolean} animate - Si debe animarse la transición
     */
    setValue(value, animate = true) {
        if (value < this.options.min) value = this.options.min;
        if (value > this.options.max) value = this.options.max;
        
        this.targetValue = value;
        
        if (animate && this.options.animate) {
            this.currentValue = this.options.value;
            this.options.value = value;
            this.animationStartTime = performance.now();
            requestAnimationFrame(this.animateValue.bind(this));
        } else {
            this.options.value = value;
            this.currentValue = value;
            this.render();
        }
    }
    
    /**
     * Anima la transición entre valores.
     * @param {number} timestamp - Marca de tiempo actual
     */
    animateValue(timestamp) {
        if (!this.animationStartTime) this.animationStartTime = timestamp;
        
        const elapsed = timestamp - this.animationStartTime;
        const progress = Math.min(elapsed / this.options.animationDuration, 1);
        
        this.currentValue = this.currentValue + (this.targetValue - this.currentValue) * this.easeOutCubic(progress);
        
        this.render();
        
        if (progress < 1) {
            requestAnimationFrame(this.animateValue.bind(this));
        } else {
            this.currentValue = this.targetValue;
            this.render();
        }
    }
    
    /**
     * Función de suavizado para animaciones.
     * @param {number} x - Valor de progreso (0-1)
     * @returns {number} Valor suavizado
     */
    easeOutCubic(x) {
        return 1 - Math.pow(1 - x, 3);
    }
    
    /**
     * Renderiza el medidor en el canvas.
     */
    render() {
        const ctx = this.ctx;
        const options = this.options;
        const centerX = options.size / 2;
        const centerY = options.size / 2;
        const radius = (options.size / 2) - (options.thickness / 2);
        
        // Limpiar canvas
        ctx.clearRect(0, 0, options.size, options.size);
        
        // Calcular ángulos
        const valueRange = options.max - options.min;
        const angleRange = options.endAngle - options.startAngle;
        const valueProgress = (this.currentValue - options.min) / valueRange;
        const valueAngle = options.startAngle + (valueProgress * angleRange);
        
        // Dibujar según el tipo
        switch (options.type) {
            case 'arc':
                this.drawArcGauge(ctx, centerX, centerY, radius, valueAngle);
                break;
            case 'bar':
                this.drawBarGauge(ctx, valueProgress);
                break;
            case 'circle':
                this.drawCircleGauge(ctx, centerX, centerY, radius, valueProgress);
                break;
            default:
                this.drawArcGauge(ctx, centerX, centerY, radius, valueAngle);
        }
        
        // Actualizar valor mostrado
        if (options.showValue && this.valueDisplay) {
            this.valueDisplay.textContent = options.valueLabel || options.formatValue(Math.round(this.currentValue));
        }
    }
    
    /**
     * Dibuja un medidor tipo arco.
     */
    drawArcGauge(ctx, centerX, centerY, radius, valueAngle) {
        const options = this.options;
        
        // Fondo
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, options.startAngle, options.endAngle);
        ctx.lineWidth = options.thickness;
        ctx.strokeStyle = options.bgColor;
        ctx.stroke();
        
        // Valor (con gradiente si está habilitado)
        if (options.gradient) {
            const gradient = ctx.createLinearGradient(0, 0, options.size, options.size);
            gradient.addColorStop(0, options.primaryColor);
            gradient.addColorStop(1, options.secondaryColor);
            ctx.strokeStyle = gradient;
        } else {
            ctx.strokeStyle = options.primaryColor;
        }
        
        // Efecto de brillo
        if (options.glowEffect) {
            ctx.shadowColor = options.primaryColor;
            ctx.shadowBlur = 10;
        }
        
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, options.startAngle, valueAngle);
        ctx.lineWidth = options.thickness;
        ctx.stroke();
        
        // Restaurar sombra
        ctx.shadowColor = 'transparent';
        ctx.shadowBlur = 0;
        
        // Dibujar punto al final del arco de valor
        if (valueAngle !== options.startAngle) {
            const endX = centerX + radius * Math.cos(valueAngle);
            const endY = centerY + radius * Math.sin(valueAngle);
            
            ctx.beginPath();
            ctx.arc(endX, endY, options.thickness / 2, 0, Math.PI * 2);
            ctx.fillStyle = options.glowEffect ? options.secondaryColor : options.primaryColor;
            
            if (options.glowEffect) {
                ctx.shadowColor = options.secondaryColor;
                ctx.shadowBlur = 10;
            }
            
            ctx.fill();
            
            ctx.shadowColor = 'transparent';
            ctx.shadowBlur = 0;
        }
    }
    
    /**
     * Dibuja un medidor tipo barra.
     */
    drawBarGauge(ctx, valueProgress) {
        const options = this.options;
        const width = options.size;
        const height = options.thickness;
        const y = (options.size - height) / 2;
        
        // Fondo
        ctx.fillStyle = options.bgColor;
        ctx.fillRect(0, y, width, height);
        
        // Valor
        const valueWidth = width * valueProgress;
        
        if (options.gradient) {
            const gradient = ctx.createLinearGradient(0, 0, width, 0);
            gradient.addColorStop(0, options.primaryColor);
            gradient.addColorStop(1, options.secondaryColor);
            ctx.fillStyle = gradient;
        } else {
            ctx.fillStyle = options.primaryColor;
        }
        
        if (options.glowEffect) {
            ctx.shadowColor = options.primaryColor;
            ctx.shadowBlur = 10;
        }
        
        ctx.fillRect(0, y, valueWidth, height);
        
        ctx.shadowColor = 'transparent';
        ctx.shadowBlur = 0;
    }
    
    /**
     * Dibuja un medidor tipo círculo (dona).
     */
    drawCircleGauge(ctx, centerX, centerY, radius, valueProgress) {
        const options = this.options;
        
        // Fondo
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
        ctx.lineWidth = options.thickness;
        ctx.strokeStyle = options.bgColor;
        ctx.stroke();
        
        // Valor
        const endAngle = Math.PI * 2 * valueProgress;
        
        if (options.gradient) {
            const gradient = ctx.createLinearGradient(
                centerX - radius, 
                centerY - radius, 
                centerX + radius, 
                centerY + radius
            );
            gradient.addColorStop(0, options.primaryColor);
            gradient.addColorStop(1, options.secondaryColor);
            ctx.strokeStyle = gradient;
        } else {
            ctx.strokeStyle = options.primaryColor;
        }
        
        if (options.glowEffect) {
            ctx.shadowColor = options.primaryColor;
            ctx.shadowBlur = 10;
        }
        
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, 0, endAngle);
        ctx.lineWidth = options.thickness;
        ctx.stroke();
        
        ctx.shadowColor = 'transparent';
        ctx.shadowBlur = 0;
    }
}

// Variantes predefinidas

/**
 * Crea un medidor de ROI.
 * @param {string} elementId - ID del elemento donde se renderizará
 * @param {number} value - Valor inicial (0-100)
 * @returns {GenesisGauge} Instancia del medidor
 */
function createROIGauge(elementId, value = 0) {
    return new GenesisGauge(elementId, {
        value: value,
        label: 'ROI',
        primaryColor: '#9c27b0',
        secondaryColor: '#673ab7',
        thickness: 15,
        formatValue: value => `${value}%`
    });
}

/**
 * Crea un medidor de riesgo.
 * @param {string} elementId - ID del elemento donde se renderizará
 * @param {number} value - Valor inicial (0-100)
 * @returns {GenesisGauge} Instancia del medidor
 */
function createRiskGauge(elementId, value = 0) {
    return new GenesisGauge(elementId, {
        value: value,
        label: 'Riesgo',
        primaryColor: '#f44336',
        secondaryColor: '#ff9800',
        thickness: 15,
        formatValue: value => `${value}%`
    });
}

/**
 * Crea un medidor de volatilidad.
 * @param {string} elementId - ID del elemento donde se renderizará
 * @param {number} value - Valor inicial (0-100)
 * @returns {GenesisGauge} Instancia del medidor
 */
function createVolatilityGauge(elementId, value = 0) {
    return new GenesisGauge(elementId, {
        value: value,
        label: 'Volatilidad',
        primaryColor: '#2196f3',
        secondaryColor: '#03a9f4',
        thickness: 15,
        formatValue: value => `${value}%`
    });
}

/**
 * Crea un medidor de crecimiento.
 * @param {string} elementId - ID del elemento donde se renderizará
 * @param {number} value - Valor inicial (0-100)
 * @returns {GenesisGauge} Instancia del medidor
 */
function createGrowthGauge(elementId, value = 0) {
    return new GenesisGauge(elementId, {
        value: value,
        label: 'Crecimiento',
        primaryColor: '#4caf50',
        secondaryColor: '#8bc34a',
        thickness: 15,
        formatValue: value => `${value}%`
    });
}

/**
 * Crea un medidor personalizado.
 * @param {string} elementId - ID del elemento donde se renderizará
 * @param {Object} options - Opciones de configuración
 * @returns {GenesisGauge} Instancia del medidor
 */
function createCustomGauge(elementId, options) {
    return new GenesisGauge(elementId, options);
}