/**
 * Sistema de medidores holográficos personalizados para Genesis Trading System
 * Implementación de medidores/gauges visuales con efectos holográficos y cósmicos
 */

// Clase para crear medidores holográficos
class HolographicGauge {
    constructor(elementId, options = {}) {
        this.element = document.getElementById(elementId);
        if (!this.element) {
            console.error(`Error: No se encontró elemento con ID ${elementId}`);
            return;
        }
        
        // Opciones por defecto
        this.options = {
            min: options.min || 0,
            max: options.max || 100,
            value: options.value || 0,
            label: options.label || 'Medidor',
            decimals: options.decimals || 1,
            colorStart: options.colorStart || '#9c27b0',
            colorEnd: options.colorEnd || '#3f51b5',
            glowColor: options.glowColor || 'rgba(156, 39, 176, 0.7)',
            size: options.size || 200,
            thickness: options.thickness || 20,
            animation: options.animation || true,
            animationDuration: options.animationDuration || 1500,
            subLabel: options.subLabel || '',
            symbol: options.symbol || '%'
        };
        
        // Inicialización
        this.initialized = false;
        this.svg = null;
        this.value = 0;
        this.init();
    }
    
    init() {
        // Establecer contenedores
        this.element.classList.add('holographic-gauge');
        this.element.style.width = `${this.options.size}px`;
        this.element.style.height = `${this.options.size}px`;
        
        // Crear elementos SVG
        this.createSVG();
        
        // Inicializar valor
        this.setValue(this.options.value);
        
        // Marcar como inicializado
        this.initialized = true;
    }
    
    createSVG() {
        // Limpiar elemento
        this.element.innerHTML = '';
        
        // Crear contenedor SVG
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('width', '100%');
        svg.setAttribute('height', '100%');
        svg.setAttribute('viewBox', '0 0 100 100');
        
        // Crear degradado
        const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
        
        // Degradado lineal
        const linearGradient = document.createElementNS('http://www.w3.org/2000/svg', 'linearGradient');
        linearGradient.setAttribute('id', `${this.element.id}-gradient`);
        linearGradient.setAttribute('x1', '0%');
        linearGradient.setAttribute('y1', '0%');
        linearGradient.setAttribute('x2', '100%');
        linearGradient.setAttribute('y2', '100%');
        
        const stop1 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
        stop1.setAttribute('offset', '0%');
        stop1.setAttribute('stop-color', this.options.colorStart);
        
        const stop2 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
        stop2.setAttribute('offset', '100%');
        stop2.setAttribute('stop-color', this.options.colorEnd);
        
        linearGradient.appendChild(stop1);
        linearGradient.appendChild(stop2);
        
        // Efecto de resplandor (glow)
        const filter = document.createElementNS('http://www.w3.org/2000/svg', 'filter');
        filter.setAttribute('id', `${this.element.id}-glow`);
        
        const feGaussianBlur = document.createElementNS('http://www.w3.org/2000/svg', 'feGaussianBlur');
        feGaussianBlur.setAttribute('stdDeviation', '2.5');
        feGaussianBlur.setAttribute('result', 'blur');
        
        const feComposite = document.createElementNS('http://www.w3.org/2000/svg', 'feComposite');
        feComposite.setAttribute('in', 'SourceGraphic');
        feComposite.setAttribute('in2', 'blur');
        feComposite.setAttribute('operator', 'atop');
        
        filter.appendChild(feGaussianBlur);
        filter.appendChild(feComposite);
        
        defs.appendChild(linearGradient);
        defs.appendChild(filter);
        svg.appendChild(defs);
        
        // Fondo del medidor (círculo gris)
        const radius = 50 - this.options.thickness / 2;
        const circumference = 2 * Math.PI * radius;
        
        // Círculo de fondo
        const circleBackground = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circleBackground.setAttribute('cx', '50');
        circleBackground.setAttribute('cy', '50');
        circleBackground.setAttribute('r', radius);
        circleBackground.setAttribute('fill', 'none');
        circleBackground.setAttribute('stroke', 'rgba(255, 255, 255, 0.1)');
        circleBackground.setAttribute('stroke-width', this.options.thickness);
        svg.appendChild(circleBackground);
        
        // Círculo de progreso
        const circleProgress = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circleProgress.setAttribute('cx', '50');
        circleProgress.setAttribute('cy', '50');
        circleProgress.setAttribute('r', radius);
        circleProgress.setAttribute('fill', 'none');
        circleProgress.setAttribute('stroke', `url(#${this.element.id}-gradient)`);
        circleProgress.setAttribute('stroke-width', this.options.thickness);
        circleProgress.setAttribute('stroke-linecap', 'round');
        circleProgress.setAttribute('transform', 'rotate(-90 50 50)');
        circleProgress.setAttribute('stroke-dasharray', circumference);
        circleProgress.setAttribute('stroke-dashoffset', circumference);
        circleProgress.setAttribute('filter', `url(#${this.element.id}-glow)`);
        svg.appendChild(circleProgress);
        
        // Grupo para textos
        const textGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        
        // Etiqueta principal
        const labelText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        labelText.setAttribute('x', '50');
        labelText.setAttribute('y', '40');
        labelText.setAttribute('text-anchor', 'middle');
        labelText.setAttribute('font-size', '10');
        labelText.setAttribute('fill', 'rgba(225, 190, 231, 0.9)');
        labelText.textContent = this.options.label;
        textGroup.appendChild(labelText);
        
        // Valor
        const valueText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        valueText.setAttribute('x', '50');
        valueText.setAttribute('y', '60');
        valueText.setAttribute('text-anchor', 'middle');
        valueText.setAttribute('font-size', '18');
        valueText.setAttribute('font-weight', 'bold');
        valueText.setAttribute('fill', 'white');
        valueText.textContent = '0';
        textGroup.appendChild(valueText);
        
        // Sub-etiqueta (si existe)
        if (this.options.subLabel) {
            const subLabelText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            subLabelText.setAttribute('x', '50');
            subLabelText.setAttribute('y', '75');
            subLabelText.setAttribute('text-anchor', 'middle');
            subLabelText.setAttribute('font-size', '8');
            subLabelText.setAttribute('fill', 'rgba(179, 157, 219, 0.8)');
            subLabelText.textContent = this.options.subLabel;
            textGroup.appendChild(subLabelText);
        }
        
        svg.appendChild(textGroup);
        
        // Guardamos referencias para manipulación posterior
        this.svg = svg;
        this.circleProgress = circleProgress;
        this.valueText = valueText;
        this.circumference = circumference;
        
        // Añadir al DOM
        this.element.appendChild(svg);
    }
    
    setValue(value) {
        // Asegurar que el valor esté dentro de los límites
        const actualValue = Math.min(Math.max(value, this.options.min), this.options.max);
        const previousValue = this.value;
        this.value = actualValue;
        
        // Si no está inicializado, salir
        if (!this.initialized) return;
        
        // Calcular el offset para el trazo del círculo
        const progress = (actualValue - this.options.min) / (this.options.max - this.options.min);
        const offset = this.circumference - progress * this.circumference;
        
        // Animar el cambio
        if (this.options.animation && previousValue !== actualValue) {
            // Animación del círculo
            const startOffset = this.circleProgress.getAttribute('stroke-dashoffset');
            this.animateValue(parseFloat(startOffset), offset, this.options.animationDuration, (value) => {
                this.circleProgress.setAttribute('stroke-dashoffset', value);
            });
            
            // Animación del texto
            this.animateValue(previousValue, actualValue, this.options.animationDuration, (value) => {
                this.valueText.textContent = value.toFixed(this.options.decimals) + this.options.symbol;
            });
        } else {
            // Sin animación
            this.circleProgress.setAttribute('stroke-dashoffset', offset);
            this.valueText.textContent = actualValue.toFixed(this.options.decimals) + this.options.symbol;
        }
    }
    
    animateValue(start, end, duration, callback) {
        const startTime = performance.now();
        const change = end - start;
        
        const animate = (time) => {
            let elapsedTime = time - startTime;
            if (elapsedTime > duration) elapsedTime = duration;
            
            const progress = elapsedTime / duration;
            const easedProgress = this.easeOutQuad(progress);
            const currentValue = start + change * easedProgress;
            
            callback(currentValue);
            
            if (elapsedTime < duration) {
                requestAnimationFrame(animate);
            }
        };
        
        requestAnimationFrame(animate);
    }
    
    easeOutQuad(t) {
        return t * (2 - t);
    }
    
    update(value) {
        this.setValue(value);
    }
}

// Clase para crear conjunto de medidores
class GaugeGroup {
    constructor(containerId, gaugeData = []) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error(`Error: No se encontró contenedor con ID ${containerId}`);
            return;
        }
        
        this.gauges = [];
        this.init(gaugeData);
    }
    
    init(gaugeData) {
        // Crear elementos para los medidores
        gaugeData.forEach((data, index) => {
            // Crear contenedor para el medidor
            const gaugeElement = document.createElement('div');
            gaugeElement.className = 'gauge-item';
            gaugeElement.id = `gauge-${index}-${Date.now()}`;
            this.container.appendChild(gaugeElement);
            
            // Crear el medidor
            const gauge = new HolographicGauge(gaugeElement.id, data);
            this.gauges.push(gauge);
        });
    }
    
    // Actualizar todos los medidores con nuevos valores
    updateAll(values) {
        if (values.length !== this.gauges.length) {
            console.error('Error: La cantidad de valores no coincide con la cantidad de medidores');
            return;
        }
        
        values.forEach((value, index) => {
            this.gauges[index].update(value);
        });
    }
    
    // Actualizar un medidor específico
    updateGauge(index, value) {
        if (index < 0 || index >= this.gauges.length) {
            console.error(`Error: Índice ${index} fuera de rango`);
            return;
        }
        
        this.gauges[index].update(value);
    }
    
    // Simulación de datos en tiempo real para demostración
    startSimulation(interval = 3000) {
        this.simulationInterval = setInterval(() => {
            this.gauges.forEach(gauge => {
                const randomValue = Math.random() * (gauge.options.max - gauge.options.min) + gauge.options.min;
                gauge.update(randomValue);
            });
        }, interval);
    }
    
    stopSimulation() {
        if (this.simulationInterval) {
            clearInterval(this.simulationInterval);
        }
    }
}