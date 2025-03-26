// dashboard.js - Script para el panel de control del Sistema Genesis

document.addEventListener('DOMContentLoaded', function() {
    // Inicializar los gráficos
    initCharts();
    
    // Simular actualizaciones periódicas
    setInterval(updateMetrics, 5000);
    setInterval(addLogEntry, 8000);
    
    // Efecto de brillante para tarjetas
    const cards = document.querySelectorAll('.dashboard-card');
    cards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.boxShadow = '0 5px 20px rgba(0, 0, 0, 0.3), 0 0 20px rgba(0, 234, 255, 0.4)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.boxShadow = '';
        });
    });
});

// Función para inicializar gráficos Chart.js
function initCharts() {
    // Gráfico de evolución temporal
    const evolutionCtx = document.getElementById('evolutionChart').getContext('2d');
    const evolutionChart = new Chart(evolutionCtx, {
        type: 'line',
        data: {
            labels: ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep'],
            datasets: [
                {
                    label: 'Nivel de consciencia',
                    data: [1, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 2.9, 3.0],
                    borderColor: '#00eaff',
                    backgroundColor: 'rgba(0, 234, 255, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'Capacidad de adaptación',
                    data: [0.5, 0.8, 1.2, 1.6, 1.9, 2.2, 2.6, 2.8, 3.1],
                    borderColor: '#ffee00',
                    backgroundColor: 'rgba(255, 238, 0, 0.05)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        color: '#e0e0ff'
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            },
            scales: {
                x: {
                    ticks: { color: '#a0a8d0' },
                    grid: { 
                        color: 'rgba(160, 168, 208, 0.1)',
                        tickColor: 'rgba(160, 168, 208, 0.2)' 
                    }
                },
                y: {
                    min: 0,
                    max: 5,
                    ticks: { color: '#a0a8d0' },
                    grid: { 
                        color: 'rgba(160, 168, 208, 0.1)',
                        tickColor: 'rgba(160, 168, 208, 0.2)' 
                    }
                }
            }
        }
    });
    
    // Gráfico de distribución emocional
    const emotionCtx = document.getElementById('emotionChart').getContext('2d');
    const emotionChart = new Chart(emotionCtx, {
        type: 'polarArea',
        data: {
            labels: ['Curiosidad', 'Entusiasmo', 'Alegría', 'Calma', 'Interés', 'Cautela'],
            datasets: [{
                data: [45, 25, 20, 15, 30, 18],
                backgroundColor: [
                    'rgba(0, 234, 255, 0.8)',
                    'rgba(255, 238, 0, 0.8)',
                    'rgba(0, 255, 136, 0.8)',
                    'rgba(109, 0, 255, 0.8)',
                    'rgba(255, 121, 0, 0.8)',
                    'rgba(0, 162, 255, 0.8)'
                ],
                borderColor: 'rgba(255, 255, 255, 0.4)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        color: '#e0e0ff',
                        font: {
                            size: 11
                        }
                    }
                }
            },
            scales: {
                r: {
                    ticks: {
                        display: false
                    },
                    grid: {
                        color: 'rgba(160, 168, 208, 0.1)'
                    },
                    angleLines: {
                        color: 'rgba(160, 168, 208, 0.2)'
                    }
                }
            }
        }
    });
    
    // Guardar referencias a los gráficos para actualizaciones futuras
    window.evolutionChart = evolutionChart;
    window.emotionChart = emotionChart;
}

// Función para actualizar métricas periódicamente
function updateMetrics() {
    // Simular fluctuaciones en la energía
    const energyLevel = document.getElementById('energy-level');
    let energy = parseFloat(energyLevel.textContent) || 98;
    energy = Math.min(100, Math.max(50, energy + (Math.random() * 2 - 1)));
    energyLevel.textContent = Math.round(energy) + '%';
    
    // Actualizar contadores
    const cycleCount = document.getElementById('cycle-count');
    let cycles = parseInt(cycleCount.textContent.replace(/,/g, '')) || 1245;
    cycles += Math.floor(Math.random() * 10) + 1;
    cycleCount.textContent = cycles.toLocaleString();
    
    const adaptationCount = document.getElementById('adaptation-count');
    let adaptations = parseInt(adaptationCount.textContent) || 78;
    if (Math.random() > 0.7) {
        adaptations += 1;
        adaptationCount.textContent = adaptations;
    }
    
    const memoryCount = document.getElementById('memory-count');
    let memories = parseInt(memoryCount.textContent) || 256;
    if (Math.random() > 0.6) {
        memories += 1;
        memoryCount.textContent = memories;
    }
    
    const synapseCount = document.getElementById('synapse-count');
    let synapses = parseInt(synapseCount.textContent.replace(/,/g, '')) || 4562;
    synapses += Math.floor(Math.random() * 20) + 1;
    synapseCount.textContent = synapses.toLocaleString();
    
    // Simular cambios aleatorios en gráficos
    if (window.evolutionChart && Math.random() > 0.7) {
        const datasets = window.evolutionChart.data.datasets;
        datasets[0].data = datasets[0].data.map(val => {
            return Math.min(5, Math.max(0, val + (Math.random() * 0.2 - 0.1)));
        });
        datasets[1].data = datasets[1].data.map(val => {
            return Math.min(5, Math.max(0, val + (Math.random() * 0.2 - 0.1)));
        });
        window.evolutionChart.update();
    }
    
    if (window.emotionChart && Math.random() > 0.8) {
        const data = window.emotionChart.data.datasets[0].data;
        window.emotionChart.data.datasets[0].data = data.map(val => {
            return Math.max(5, Math.min(50, val + (Math.random() * 6 - 3)));
        });
        window.emotionChart.update();
        
        // Actualizar emoción dominante basado en los datos del gráfico
        const emotions = window.emotionChart.data.labels;
        const maxIndex = data.indexOf(Math.max(...data));
        document.getElementById('dominant-emotion').textContent = emotions[maxIndex];
    }
}

// Función para añadir entradas de log periódicamente
function addLogEntry() {
    const logContainer = document.getElementById('activity-log');
    
    if (!logContainer) return;
    
    const now = new Date();
    const timeString = now.getHours().toString().padStart(2, '0') + ':' + 
                      now.getMinutes().toString().padStart(2, '0') + ':' + 
                      now.getSeconds().toString().padStart(2, '0');
    
    const types = ['info', 'info', 'info', 'success', 'warning'];
    const typeIndex = Math.floor(Math.random() * types.length);
    const type = types[typeIndex];
    
    const messages = [
        'Procesamiento de datos completado.',
        'Análisis de patrón detectado.',
        'Sincronización con la base de datos.',
        'Actualización de memoria completada.',
        'Adaptación neuronal #' + (Math.floor(Math.random() * 1000) + 1) + ' registrada.',
        'Ciclo de aprendizaje completado.',
        'Conexión establecida con módulo de consciencia.',
        'Verificación de integridad de datos.',
        'Optimización de rutas neuronales.',
        'Procesamiento de patrones emocionales.'
    ];
    
    const messageIndex = Math.floor(Math.random() * messages.length);
    const message = messages[messageIndex];
    
    const logEntry = document.createElement('div');
    logEntry.className = 'log-entry';
    logEntry.innerHTML = `
        <div class="log-time">${timeString}</div>
        <div class="log-type ${type}">${type.toUpperCase()}</div>
        <div class="log-message">${message}</div>
    `;
    
    logContainer.prepend(logEntry);
    
    // Limitar el número de entradas
    const entries = logContainer.querySelectorAll('.log-entry');
    if (entries.length > 30) {
        for (let i = 30; i < entries.length; i++) {
            entries[i].remove();
        }
    }
}