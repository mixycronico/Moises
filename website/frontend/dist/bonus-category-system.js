/**
 * Sistema de Bonos y Categorías para Proto Genesis - Frontend
 * 
 * Este módulo implementa la interfaz de usuario para el sistema de bonos y categorías
 * en el dashboard del inversionista, permitiendo visualizar bonos recibidos,
 * simulaciones y categorías actuales.
 */

// Componente principal para el sistema de bonos y categorías
class BonusAndCategorySystem {
    constructor() {
        this.initialized = false;
        this.bonusHistory = null;
        this.bonusSimulation = null;
        this.lastRendered = null;
        this.loadingState = {
            history: false,
            simulation: false
        };
    }

    // Inicializar el sistema
    async initialize() {
        if (this.initialized) return true;
        
        try {
            // Verificar si está cargado el script del dashboard
            if (typeof GenesisUI === 'undefined') {
                console.error("El componente GenesisUI no está disponible. Cargando funcionalidad limitada.");
                return false;
            }
            
            // Registrar eventos y componentes
            this.registerUIComponents();
            
            // Cargar datos iniciales
            await this.loadBonusData();
            
            this.initialized = true;
            return true;
        } catch (error) {
            console.error("Error al inicializar sistema de bonos:", error);
            return false;
        }
    }
    
    // Registrar componentes de UI
    registerUIComponents() {
        // Agregar enlaces en el menú lateral
        GenesisUI.addMenuItem({
            id: 'bonus-system',
            text: 'Sistema de Bonos',
            icon: 'gift',
            section: 'investor',
            onClick: () => this.showBonusPanel()
        });
        
        GenesisUI.addMenuItem({
            id: 'investor-category',
            text: 'Mi Categoría',
            icon: 'star',
            section: 'investor',
            onClick: () => this.showCategoryPanel()
        });
        
        // Crear contenedores de paneles
        const mainContainer = document.querySelector('.dashboard-content');
        if (mainContainer) {
            // Panel de bonos
            const bonusPanel = document.createElement('div');
            bonusPanel.id = 'bonus-panel';
            bonusPanel.className = 'dashboard-panel';
            bonusPanel.style.display = 'none';
            
            // Panel de categorías
            const categoryPanel = document.createElement('div');
            categoryPanel.id = 'category-panel';
            categoryPanel.className = 'dashboard-panel';
            categoryPanel.style.display = 'none';
            
            mainContainer.appendChild(bonusPanel);
            mainContainer.appendChild(categoryPanel);
        }
    }
    
    // Cargar datos de bonos
    async loadBonusData() {
        try {
            this.loadingState.history = true;
            this.loadingState.simulation = true;
            
            // Historial de bonos
            const historyResponse = await fetch('/api/bonus/status');
            
            if (historyResponse.ok) {
                const historyData = await historyResponse.json();
                this.bonusHistory = historyData.data;
            } else {
                console.warn("No se pudo cargar el historial de bonos");
                this.bonusHistory = null;
            }
            
            // Simulación de bonos
            const simulationResponse = await fetch('/api/bonus/simulate');
            
            if (simulationResponse.ok) {
                const simulationData = await simulationResponse.json();
                this.bonusSimulation = simulationData.data;
            } else {
                console.warn("No se pudo cargar la simulación de bonos");
                this.bonusSimulation = null;
            }
            
            return true;
        } catch (error) {
            console.error("Error al cargar datos de bonos:", error);
            return false;
        } finally {
            this.loadingState.history = false;
            this.loadingState.simulation = false;
        }
    }
    
    // Mostrar panel de bonos
    async showBonusPanel() {
        try {
            // Ocultar todos los paneles
            document.querySelectorAll('.dashboard-panel').forEach(panel => {
                panel.style.display = 'none';
            });
            
            // Mostrar panel de bonos
            const bonusPanel = document.getElementById('bonus-panel');
            if (bonusPanel) {
                bonusPanel.style.display = 'block';
                
                // Actualizar datos si es necesario
                if (!this.bonusHistory || this.lastRendered && (Date.now() - this.lastRendered) > 60000) {
                    await this.loadBonusData();
                }
                
                // Renderizar contenido
                this.renderBonusPanel(bonusPanel);
                this.lastRendered = Date.now();
            }
        } catch (error) {
            console.error("Error al mostrar panel de bonos:", error);
            GenesisUI.showNotification('error', 'Error al cargar sistema de bonos');
        }
    }
    
    // Mostrar panel de categorías
    async showCategoryPanel() {
        try {
            // Ocultar todos los paneles
            document.querySelectorAll('.dashboard-panel').forEach(panel => {
                panel.style.display = 'none';
            });
            
            // Mostrar panel de categorías
            const categoryPanel = document.getElementById('category-panel');
            if (categoryPanel) {
                categoryPanel.style.display = 'block';
                
                // Actualizar datos si es necesario
                if (!this.bonusSimulation || this.lastRendered && (Date.now() - this.lastRendered) > 60000) {
                    await this.loadBonusData();
                }
                
                // Renderizar contenido
                this.renderCategoryPanel(categoryPanel);
                this.lastRendered = Date.now();
            }
        } catch (error) {
            console.error("Error al mostrar panel de categorías:", error);
            GenesisUI.showNotification('error', 'Error al cargar sistema de categorías');
        }
    }
    
    // Renderizar panel de bonos
    renderBonusPanel(container) {
        // Limpiar contenedor
        container.innerHTML = '';
        
        // Título del panel
        const title = document.createElement('h2');
        title.className = 'panel-title';
        title.innerHTML = '<i class="fas fa-gift"></i> Sistema de Bonos';
        container.appendChild(title);
        
        // Contenido principal
        const content = document.createElement('div');
        content.className = 'panel-content';
        
        if (this.loadingState.history) {
            content.innerHTML = '<div class="loading-spinner"></div><p>Cargando datos de bonos...</p>';
        } else if (!this.bonusHistory) {
            content.innerHTML = `
                <div class="info-card">
                    <div class="card-icon"><i class="fas fa-info-circle"></i></div>
                    <div class="card-content">
                        <h3>Sin historial disponible</h3>
                        <p>No se encontró historial de bonos o aún no eres elegible para recibir bonos.</p>
                        <p>Los inversionistas con más de 3 meses en el sistema y rendimiento excelente reciben bonos adicionales según su categoría.</p>
                    </div>
                </div>
            `;
        } else {
            // Información de elegibilidad
            const isEligible = this.bonusSimulation?.elegibilidad?.es_elegible;
            const daysToEligible = this.bonusSimulation?.elegibilidad?.dias_para_elegibilidad || 0;
            
            const eligibilityCard = document.createElement('div');
            eligibilityCard.className = 'info-card ' + (isEligible ? 'success-card' : 'warning-card');
            eligibilityCard.innerHTML = `
                <div class="card-icon"><i class="fas fa-${isEligible ? 'check-circle' : 'clock'}"></i></div>
                <div class="card-content">
                    <h3>${isEligible ? 'Elegible para bonos' : 'Aún no elegible'}</h3>
                    ${isEligible ? 
                        '<p>¡Felicidades! Ya eres elegible para recibir bonos por rendimiento excelente.</p>' : 
                        `<p>Necesitas ${daysToEligible} días más para ser elegible para bonos (mínimo 3 meses).</p>`
                    }
                </div>
            `;
            content.appendChild(eligibilityCard);
            
            // Resumen de bonos recibidos
            if (this.bonusHistory?.bonos?.length > 0) {
                const totalBonus = this.bonusHistory.resumen.total_bonos;
                const bonusCount = this.bonusHistory.bonos.length;
                
                const summaryCard = document.createElement('div');
                summaryCard.className = 'info-card success-card';
                summaryCard.innerHTML = `
                    <div class="card-icon"><i class="fas fa-award"></i></div>
                    <div class="card-content">
                        <h3>Resumen de Bonos</h3>
                        <p>Has recibido un total de ${bonusCount} bonos por un monto de $${totalBonus.toFixed(2)}.</p>
                        <div class="stat-container">
                            <div class="stat-item">
                                <span class="stat-title">Bonos diarios</span>
                                <span class="stat-value">${this.bonusHistory.resumen.por_tipo.daily_excellent?.cantidad || 0}</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-title">Bonos mensuales</span>
                                <span class="stat-value">${this.bonusHistory.resumen.por_tipo.monthly?.cantidad || 0}</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-title">Monto total</span>
                                <span class="stat-value">$${totalBonus.toFixed(2)}</span>
                            </div>
                        </div>
                    </div>
                `;
                content.appendChild(summaryCard);
                
                // Historial detallado
                const historyCard = document.createElement('div');
                historyCard.className = 'table-card';
                
                // Crear tabla de historial
                const historyTable = document.createElement('table');
                historyTable.className = 'data-table';
                historyTable.innerHTML = `
                    <thead>
                        <tr>
                            <th>Fecha</th>
                            <th>Descripción</th>
                            <th>Monto</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${this.bonusHistory.bonos.map(bonus => `
                            <tr>
                                <td>${new Date(bonus.fecha).toLocaleDateString()}</td>
                                <td>${bonus.descripcion}</td>
                                <td class="amount">$${bonus.monto.toFixed(2)}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                `;
                
                historyCard.appendChild(historyTable);
                content.appendChild(historyCard);
            } else {
                const noHistoryCard = document.createElement('div');
                noHistoryCard.className = 'info-card';
                noHistoryCard.innerHTML = `
                    <div class="card-icon"><i class="fas fa-info-circle"></i></div>
                    <div class="card-content">
                        <h3>Sin bonos recibidos</h3>
                        <p>Aún no has recibido bonos. Los bonos se otorgan en días de rendimiento excelente.</p>
                    </div>
                `;
                content.appendChild(noHistoryCard);
            }
            
            // Simulación de bonos potenciales
            if (this.bonusSimulation) {
                const category = this.bonusSimulation.inversionista.categoria;
                const capitalAmount = this.bonusSimulation.inversionista.capital;
                const dailyRate = this.bonusSimulation.bonos_potenciales.diario.tasa;
                const dailyAmount = this.bonusSimulation.bonos_potenciales.diario.monto_estimado;
                const monthlyRate = this.bonusSimulation.bonos_potenciales.mensual.tasa;
                const monthlyAmount = this.bonusSimulation.bonos_potenciales.mensual.monto_estimado;
                
                const simulationCard = document.createElement('div');
                simulationCard.className = 'info-card primary-card';
                simulationCard.innerHTML = `
                    <div class="card-icon"><i class="fas fa-calculator"></i></div>
                    <div class="card-content">
                        <h3>Simulación de Bonos</h3>
                        <p>Con tu categoría actual <strong>${category.toUpperCase()}</strong> y capital de <strong>$${capitalAmount.toFixed(2)}</strong>:</p>
                        <div class="stat-container">
                            <div class="stat-item">
                                <span class="stat-title">Bono diario (días excelentes)</span>
                                <span class="stat-value">${dailyRate}% = $${dailyAmount.toFixed(2)}</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-title">Bono mensual</span>
                                <span class="stat-value">${monthlyRate}% = $${monthlyAmount.toFixed(2)}</span>
                            </div>
                        </div>
                    </div>
                `;
                content.appendChild(simulationCard);
            }
        }
        
        container.appendChild(content);
    }
    
    // Renderizar panel de categorías
    renderCategoryPanel(container) {
        // Limpiar contenedor
        container.innerHTML = '';
        
        // Título del panel
        const title = document.createElement('h2');
        title.className = 'panel-title';
        title.innerHTML = '<i class="fas fa-star"></i> Mi Categoría de Inversionista';
        container.appendChild(title);
        
        // Contenido principal
        const content = document.createElement('div');
        content.className = 'panel-content';
        
        if (this.loadingState.simulation) {
            content.innerHTML = '<div class="loading-spinner"></div><p>Cargando datos de categoría...</p>';
        } else if (!this.bonusSimulation) {
            content.innerHTML = `
                <div class="info-card">
                    <div class="card-icon"><i class="fas fa-info-circle"></i></div>
                    <div class="card-content">
                        <h3>Información no disponible</h3>
                        <p>No se pudo cargar la información de tu categoría.</p>
                    </div>
                </div>
            `;
        } else {
            const category = this.bonusSimulation.inversionista.categoria;
            const capitalAmount = this.bonusSimulation.inversionista.capital;
            const creationDate = this.bonusSimulation.inversionista.creado ? new Date(this.bonusSimulation.inversionista.creado) : null;
            
            // Información de categoría actual
            const categoryIcons = {
                'platinum': 'crown',
                'gold': 'trophy',
                'silver': 'medal',
                'bronze': 'award'
            };
            
            const categoryColors = {
                'platinum': '#9C27B0', // Morado
                'gold': '#FFA000',     // Dorado
                'silver': '#78909C',   // Plateado
                'bronze': '#8D6E63'    // Bronce
            };
            
            const categoryDescriptions = {
                'platinum': 'Inversionistas élite con alto capital y rendimiento sostenido.',
                'gold': 'Inversionistas destacados con excelente rendimiento.',
                'silver': 'Inversionistas establecidos con buen rendimiento.',
                'bronze': 'Inversionistas en etapa inicial.'
            };
            
            const bonusRates = {
                'platinum': '10%',
                'gold': '7%',
                'silver': '5%',
                'bronze': '3%'
            };
            
            const icon = categoryIcons[category] || 'star';
            const color = categoryColors[category] || '#607D8B';
            const description = categoryDescriptions[category] || 'Categoría estándar de inversionista.';
            const bonusRate = bonusRates[category] || '0%';
            
            // Tarjeta de categoría
            const categoryCard = document.createElement('div');
            categoryCard.className = 'category-card';
            categoryCard.style.borderColor = color;
            categoryCard.innerHTML = `
                <div class="category-icon" style="background-color: ${color}">
                    <i class="fas fa-${icon}"></i>
                </div>
                <div class="category-content">
                    <h3 class="category-title" style="color: ${color}">${category.toUpperCase()}</h3>
                    <p class="category-description">${description}</p>
                    <div class="category-stats">
                        <div class="stat-item">
                            <span class="stat-title">Capital</span>
                            <span class="stat-value">$${capitalAmount.toFixed(2)}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-title">Tasa de bono</span>
                            <span class="stat-value">${bonusRate}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-title">Antigüedad</span>
                            <span class="stat-value">${creationDate ? Math.floor((new Date() - creationDate) / (1000 * 60 * 60 * 24)) : 'N/A'} días</span>
                        </div>
                    </div>
                </div>
            `;
            content.appendChild(categoryCard);
            
            // Información sobre las categorías
            const infoCard = document.createElement('div');
            infoCard.className = 'info-card';
            infoCard.innerHTML = `
                <div class="card-icon"><i class="fas fa-info-circle"></i></div>
                <div class="card-content">
                    <h3>Sistema de Categorías</h3>
                    <p>El sistema de categorías evalúa periódicamente a los inversionistas basado en:</p>
                    <ul>
                        <li><strong>Capital:</strong> Monto total invertido</li>
                        <li><strong>Antigüedad:</strong> Tiempo como inversionista</li>
                        <li><strong>Rendimiento:</strong> Resultados obtenidos</li>
                        <li><strong>Comportamiento:</strong> Consistencia en depósitos y operaciones</li>
                    </ul>
                </div>
            `;
            content.appendChild(infoCard);
            
            // Tabla comparativa de categorías
            const tableCard = document.createElement('div');
            tableCard.className = 'table-card';
            tableCard.innerHTML = `
                <h3>Comparativa de Categorías</h3>
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Categoría</th>
                            <th>Capital Mínimo</th>
                            <th>Antigüedad</th>
                            <th>Tasa de Bono</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr class="${category === 'platinum' ? 'highlighted-row' : ''}">
                            <td><i class="fas fa-crown"></i> Platinum</td>
                            <td>$100,000</td>
                            <td>12+ meses</td>
                            <td>10%</td>
                        </tr>
                        <tr class="${category === 'gold' ? 'highlighted-row' : ''}">
                            <td><i class="fas fa-trophy"></i> Gold</td>
                            <td>$50,000</td>
                            <td>6+ meses</td>
                            <td>7%</td>
                        </tr>
                        <tr class="${category === 'silver' ? 'highlighted-row' : ''}">
                            <td><i class="fas fa-medal"></i> Silver</td>
                            <td>$10,000</td>
                            <td>2+ meses</td>
                            <td>5%</td>
                        </tr>
                        <tr class="${category === 'bronze' ? 'highlighted-row' : ''}">
                            <td><i class="fas fa-award"></i> Bronze</td>
                            <td>$1,000</td>
                            <td>0+ meses</td>
                            <td>3%</td>
                        </tr>
                    </tbody>
                </table>
            `;
            content.appendChild(tableCard);
        }
        
        container.appendChild(content);
    }
}

// Inicializar cuando el documento esté listo
document.addEventListener('DOMContentLoaded', function() {
    // Verificar que estamos en la página del dashboard
    if (window.location.pathname.includes('portfolio') || 
        window.location.pathname.includes('investor') || 
        window.location.pathname.includes('admin')) {
        
        // Esperar a que GenesisUI esté cargado
        const waitForGenesisUI = setInterval(function() {
            if (typeof GenesisUI !== 'undefined') {
                clearInterval(waitForGenesisUI);
                
                // Inicializar sistema de bonos y categorías
                window.BonusSystem = new BonusAndCategorySystem();
                window.BonusSystem.initialize().then(initialized => {
                    if (initialized) {
                        console.log("Sistema de bonos y categorías inicializado correctamente.");
                    } else {
                        console.warn("Sistema de bonos y categorías inicializado con capacidades limitadas.");
                    }
                });
            }
        }, 100);
    }
});