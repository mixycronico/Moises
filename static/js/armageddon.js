/**
 * Controlador para la Prueba ARMAGEDÓN DIVINA
 * Sistema Genesis Trascendental - Versión 100% DIVINA
 * 
 * Este script maneja toda la interacción con la prueba ARMAGEDÓN DIVINA,
 * incluyendo inicialización, ejecución, monitoreo y visualización de resultados.
 */

document.addEventListener('DOMContentLoaded', function() {
    // Referencias a elementos del DOM
    const btnInitialize = document.getElementById('btn-initialize');
    const btnPrepareDb = document.getElementById('btn-prepare-db');
    const btnStartTest = document.getElementById('btn-start-test');
    const btnStopTest = document.getElementById('btn-stop-test');
    
    const systemStatusIndicator = document.getElementById('system-status-indicator');
    const systemStatus = document.getElementById('system-status');
    const progressContainer = document.getElementById('progress-container');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    
    const patternsExecuted = document.getElementById('patterns-executed');
    const elapsedTime = document.getElementById('elapsed-time');
    const operationsProcessed = document.getElementById('operations-processed');
    const operationsPerSecond = document.getElementById('operations-per-second');
    
    const resultsPanel = document.getElementById('results-panel');
    const successRate = document.getElementById('success-rate');
    const totalOperations = document.getElementById('total-operations');
    const opsPerSecond = document.getElementById('ops-per-second');
    const avgResponseTime = document.getElementById('avg-response-time');
    const concurrentPeak = document.getElementById('concurrent-peak');
    const totalDuration = document.getElementById('total-duration');
    
    const patternTable = document.getElementById('pattern-table').querySelector('tbody');
    
    const recoveryEvents = document.getElementById('recovery-events');
    const recoverySuccessRate = document.getElementById('recovery-success-rate');
    const recoveryAvgTime = document.getElementById('recovery-avg-time');
    
    const reportContainer = document.getElementById('report-container');
    const markdownReport = document.getElementById('markdown-report');
    
    // Variables de estado
    let isInitialized = false;
    let isDbPrepared = false;
    let isTestRunning = false;
    let testStartTime = null;
    let statusUpdateInterval = null;
    let testMonitorInterval = null;
    
    // Configuración ARMAGEDÓN
    const armageddonPatterns = [
        'DEVASTADOR_TOTAL',
        'AVALANCHA_CONEXIONES',
        'TSUNAMI_OPERACIONES',
        'SOBRECARGA_MEMORIA',
        'INYECCION_CAOS',
        'OSCILACION_EXTREMA',
        'INTERMITENCIA_BRUTAL',
        'APOCALIPSIS_FINAL'
    ];
    
    // Event Listeners
    btnInitialize.addEventListener('click', initializeSystem);
    btnPrepareDb.addEventListener('click', prepareDatabase);
    btnStartTest.addEventListener('click', startArmageddonTest);
    btnStopTest.addEventListener('click', stopArmageddonTest);
    
    // Funciones principales
    async function initializeSystem() {
        if (isInitialized) return;
        
        updateSystemStatus('Inicializando...', 'status-running');
        btnInitialize.disabled = true;
        
        try {
            const response = await fetch('/api/armageddon/initialize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const data = await response.json();
            
            if (data.success) {
                isInitialized = true;
                updateSystemStatus('Inicializado', 'status-success');
                btnPrepareDb.disabled = false;
                
                showToast('Sistema inicializado correctamente', 'success');
            } else {
                updateSystemStatus('Error de inicialización', 'status-error');
                btnInitialize.disabled = false;
                
                showToast(`Error al inicializar: ${data.error}`, 'error');
            }
        } catch (error) {
            console.error('Error al inicializar sistema:', error);
            updateSystemStatus('Error de conexión', 'status-error');
            btnInitialize.disabled = false;
            
            showToast('Error de conexión al inicializar sistema', 'error');
        }
    }
    
    async function prepareDatabase() {
        if (!isInitialized || isDbPrepared) return;
        
        updateSystemStatus('Preparando base de datos...', 'status-running');
        btnPrepareDb.disabled = true;
        
        try {
            const response = await fetch('/api/armageddon/prepare-db', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const data = await response.json();
            
            if (data.success) {
                isDbPrepared = true;
                updateSystemStatus('Base de datos preparada', 'status-success');
                btnStartTest.disabled = false;
                
                showToast('Base de datos preparada correctamente', 'success');
            } else {
                updateSystemStatus('Error al preparar BD', 'status-error');
                btnPrepareDb.disabled = false;
                
                // Permitir continuar aunque falle la preparación de la BD
                showToast(`Advertencia: ${data.error}. Puede continuar con la prueba.`, 'warning');
                btnStartTest.disabled = false;
            }
        } catch (error) {
            console.error('Error al preparar base de datos:', error);
            updateSystemStatus('Error de conexión', 'status-error');
            btnPrepareDb.disabled = false;
            
            // Permitir continuar aunque falle la preparación de la BD
            showToast('Error de conexión al preparar base de datos. Puede continuar.', 'warning');
            btnStartTest.disabled = false;
        }
    }
    
    async function startArmageddonTest() {
        if (isTestRunning) return;
        
        clearResults();
        resultsPanel.style.display = 'none';
        reportContainer.style.display = 'none';
        
        updateSystemStatus('ARMAGEDÓN EN PROGRESO', 'status-running');
        progressContainer.style.display = 'block';
        updateProgress(0);
        
        btnStartTest.disabled = true;
        btnStopTest.disabled = false;
        
        testStartTime = Date.now();
        isTestRunning = true;
        
        // Iniciar intervalos de actualización
        startStatusUpdates();
        
        try {
            const response = await fetch('/api/armageddon/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const data = await response.json();
            
            stopStatusUpdates();
            isTestRunning = false;
            btnStopTest.disabled = true;
            btnStartTest.disabled = false;
            
            if (data.success) {
                updateSystemStatus('ARMAGEDÓN COMPLETADO', 'status-success');
                updateProgress(100);
                
                showToast('Prueba ARMAGEDÓN completada con éxito', 'success');
                
                // Mostrar resultados
                displayTestResults(data.results);
            } else {
                updateSystemStatus('ERROR EN ARMAGEDÓN', 'status-error');
                
                showToast(`Error en prueba ARMAGEDÓN: ${data.error}`, 'error');
            }
        } catch (error) {
            console.error('Error al ejecutar prueba ARMAGEDÓN:', error);
            updateSystemStatus('Error de conexión', 'status-error');
            stopStatusUpdates();
            isTestRunning = false;
            btnStopTest.disabled = true;
            btnStartTest.disabled = false;
            
            showToast('Error de conexión al ejecutar prueba ARMAGEDÓN', 'error');
        }
    }
    
    async function stopArmageddonTest() {
        if (!isTestRunning) return;
        
        btnStopTest.disabled = true;
        
        try {
            const response = await fetch('/api/armageddon/stop', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const data = await response.json();
            
            if (data.success) {
                showToast('Prueba ARMAGEDÓN detenida correctamente', 'info');
            } else {
                showToast(`Error al detener prueba: ${data.error}`, 'error');
            }
        } catch (error) {
            console.error('Error al detener prueba:', error);
            showToast('Error de conexión al detener prueba', 'error');
        }
        
        isTestRunning = false;
        stopStatusUpdates();
        btnStartTest.disabled = false;
        updateSystemStatus('Prueba detenida', 'status-idle');
    }
    
    // Funciones de monitoreo y actualización
    function startStatusUpdates() {
        // Actualizar la interfaz cada 500ms
        statusUpdateInterval = setInterval(updateTestStatus, 500);
        
        // Verificar estado del test cada 2 segundos
        testMonitorInterval = setInterval(fetchTestStatus, 2000);
    }
    
    function stopStatusUpdates() {
        if (statusUpdateInterval) {
            clearInterval(statusUpdateInterval);
            statusUpdateInterval = null;
        }
        
        if (testMonitorInterval) {
            clearInterval(testMonitorInterval);
            testMonitorInterval = null;
        }
    }
    
    function updateTestStatus() {
        if (!isTestRunning || !testStartTime) return;
        
        // Actualizar tiempo transcurrido
        const currentTime = Date.now();
        const elapsed = Math.floor((currentTime - testStartTime) / 1000);
        elapsedTime.textContent = formatTime(elapsed);
    }
    
    async function fetchTestStatus() {
        if (!isTestRunning) return;
        
        try {
            const response = await fetch('/api/armageddon/status');
            const data = await response.json();
            
            if (data.running) {
                // Actualizar indicadores de progreso
                let progressPercentage = 0;
                if (data.current_pattern && armageddonPatterns.includes(data.current_pattern)) {
                    const patternIndex = armageddonPatterns.indexOf(data.current_pattern);
                    progressPercentage = Math.floor((patternIndex / armageddonPatterns.length) * 100);
                }
                
                updateProgress(progressPercentage);
                
                // Actualizar estadísticas
                patternsExecuted.textContent = `${data.current_pattern_index || 0}/8`;
                operationsProcessed.textContent = formatNumber(data.operations_total || 0);
                operationsPerSecond.textContent = formatNumber(data.operations_per_second || 0);
            } else if (data.completed) {
                // La prueba terminó, se manejará por la respuesta al inicio
                isTestRunning = false;
                stopStatusUpdates();
            }
        } catch (error) {
            console.error('Error al obtener estado de la prueba:', error);
        }
    }
    
    function updateSystemStatus(status, statusClass) {
        systemStatus.textContent = status;
        
        systemStatusIndicator.classList.remove('status-idle', 'status-running', 'status-success', 'status-error');
        systemStatusIndicator.classList.add(statusClass);
    }
    
    function updateProgress(percentage) {
        progressBar.style.width = `${percentage}%`;
        progressText.textContent = `${percentage}%`;
    }
    
    // Funciones para mostrar resultados
    function displayTestResults(results) {
        resultsPanel.style.display = 'block';
        
        const metrics = results.metrics_summary;
        const patterns = results.patterns_results;
        
        // Actualizar métricas generales
        successRate.textContent = `${metrics.success_rate.toFixed(2)}%`;
        totalOperations.textContent = formatNumber(metrics.operations_total);
        opsPerSecond.textContent = formatNumber(metrics.operations_per_second);
        avgResponseTime.textContent = `${metrics.latency_ms.avg.toFixed(2)} ms`;
        concurrentPeak.textContent = formatNumber(metrics.concurrent_peak);
        totalDuration.textContent = formatTime(results.duration);
        
        // Actualizar tabla de patrones
        patternTable.innerHTML = '';
        
        for (const [pattern, data] of Object.entries(patterns)) {
            const row = document.createElement('tr');
            
            const patternCell = document.createElement('td');
            patternCell.textContent = pattern;
            
            const successRateCell = document.createElement('td');
            const successRateValue = data.success_rate.toFixed(2);
            successRateCell.textContent = `${successRateValue}%`;
            
            // Colorear según tasa de éxito
            if (data.success_rate >= 98) {
                successRateCell.classList.add('success-rate-high');
            } else if (data.success_rate >= 90) {
                successRateCell.classList.add('success-rate-medium');
            } else {
                successRateCell.classList.add('success-rate-low');
            }
            
            const operationsCell = document.createElement('td');
            operationsCell.textContent = formatNumber(data.operations || 0);
            
            row.appendChild(patternCell);
            row.appendChild(successRateCell);
            row.appendChild(operationsCell);
            
            patternTable.appendChild(row);
        }
        
        // Actualizar métricas de recuperación
        recoveryEvents.textContent = formatNumber(metrics.recovery.events);
        recoverySuccessRate.textContent = `${metrics.recovery.success_rate.toFixed(2)}%`;
        recoveryAvgTime.textContent = `${metrics.recovery.avg_time_ms.toFixed(2)} ms`;
        
        // Cargar el reporte Markdown si está disponible
        if (results.report_path) {
            loadMarkdownReport(results.report_path);
        }
    }
    
    async function loadMarkdownReport(reportPath) {
        try {
            const response = await fetch(`/api/armageddon/report?path=${encodeURIComponent(reportPath)}`);
            const data = await response.json();
            
            if (data.success && data.content) {
                // Convertir Markdown a HTML
                markdownReport.innerHTML = marked.parse(data.content);
                reportContainer.style.display = 'block';
            } else {
                console.error('Error al cargar reporte:', data.error);
            }
        } catch (error) {
            console.error('Error al cargar reporte Markdown:', error);
        }
    }
    
    function clearResults() {
        // Limpiar métricas generales
        successRate.textContent = 'N/A';
        totalOperations.textContent = 'N/A';
        opsPerSecond.textContent = 'N/A';
        avgResponseTime.textContent = 'N/A';
        concurrentPeak.textContent = 'N/A';
        totalDuration.textContent = 'N/A';
        
        // Limpiar tabla de patrones
        patternTable.innerHTML = '';
        
        // Limpiar métricas de recuperación
        recoveryEvents.textContent = 'N/A';
        recoverySuccessRate.textContent = 'N/A';
        recoveryAvgTime.textContent = 'N/A';
        
        // Limpiar reporte
        markdownReport.innerHTML = '';
        
        // Reiniciar contadores de estado
        patternsExecuted.textContent = '0/8';
        elapsedTime.textContent = '0s';
        operationsProcessed.textContent = '0';
        operationsPerSecond.textContent = '0';
    }
    
    // Funciones auxiliares
    function formatNumber(num) {
        return new Intl.NumberFormat().format(num);
    }
    
    function formatTime(seconds) {
        if (seconds < 60) {
            return `${seconds}s`;
        } else if (seconds < 3600) {
            const minutes = Math.floor(seconds / 60);
            const remainingSeconds = seconds % 60;
            return `${minutes}m ${remainingSeconds}s`;
        } else {
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            const remainingSeconds = seconds % 60;
            return `${hours}h ${minutes}m ${remainingSeconds}s`;
        }
    }
    
    function showToast(message, type = 'info') {
        // Crear elemento de toast si no existe
        if (!document.getElementById('toast-container')) {
            const toastContainer = document.createElement('div');
            toastContainer.id = 'toast-container';
            toastContainer.style.position = 'fixed';
            toastContainer.style.bottom = '20px';
            toastContainer.style.right = '20px';
            toastContainer.style.zIndex = '9999';
            document.body.appendChild(toastContainer);
        }
        
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.style.backgroundColor = type === 'success' ? 'rgba(46, 125, 50, 0.9)' : 
                                     type === 'error' ? 'rgba(198, 40, 40, 0.9)' :
                                     type === 'warning' ? 'rgba(237, 108, 2, 0.9)' : 'rgba(2, 136, 209, 0.9)';
        toast.style.color = 'white';
        toast.style.padding = '12px 20px';
        toast.style.marginBottom = '10px';
        toast.style.borderRadius = '5px';
        toast.style.boxShadow = '0 4px 8px rgba(0, 0, 0, 0.2)';
        toast.style.minWidth = '250px';
        toast.style.opacity = '0';
        toast.style.transition = 'opacity 0.3s ease-in-out';
        
        const icon = document.createElement('i');
        icon.className = 'fas ' + (type === 'success' ? 'fa-check-circle' : 
                                 type === 'error' ? 'fa-exclamation-circle' :
                                 type === 'warning' ? 'fa-exclamation-triangle' : 'fa-info-circle');
        icon.style.marginRight = '10px';
        
        toast.appendChild(icon);
        toast.appendChild(document.createTextNode(message));
        
        document.getElementById('toast-container').appendChild(toast);
        
        setTimeout(() => {
            toast.style.opacity = '1';
        }, 10);
        
        setTimeout(() => {
            toast.style.opacity = '0';
            setTimeout(() => {
                toast.remove();
            }, 300);
        }, 5000);
    }
    
    // Inicialización adicional
    function checkAPIAvailability() {
        fetch('/api/armageddon/check')
            .then(response => response.json())
            .then(data => {
                if (data.available) {
                    btnInitialize.disabled = false;
                    showToast('API ARMAGEDÓN disponible. Listo para iniciar.', 'info');
                } else {
                    updateSystemStatus('API no disponible', 'status-error');
                    showToast('API ARMAGEDÓN no disponible. Contacte al administrador.', 'error');
                }
            })
            .catch(error => {
                console.error('Error al verificar API:', error);
                updateSystemStatus('Error de conexión', 'status-error');
                showToast('Error al verificar disponibilidad de API ARMAGEDÓN.', 'error');
            });
    }
    
    // Verificar disponibilidad de la API al cargar
    checkAPIAvailability();
});