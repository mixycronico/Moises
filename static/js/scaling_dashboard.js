/**
 * Visualización de datos de escalabilidad de capital para el Sistema Genesis.
 * 
 * Este script genera tablas interactivas y visualizaciones para los datos
 * de escalabilidad de capital, mostrando la distribución óptima por instrumento,
 * puntos de saturación y otras métricas relacionadas.
 */

// Constantes de configuración
const COLORS = {
  primary: '#5690ff',
  secondary: '#84baff',
  accent: '#00b0ff',
  success: '#28a745',
  warning: '#ffc107',
  danger: '#dc3545',
  light: '#f8f9fa',
  dark: '#343a40',
  saturation: '#9c27b0',
  efficiency: '#00bcd4'
};

const CURRENCY_FORMATTER = new Intl.NumberFormat('es-ES', {
  style: 'currency',
  currency: 'USD',
  minimumFractionDigits: 2,
  maximumFractionDigits: 2
});

const PERCENT_FORMATTER = new Intl.NumberFormat('es-ES', {
  style: 'percent',
  minimumFractionDigits: 2,
  maximumFractionDigits: 2
});

/**
 * Crear tabla de distribución de capital.
 * 
 * @param {string} containerId - ID del contenedor HTML
 * @param {Object} data - Datos de distribución
 */
function createAllocationTable(containerId, data) {
  const container = document.getElementById(containerId);
  if (!container) return;
  
  // Crear tabla con estilos
  const table = document.createElement('table');
  table.className = 'table table-hover table-sm allocation-table';
  
  // Crear encabezado
  const thead = document.createElement('thead');
  const headerRow = document.createElement('tr');
  
  ['Instrumento', 'Asignación', '% del Capital', 'Punto de Saturación', 'Utilización'].forEach(text => {
    const th = document.createElement('th');
    th.textContent = text;
    headerRow.appendChild(th);
  });
  
  thead.appendChild(headerRow);
  table.appendChild(thead);
  
  // Crear cuerpo de la tabla
  const tbody = document.createElement('tbody');
  const { asignaciones, capital_total, puntos_saturacion } = data;
  
  // Ordenar por asignación descendente
  const instrumentos = Object.keys(asignaciones).sort((a, b) => {
    return asignaciones[b] - asignaciones[a];
  });
  
  instrumentos.forEach(instrumento => {
    const asignacion = asignaciones[instrumento];
    const porcentaje = asignacion / capital_total;
    const saturacion = puntos_saturacion[instrumento] || 0;
    const utilizacion = saturacion > 0 ? asignacion / saturacion : 0;
    
    const row = document.createElement('tr');
    
    // Columna instrumento
    const tdInstrumento = document.createElement('td');
    tdInstrumento.className = 'font-weight-bold';
    tdInstrumento.textContent = instrumento;
    row.appendChild(tdInstrumento);
    
    // Columna asignación
    const tdAsignacion = document.createElement('td');
    tdAsignacion.textContent = CURRENCY_FORMATTER.format(asignacion);
    row.appendChild(tdAsignacion);
    
    // Columna porcentaje
    const tdPorcentaje = document.createElement('td');
    tdPorcentaje.textContent = PERCENT_FORMATTER.format(porcentaje);
    row.appendChild(tdPorcentaje);
    
    // Columna punto saturación
    const tdSaturacion = document.createElement('td');
    tdSaturacion.textContent = CURRENCY_FORMATTER.format(saturacion);
    row.appendChild(tdSaturacion);
    
    // Columna utilización con barra de progreso
    const tdUtilizacion = document.createElement('td');
    const utilizacionFormateada = PERCENT_FORMATTER.format(utilizacion);
    
    const progressContainer = document.createElement('div');
    progressContainer.className = 'progress';
    progressContainer.style.height = '20px';
    
    const progressBar = document.createElement('div');
    progressBar.className = 'progress-bar';
    progressBar.style.width = `${utilizacion * 100}%`;
    progressBar.textContent = utilizacionFormateada;
    
    // Colorear según nivel de utilización
    if (utilizacion > 0.9) {
      progressBar.className += ' bg-danger';
    } else if (utilizacion > 0.7) {
      progressBar.className += ' bg-warning';
    } else {
      progressBar.className += ' bg-success';
    }
    
    progressContainer.appendChild(progressBar);
    tdUtilizacion.appendChild(progressContainer);
    row.appendChild(tdUtilizacion);
    
    tbody.appendChild(row);
  });
  
  table.appendChild(tbody);
  container.innerHTML = '';
  container.appendChild(table);
  
  // Añadir resumen 
  const summary = document.createElement('div');
  summary.className = 'allocation-summary mt-3';
  
  const totalCapital = document.createElement('p');
  totalCapital.innerHTML = `<strong>Capital Total:</strong> ${CURRENCY_FORMATTER.format(capital_total)}`;
  
  const totalAsignado = document.createElement('p');
  const sumaAsignaciones = Object.values(asignaciones).reduce((sum, val) => sum + val, 0);
  totalAsignado.innerHTML = `<strong>Capital Asignado:</strong> ${CURRENCY_FORMATTER.format(sumaAsignaciones)}`;
  
  const utilizacionCapital = document.createElement('p');
  const porcentajeUtilizacion = sumaAsignaciones / capital_total;
  utilizacionCapital.innerHTML = `<strong>Utilización Capital:</strong> ${PERCENT_FORMATTER.format(porcentajeUtilizacion)}`;
  
  summary.appendChild(totalCapital);
  summary.appendChild(totalAsignado);
  summary.appendChild(utilizacionCapital);
  container.appendChild(summary);
}

/**
 * Crear gráfico de distribución de capital.
 * 
 * @param {string} containerId - ID del contenedor HTML 
 * @param {Object} data - Datos de distribución
 */
function createAllocationChart(containerId, data) {
  const container = document.getElementById(containerId);
  if (!container || !window.Chart) return;
  
  const canvas = document.createElement('canvas');
  canvas.id = 'allocationChart';
  container.innerHTML = '';
  container.appendChild(canvas);
  
  const { asignaciones } = data;
  const instrumentos = Object.keys(asignaciones);
  const valores = instrumentos.map(i => asignaciones[i]);
  
  // Generar colores para cada instrumento
  const colors = instrumentos.map((_, i) => {
    const hue = (i * 137.5) % 360;
    return `hsl(${hue}, 70%, 60%)`;
  });
  
  // Crear gráfico
  new Chart(canvas, {
    type: 'doughnut',
    data: {
      labels: instrumentos,
      datasets: [{
        data: valores,
        backgroundColor: colors,
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      legend: {
        position: 'right'
      },
      title: {
        display: true,
        text: 'Distribución de Capital por Instrumento'
      },
      tooltips: {
        callbacks: {
          label: (tooltipItem, data) => {
            const dataset = data.datasets[tooltipItem.datasetIndex];
            const total = dataset.data.reduce((sum, val) => sum + val, 0);
            const valor = dataset.data[tooltipItem.index];
            const porcentaje = valor / total * 100;
            return `${data.labels[tooltipItem.index]}: ${CURRENCY_FORMATTER.format(valor)} (${porcentaje.toFixed(2)}%)`;
          }
        }
      }
    }
  });
}

/**
 * Crear gráfico de puntos de saturación.
 * 
 * @param {string} containerId - ID del contenedor HTML
 * @param {Object} data - Datos de saturación
 */
function createSaturationChart(containerId, data) {
  const container = document.getElementById(containerId);
  if (!container || !window.Chart) return;
  
  const canvas = document.createElement('canvas');
  canvas.id = 'saturationChart';
  container.innerHTML = '';
  container.appendChild(canvas);
  
  const { asignaciones, puntos_saturacion } = data;
  const instrumentos = Object.keys(asignaciones);
  
  // Preparar datos
  const asignacionesData = instrumentos.map(i => asignaciones[i]);
  const saturacionesData = instrumentos.map(i => puntos_saturacion[i] || 0);
  
  // Crear gráfico
  new Chart(canvas, {
    type: 'bar',
    data: {
      labels: instrumentos,
      datasets: [
        {
          label: 'Asignación Actual',
          data: asignacionesData,
          backgroundColor: COLORS.primary,
          borderColor: COLORS.primary,
          borderWidth: 1
        },
        {
          label: 'Punto de Saturación',
          data: saturacionesData,
          backgroundColor: COLORS.saturation,
          borderColor: COLORS.saturation,
          borderWidth: 1
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        yAxes: [{
          ticks: {
            beginAtZero: true,
            callback: (value) => {
              return CURRENCY_FORMATTER.format(value);
            }
          },
          scaleLabel: {
            display: true,
            labelString: 'Capital (USD)'
          }
        }]
      },
      tooltips: {
        callbacks: {
          label: (tooltipItem, data) => {
            const valor = tooltipItem.yLabel;
            return `${data.datasets[tooltipItem.datasetIndex].label}: ${CURRENCY_FORMATTER.format(valor)}`;
          }
        }
      }
    }
  });
}

/**
 * Crear tabla con métricas de eficiencia.
 * 
 * @param {string} containerId - ID del contenedor HTML
 * @param {Object} data - Datos de eficiencia
 */
function createEfficiencyTable(containerId, data) {
  const container = document.getElementById(containerId);
  if (!container) return;
  
  const { metricas, factor_escala } = data;
  
  // Crear tarjetas para métricas principales
  const row = document.createElement('div');
  row.className = 'row metrics-row';
  
  const createMetricCard = (title, value, color, icon, formatter) => {
    const formattedValue = formatter ? formatter(value) : value;
    
    const col = document.createElement('div');
    col.className = 'col-md-4 mb-3';
    
    const card = document.createElement('div');
    card.className = 'card h-100';
    card.style.borderLeft = `5px solid ${color}`;
    
    const cardBody = document.createElement('div');
    cardBody.className = 'card-body d-flex flex-column align-items-center';
    
    const cardTitle = document.createElement('div');
    cardTitle.className = 'card-title text-center text-muted mb-0';
    cardTitle.textContent = title;
    
    const cardValue = document.createElement('div');
    cardValue.className = 'card-value text-center display-4 mt-2';
    cardValue.textContent = formattedValue;
    
    const cardIcon = document.createElement('i');
    cardIcon.className = `fas ${icon} card-icon text-muted`;
    
    cardBody.appendChild(cardTitle);
    cardBody.appendChild(cardValue);
    cardBody.appendChild(cardIcon);
    card.appendChild(cardBody);
    col.appendChild(card);
    
    return col;
  };
  
  // Crear tarjetas de métricas
  row.appendChild(createMetricCard(
    'Eficiencia Promedio',
    metricas.eficiencia_promedio,
    COLORS.efficiency,
    'fa-chart-line',
    PERCENT_FORMATTER.format
  ));
  
  row.appendChild(createMetricCard(
    'Utilización de Capital',
    metricas.utilizacion_capital,
    COLORS.primary,
    'fa-coins',
    PERCENT_FORMATTER.format
  ));
  
  row.appendChild(createMetricCard(
    'Factor de Escala',
    factor_escala,
    COLORS.accent,
    'fa-expand-arrows-alt',
    (value) => value.toFixed(2) + 'x'
  ));
  
  container.innerHTML = '';
  container.appendChild(row);
  
  // Añadir entropía si está disponible
  if (metricas.entropia_asignacion !== undefined) {
    const entropiaRow = document.createElement('div');
    entropiaRow.className = 'row mt-3';
    
    const entropiaCol = document.createElement('div');
    entropiaCol.className = 'col-12';
    
    const entropiaCard = document.createElement('div');
    entropiaCard.className = 'card';
    
    const cardBody = document.createElement('div');
    cardBody.className = 'card-body';
    
    const cardTitle = document.createElement('h5');
    cardTitle.className = 'card-title';
    cardTitle.textContent = 'Entropía de Asignación';
    
    const progressContainer = document.createElement('div');
    progressContainer.className = 'progress mt-2';
    progressContainer.style.height = '30px';
    
    const entropia = metricas.entropia_asignacion;
    const progressBar = document.createElement('div');
    progressBar.className = 'progress-bar';
    progressBar.style.width = `${entropia * 100}%`;
    progressBar.textContent = PERCENT_FORMATTER.format(entropia);
    
    // Color según nivel de entropía (mayor es mejor en este caso)
    if (entropia > 0.8) {
      progressBar.className += ' bg-success';
    } else if (entropia > 0.5) {
      progressBar.className += ' bg-info';
    } else {
      progressBar.className += ' bg-warning';
    }
    
    const description = document.createElement('p');
    description.className = 'card-text mt-2';
    description.textContent = 'La entropía de asignación mide la diversificación del capital. Un valor cercano a 1 indica distribución óptima.';
    
    progressContainer.appendChild(progressBar);
    cardBody.appendChild(cardTitle);
    cardBody.appendChild(progressContainer);
    cardBody.appendChild(description);
    entropiaCard.appendChild(cardBody);
    entropiaCol.appendChild(entropiaCard);
    entropiaRow.appendChild(entropiaCol);
    
    container.appendChild(entropiaRow);
  }
}

/**
 * Inicializar tablero de escalabilidad.
 * 
 * @param {Object} data - Datos completos para el tablero
 */
function initScalingDashboard(data) {
  if (!data) return;
  
  // Inicializar componentes
  createAllocationTable('allocationTableContainer', data);
  createAllocationChart('allocationChartContainer', data);
  createSaturationChart('saturationChartContainer', data);
  createEfficiencyTable('efficiencyMetricsContainer', data);
  
  // Actualizar timestamp
  const timestampElement = document.getElementById('lastUpdateTimestamp');
  if (timestampElement) {
    const date = new Date(data.timestamp);
    timestampElement.textContent = `Última actualización: ${date.toLocaleString()}`;
  }
}

// Cargar datos y actualizar tablero
function loadScalingData() {
  fetch('/api/v1/scaling/current_allocation')
    .then(response => response.json())
    .then(data => {
      initScalingDashboard(data);
    })
    .catch(error => {
      console.error('Error cargando datos de escalabilidad:', error);
      document.getElementById('scalingDashboardError').classList.remove('d-none');
    });
}

// Inicializar cuando se carga la página
document.addEventListener('DOMContentLoaded', () => {
  loadScalingData();
  
  // Actualizar cada 5 minutos
  setInterval(loadScalingData, 5 * 60 * 1000);
  
  // Manejar botón de recarga manual
  const reloadButton = document.getElementById('reloadScalingData');
  if (reloadButton) {
    reloadButton.addEventListener('click', () => {
      loadScalingData();
    });
  }
});