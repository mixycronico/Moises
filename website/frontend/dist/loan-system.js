/**
 * Sistema de préstamos para inversionistas Genesis
 * 
 * Este script implementa la interfaz de usuario para la gestión de préstamos,
 * permitiendo a los inversionistas ver su estado de elegibilidad, solicitar
 * préstamos y ver el historial de pagos.
 */

// Clases para el sistema de préstamos
class LoanSystem {
  constructor() {
    this.initialized = false;
    this.loanStatus = null;
    this.activeLoan = null;
    this.eligibilityMessage = "";
    this.maxAmount = 0;
    this.isEligible = false;
  }

  // Inicializar sistema de préstamos
  async initialize() {
    if (this.initialized) return true;
    
    try {
      await this.refreshLoanStatus();
      this.initialized = true;
      return true;
    } catch (error) {
      console.error("Error inicializando sistema de préstamos:", error);
      return false;
    }
  }
  
  // Obtener estado actual de préstamos
  async refreshLoanStatus() {
    try {
      const response = await fetch('/api/investor/loan/status');
      const data = await response.json();
      
      if (data.success) {
        this.loanStatus = data.status;
        this.activeLoan = data.status.active_loan;
        this.eligibilityMessage = data.status.message;
        this.maxAmount = data.status.max_amount;
        this.isEligible = data.status.eligible;
        return data.status;
      } else {
        throw new Error(data.message || "Error desconocido al obtener estado de préstamos");
      }
    } catch (error) {
      console.error("Error al obtener estado de préstamos:", error);
      throw error;
    }
  }
  
  // Solicitar un préstamo
  async requestLoan(amount) {
    try {
      // Validaciones básicas
      if (!this.isEligible) {
        throw new Error("No eres elegible para solicitar un préstamo en este momento.");
      }
      
      if (!amount || isNaN(amount) || amount <= 0) {
        throw new Error("El monto debe ser un número positivo.");
      }
      
      if (amount > this.maxAmount) {
        throw new Error(`El monto máximo permitido es $${this.maxAmount.toFixed(2)}`);
      }
      
      // Enviar solicitud
      const response = await fetch('/api/investor/loan/request', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ amount })
      });
      
      const data = await response.json();
      
      if (data.success) {
        // Actualizar estado
        await this.refreshLoanStatus();
        return {
          success: true,
          message: data.message,
          loan: data.loan
        };
      } else {
        throw new Error(data.message || "Error al solicitar préstamo");
      }
    } catch (error) {
      console.error("Error solicitar préstamo:", error);
      return {
        success: false,
        message: error.message || "Error desconocido al solicitar préstamo"
      };
    }
  }
  
  // Realizar un pago manual a un préstamo
  async makePayment(loanId, amount) {
    try {
      // Validaciones básicas
      if (!this.activeLoan) {
        throw new Error("No tienes un préstamo activo.");
      }
      
      if (!loanId) {
        loanId = this.activeLoan.id;
      }
      
      if (!amount || isNaN(amount) || amount <= 0) {
        throw new Error("El monto debe ser un número positivo.");
      }
      
      // Enviar solicitud
      const response = await fetch(`/api/investor/loan/pay/${loanId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ amount })
      });
      
      const data = await response.json();
      
      if (data.success) {
        // Actualizar estado
        await this.refreshLoanStatus();
        return {
          success: true,
          message: data.message,
          newBalance: data.new_balance
        };
      } else {
        throw new Error(data.message || "Error al procesar pago");
      }
    } catch (error) {
      console.error("Error al realizar pago:", error);
      return {
        success: false,
        message: error.message || "Error desconocido al realizar pago"
      };
    }
  }
  
  // Renderizar UI del sistema de préstamos
  renderLoanUI(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    // Limpiar contenedor
    container.innerHTML = '';
    
    // Crear contenedor principal con estilo cósmico
    const loanSection = document.createElement('div');
    loanSection.className = 'loan-system-container';
    
    // Título con estilo
    const title = document.createElement('h2');
    title.className = 'loan-system-title';
    title.innerHTML = '<span class="loan-icon">💸</span> Sistema de Préstamos Genesis';
    loanSection.appendChild(title);
    
    // Estado de elegibilidad
    const statusCard = this._createStatusCard();
    loanSection.appendChild(statusCard);
    
    // Préstamo activo (si existe)
    if (this.activeLoan) {
      const activeLoanCard = this._createActiveLoanCard();
      loanSection.appendChild(activeLoanCard);
    } 
    // Formulario de solicitud (si es elegible)
    else if (this.isEligible) {
      const requestForm = this._createRequestForm();
      loanSection.appendChild(requestForm);
    }
    
    // Historial de préstamos (si hay)
    if (this.loanStatus && this.loanStatus.loan_history && this.loanStatus.loan_history.length > 0) {
      const historyCard = this._createHistoryCard();
      loanSection.appendChild(historyCard);
    }
    
    // Gráfico del sistema de préstamos
    const loanSystemInfo = this._createLoanSystemInfo();
    loanSection.appendChild(loanSystemInfo);
    
    // Agregar al contenedor
    container.appendChild(loanSection);
    
    // Inicializar eventos
    this._initEvents();
  }
  
  // Crear tarjeta de estado
  _createStatusCard() {
    const card = document.createElement('div');
    card.className = 'loan-status-card';
    
    // Contenido según estado
    let statusClass = 'status-not-eligible';
    let statusIcon = '⚠️';
    let statusTitle = 'No Elegible';
    
    if (this.isEligible) {
      statusClass = 'status-eligible';
      statusIcon = '✅';
      statusTitle = 'Elegible para Préstamo';
    } else if (this.activeLoan) {
      statusClass = 'status-active-loan';
      statusIcon = '💰';
      statusTitle = 'Préstamo Activo';
    }
    
    card.innerHTML = `
      <div class="status-header ${statusClass}">
        <span class="status-icon">${statusIcon}</span>
        <h3>${statusTitle}</h3>
      </div>
      <div class="status-content">
        <p class="status-message">${this.eligibilityMessage}</p>
        ${this.isEligible ? `<p class="max-amount">Monto máximo disponible: <strong>$${this.maxAmount.toFixed(2)}</strong></p>` : ''}
        ${this.loanStatus ? `<p class="investor-info">Tiempo como inversionista: <strong>${this.loanStatus.days_as_investor} días</strong></p>` : ''}
      </div>
    `;
    
    return card;
  }
  
  // Crear tarjeta de préstamo activo
  _createActiveLoanCard() {
    const card = document.createElement('div');
    card.className = 'active-loan-card';
    
    // Calcular porcentaje pagado
    const loan = this.activeLoan;
    const totalPaid = loan.amount - loan.remaining;
    const percentPaid = (totalPaid / loan.amount) * 100;
    
    card.innerHTML = `
      <div class="loan-header">
        <h3>Préstamo Activo</h3>
        <div class="loan-date">Solicitado: ${loan.date}</div>
      </div>
      <div class="loan-details">
        <div class="loan-amounts">
          <div class="amount-item">
            <span class="amount-label">Monto original:</span>
            <span class="amount-value">$${loan.amount.toFixed(2)}</span>
          </div>
          <div class="amount-item">
            <span class="amount-label">Pagado hasta ahora:</span>
            <span class="amount-value">$${totalPaid.toFixed(2)}</span>
          </div>
          <div class="amount-item">
            <span class="amount-label">Saldo pendiente:</span>
            <span class="amount-value">$${loan.remaining.toFixed(2)}</span>
          </div>
        </div>
        
        <div class="payment-progress">
          <div class="progress-bar-container">
            <div class="progress-bar" style="width: ${percentPaid}%"></div>
          </div>
          <div class="progress-label">${percentPaid.toFixed(1)}% completado</div>
        </div>
        
        <div class="payment-info">
          <p class="payment-details">
            El sistema descuenta automáticamente el <strong>30% de tus ganancias diarias</strong> para pagar esta deuda.
          </p>
          <p class="last-payment">Último pago: ${loan.last_payment}</p>
        </div>
        
        <div class="manual-payment">
          <h4>Realizar Pago Manual</h4>
          <div class="payment-form">
            <input type="number" id="manualPaymentAmount" placeholder="Monto a pagar" min="1" step="0.01" class="payment-input">
            <button id="makePaymentBtn" class="payment-button">Pagar</button>
          </div>
          <div id="paymentResult" class="payment-result"></div>
        </div>
      </div>
    `;
    
    return card;
  }
  
  // Crear formulario de solicitud
  _createRequestForm() {
    const form = document.createElement('div');
    form.className = 'loan-request-form';
    
    form.innerHTML = `
      <h3>Solicitar Préstamo</h3>
      <p class="form-description">
        Como inversionista calificado, puedes solicitar un préstamo de hasta <strong>$${this.maxAmount.toFixed(2)}</strong>.
        El préstamo se pagará automáticamente con el 30% de tus ganancias diarias.
      </p>
      
      <div class="form-group">
        <label for="loanAmount">Monto a solicitar:</label>
        <input type="number" id="loanAmount" placeholder="Ingresa el monto" min="1" max="${this.maxAmount}" step="0.01" class="form-input">
      </div>
      
      <div class="loan-calculator">
        <h4>Calculadora de pagos</h4>
        <div class="calculator-results">
          <div class="calc-item">Monto a solicitar: <span id="calcAmount">$0.00</span></div>
          <div class="calc-item">Pago diario (aproximado): <span id="calcDailyPayment">$0.00</span></div>
          <div class="calc-item">Tiempo estimado: <span id="calcTime">N/A</span></div>
        </div>
      </div>
      
      <div class="form-actions">
        <button id="requestLoanBtn" class="form-button">Solicitar Préstamo</button>
      </div>
      
      <div id="requestResult" class="request-result"></div>
    `;
    
    return form;
  }
  
  // Crear tarjeta de historial
  _createHistoryCard() {
    const card = document.createElement('div');
    card.className = 'loan-history-card';
    
    let historyItems = '';
    this.loanStatus.loan_history.forEach(loan => {
      historyItems += `
        <div class="history-item">
          <div class="history-header">
            <span class="history-id">Préstamo #${loan.id}</span>
            <span class="history-amount">$${loan.amount.toFixed(2)}</span>
          </div>
          <div class="history-details">
            <div class="history-date">Solicitado: ${loan.date}</div>
            <div class="history-paid">Pagado el: ${loan.paid_date}</div>
            <div class="history-days">Tiempo para pagar: ${loan.days_to_payoff} días</div>
          </div>
        </div>
      `;
    });
    
    card.innerHTML = `
      <h3>Historial de Préstamos</h3>
      <div class="history-list">
        ${historyItems}
      </div>
    `;
    
    return card;
  }
  
  // Crear información del sistema
  _createLoanSystemInfo() {
    const infoCard = document.createElement('div');
    infoCard.className = 'loan-system-info';
    
    infoCard.innerHTML = `
      <h3>Cómo Funciona el Sistema de Préstamos</h3>
      <div class="info-content">
        <div class="info-item">
          <div class="info-icon">⏱️</div>
          <div class="info-text">
            <h4>Período de Espera</h4>
            <p>Debes tener al menos <strong>3 meses</strong> como inversionista para acceder a préstamos.</p>
          </div>
        </div>
        
        <div class="info-item">
          <div class="info-icon">💰</div>
          <div class="info-text">
            <h4>Monto del Préstamo</h4>
            <p>Puedes solicitar hasta el <strong>40%</strong> de tu capital invertido.</p>
          </div>
        </div>
        
        <div class="info-item">
          <div class="info-icon">💸</div>
          <div class="info-text">
            <h4>Sistema de Pago</h4>
            <p>El sistema descuenta automáticamente el <strong>30%</strong> de tus ganancias diarias para pagar el préstamo.</p>
          </div>
        </div>
        
        <div class="info-item">
          <div class="info-icon">⚠️</div>
          <div class="info-text">
            <h4>Restricciones</h4>
            <p>No puedes tener más de un préstamo activo al mismo tiempo. Si deseas retirar tu capital, primero deberás pagar completamente cualquier préstamo pendiente.</p>
          </div>
        </div>
      </div>
    `;
    
    return infoCard;
  }
  
  // Inicializar eventos
  _initEvents() {
    // Formulario de solicitud
    const requestBtn = document.getElementById('requestLoanBtn');
    if (requestBtn) {
      requestBtn.addEventListener('click', async () => {
        const amountInput = document.getElementById('loanAmount');
        const resultDiv = document.getElementById('requestResult');
        
        if (!amountInput || !resultDiv) return;
        
        const amount = parseFloat(amountInput.value);
        resultDiv.innerHTML = '<div class="loading">Procesando solicitud...</div>';
        
        const result = await this.requestLoan(amount);
        
        if (result.success) {
          resultDiv.innerHTML = `<div class="success-message">${result.message}</div>`;
          // Volver a renderizar después de 2 segundos
          setTimeout(() => this.renderLoanUI('loanSystemContainer'), 2000);
        } else {
          resultDiv.innerHTML = `<div class="error-message">${result.message}</div>`;
        }
      });
      
      // Calculadora
      const amountInput = document.getElementById('loanAmount');
      if (amountInput) {
        amountInput.addEventListener('input', () => {
          const amount = parseFloat(amountInput.value) || 0;
          const calcAmount = document.getElementById('calcAmount');
          const calcDailyPayment = document.getElementById('calcDailyPayment');
          const calcTime = document.getElementById('calcTime');
          
          if (calcAmount) calcAmount.textContent = `$${amount.toFixed(2)}`;
          
          // Estimar pago diario (asumiendo una ganancia diaria del 0.5% del capital)
          const estimatedDailyProfit = this.maxAmount * 2.5 * 0.005; // 0.5% de 2.5 veces el préstamo máximo
          const dailyPayment = estimatedDailyProfit * 0.3; // 30% de las ganancias
          
          if (calcDailyPayment) calcDailyPayment.textContent = `$${dailyPayment.toFixed(2)}`;
          
          // Estimar tiempo (en días)
          if (amount > 0 && dailyPayment > 0) {
            const estimatedDays = Math.ceil(amount / dailyPayment);
            if (calcTime) calcTime.textContent = `Aproximadamente ${estimatedDays} días`;
          } else {
            if (calcTime) calcTime.textContent = 'N/A';
          }
        });
      }
    }
    
    // Formulario de pago manual
    const paymentBtn = document.getElementById('makePaymentBtn');
    if (paymentBtn) {
      paymentBtn.addEventListener('click', async () => {
        const amountInput = document.getElementById('manualPaymentAmount');
        const resultDiv = document.getElementById('paymentResult');
        
        if (!amountInput || !resultDiv) return;
        
        const amount = parseFloat(amountInput.value);
        resultDiv.innerHTML = '<div class="loading">Procesando pago...</div>';
        
        const result = await this.makePayment(this.activeLoan.id, amount);
        
        if (result.success) {
          resultDiv.innerHTML = `<div class="success-message">${result.message}</div>`;
          // Volver a renderizar después de 2 segundos
          setTimeout(() => this.renderLoanUI('loanSystemContainer'), 2000);
        } else {
          resultDiv.innerHTML = `<div class="error-message">${result.message}</div>`;
        }
      });
    }
  }
}

// Estilos CSS para el sistema de préstamos
const loanSystemStyles = `
.loan-system-container {
  font-family: 'Roboto', Arial, sans-serif;
  background: rgba(10, 10, 40, 0.4);
  border-radius: 12px;
  padding: 20px;
  color: #e5e5e5;
  margin-bottom: A0px;
  border: 1px solid rgba(128, 0, 255, 0.3);
  box-shadow: 0 0 15px rgba(128, 0, 255, 0.2);
}

.loan-system-title {
  display: flex;
  align-items: center;
  color: #ffffff;
  font-size: 24px;
  margin-bottom: 20px;
  padding-bottom: 10px;
  border-bottom: 1px solid rgba(128, 0, 255, 0.3);
}

.loan-icon {
  margin-right: 10px;
  font-size: 28px;
}

/* Tarjeta de estado */
.loan-status-card {
  background: rgba(20, 20, 50, 0.5);
  border-radius: 8px;
  overflow: hidden;
  margin-bottom: 20px;
  border: 1px solid rgba(128, 0, 255, 0.2);
}

.status-header {
  padding: 15px;
  display: flex;
  align-items: center;
}

.status-not-eligible {
  background: linear-gradient(135deg, rgba(180, 0, 0, 0.4), rgba(255, 0, 0, 0.2));
}

.status-eligible {
  background: linear-gradient(135deg, rgba(0, 180, 0, 0.4), rgba(0, 255, 0, 0.2));
}

.status-active-loan {
  background: linear-gradient(135deg, rgba(0, 0, 180, 0.4), rgba(0, 128, 255, 0.2));
}

.status-icon {
  font-size: 24px;
  margin-right: 10px;
}

.status-header h3 {
  margin: 0;
  font-size: 18px;
  color: #ffffff;
}

.status-content {
  padding: 15px;
}

.status-message {
  margin: 0 0 10px 0;
  line-height: 1.5;
}

.max-amount, .investor-info {
  font-size: 14px;
  margin: 5px 0;
}

.max-amount strong, .investor-info strong {
  color: rgba(128, 255, 255, 0.9);
}

/* Tarjeta de préstamo activo */
.active-loan-card {
  background: rgba(20, 20, 50, 0.5);
  border-radius: 8px;
  overflow: hidden;
  margin-bottom: 20px;
  border: 1px solid rgba(128, 0, 255, 0.2);
}

.loan-header {
  padding: 15px;
  background: linear-gradient(135deg, rgba(30, 30, 70, 0.7), rgba(60, 60, 100, 0.5));
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.loan-header h3 {
  margin: 0;
  font-size: 18px;
  color: #ffffff;
}

.loan-date {
  font-size: 14px;
  color: rgba(200, 200, 255, 0.8);
}

.loan-details {
  padding: 15px;
}

.loan-amounts {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 15px;
  margin-bottom: 15px;
}

.amount-item {
  display: flex;
  justify-content: space-between;
  padding: 8px;
  background: rgba(30, 30, 70, 0.3);
  border-radius: 5px;
}

.amount-label {
  color: rgba(200, 200, 255, 0.7);
}

.amount-value {
  font-weight: bold;
  color: rgba(128, 255, 255, 0.9);
}

.payment-progress {
  margin: 20px 0;
}

.progress-bar-container {
  height: 10px;
  background: rgba(30, 30, 70, 0.3);
  border-radius: 5px;
  overflow: hidden;
}

.progress-bar {
  height: 100%;
  background: linear-gradient(90deg, rgba(0, 128, 255, 0.8), rgba(128, 0, 255, 0.8));
  border-radius: 5px;
  transition: width 0.5s ease;
}

.progress-label {
  text-align: right;
  margin-top: 5px;
  font-size: 14px;
  color: rgba(200, 200, 255, 0.7);
}

.payment-info {
  margin: 15px 0;
  padding: 10px;
  background: rgba(30, 30, 70, 0.3);
  border-radius: 5px;
}

.payment-details {
  margin: 0 0 10px 0;
  line-height: 1.5;
}

.payment-details strong {
  color: rgba(128, 255, 255, 0.9);
}

.last-payment {
  font-size: 14px;
  color: rgba(200, 200, 255, 0.7);
  margin: 0;
}

.manual-payment {
  margin-top: 20px;
  padding: 15px;
  background: rgba(30, 30, 70, 0.3);
  border-radius: 5px;
}

.manual-payment h4 {
  margin: 0 0 10px 0;
  color: #ffffff;
}

.payment-form {
  display: flex;
  gap: 10px;
}

.payment-input {
  flex: 1;
  padding: 8px 12px;
  background: rgba(50, 50, 80, 0.5);
  border: 1px solid rgba(128, 0, 255, 0.3);
  border-radius: 5px;
  color: white;
  outline: none;
}

.payment-button {
  padding: 8px 15px;
  background: linear-gradient(135deg, rgba(128, 0, 255, 0.5), rgba(80, 0, 180, 0.7));
  border: 1px solid rgba(128, 0, 255, 0.5);
  border-radius: 5px;
  color: white;
  cursor: pointer;
  transition: all 0.3s;
}

.payment-button:hover {
  background: linear-gradient(135deg, rgba(148, 20, 255, 0.6), rgba(100, 20, 200, 0.8));
  box-shadow: 0 0 10px rgba(128, 0, 255, 0.4);
}

/* Formulario de solicitud */
.loan-request-form {
  background: rgba(20, 20, 50, 0.5);
  border-radius: 8px;
  overflow: hidden;
  margin-bottom: 20px;
  padding: 15px;
  border: 1px solid rgba(128, 0, 255, 0.2);
}

.loan-request-form h3 {
  margin: 0 0 15px 0;
  font-size: 18px;
  color: #ffffff;
}

.form-description {
  margin: 0 0 15px 0;
  line-height: 1.5;
}

.form-description strong {
  color: rgba(128, 255, 255, 0.9);
}

.form-group {
  margin-bottom: 15px;
}

.form-group label {
  display: block;
  margin-bottom: 5px;
  color: rgba(200, 200, 255, 0.9);
}

.form-input {
  width: 100%;
  padding: 10px;
  background: rgba(50, 50, 80, 0.5);
  border: 1px solid rgba(128, 0, 255, 0.3);
  border-radius: 5px;
  color: white;
  outline: none;
}

.loan-calculator {
  margin: 15px 0;
  padding: 10px;
  background: rgba(30, 30, 70, 0.3);
  border-radius: 5px;
}

.loan-calculator h4 {
  margin: 0 0 10px 0;
  color: #ffffff;
}

.calculator-results {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 10px;
}

.calc-item {
  font-size: 14px;
  color: rgba(200, 200, 255, 0.8);
}

.calc-item span {
  font-weight: bold;
  color: rgba(128, 255, 255, 0.9);
}

.form-actions {
  margin-top: 20px;
  display: flex;
  justify-content: flex-end;
}

.form-button {
  padding: 10px 20px;
  background: linear-gradient(135deg, rgba(0, 128, 255, 0.5), rgba(128, 0, 255, 0.7));
  border: 1px solid rgba(128, 0, 255, 0.5);
  border-radius: 5px;
  color: white;
  cursor: pointer;
  transition: all 0.3s;
}

.form-button:hover {
  background: linear-gradient(135deg, rgba(20, 148, 255, 0.6), rgba(148, 20, 255, 0.8));
  box-shadow: 0 0 10px rgba(128, 0, 255, 0.4);
}

/* Tarjeta de historial */
.loan-history-card {
  background: rgba(20, 20, 50, 0.5);
  border-radius: 8px;
  overflow: hidden;
  margin-bottom: 20px;
  padding: 15px;
  border: 1px solid rgba(128, 0, 255, 0.2);
}

.loan-history-card h3 {
  margin: 0 0 15px 0;
  font-size: 18px;
  color: #ffffff;
}

.history-list {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: 15px;
}

.history-item {
  background: rgba(30, 30, 70, 0.3);
  border-radius: 5px;
  padding: 10px;
  border: 1px solid rgba(128, 0, 255, 0.1);
}

.history-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 10px;
  padding-bottom: 5px;
  border-bottom: 1px solid rgba(128, 0, 255, 0.2);
}

.history-id {
  font-size: 14px;
  color: rgba(200, 200, 255, 0.7);
}

.history-amount {
  font-weight: bold;
  color: rgba(128, 255, 255, 0.9);
}

.history-details div {
  font-size: 13px;
  margin: 3px 0;
  color: rgba(200, 200, 255, 0.7);
}

/* Información del sistema */
.loan-system-info {
  background: rgba(20, 20, 50, 0.5);
  border-radius: 8px;
  overflow: hidden;
  padding: 15px;
  border: 1px solid rgba(128, 0, 255, 0.2);
}

.loan-system-info h3 {
  margin: 0 0 15px 0;
  font-size: 18px;
  color: #ffffff;
}

.info-content {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 15px;
}

.info-item {
  display: flex;
  background: rgba(30, 30, 70, 0.3);
  border-radius: 8px;
  padding: 12px;
}

.info-icon {
  font-size: 24px;
  margin-right: 15px;
  display: flex;
  align-items: center;
}

.info-text h4 {
  margin: 0 0 5px 0;
  color: rgba(200, 200, 255, 0.9);
}

.info-text p {
  margin: 0;
  font-size: 14px;
  line-height: 1.5;
  color: rgba(200, 200, 255, 0.7);
}

.info-text strong {
  color: rgba(128, 255, 255, 0.9);
}

/* Mensajes de resultado */
.request-result, .payment-result {
  margin-top: 15px;
}

.success-message {
  background: rgba(0, 180, 0, 0.2);
  border: 1px solid rgba(0, 255, 0, 0.3);
  color: rgba(200, 255, 200, 0.9);
  padding: 10px;
  border-radius: 5px;
}

.error-message {
  background: rgba(180, 0, 0, 0.2);
  border: 1px solid rgba(255, 0, 0, 0.3);
  color: rgba(255, 200, 200, 0.9);
  padding: 10px;
  border-radius: 5px;
}

.loading {
  color: rgba(200, 200, 255, 0.8);
  padding: 10px;
  text-align: center;
}

/* Responsive */
@media (max-width: 768px) {
  .loan-amounts, .calculator-results, .info-content {
    grid-template-columns: 1fr;
  }
  
  .history-list {
    grid-template-columns: 1fr;
  }
  
  .payment-form {
    flex-direction: column;
  }
  
  .payment-button {
    margin-top: 10px;
  }
}

/* Animaciones */
@keyframes glow {
  0% { box-shadow: 0 0 5px rgba(128, 0, 255, 0.3); }
  50% { box-shadow: 0 0 15px rgba(128, 0, 255, 0.5); }
  100% { box-shadow: 0 0 5px rgba(128, 0, 255, 0.3); }
}

.form-button, .payment-button {
  animation: glow 2s infinite;
}
`;

// Inicializar el sistema de préstamos cuando el DOM esté listo
document.addEventListener('DOMContentLoaded', async () => {
  // Crear estilos CSS
  const styleElement = document.createElement('style');
  styleElement.textContent = loanSystemStyles;
  document.head.appendChild(styleElement);
  
  // Crear contenedor para el sistema si no existe
  if (!document.getElementById('loanSystemContainer')) {
    // Buscar donde colocarlo (después de la sección de portafolio)
    const portfolioSection = document.querySelector('.investment-portfolio');
    if (portfolioSection) {
      const container = document.createElement('div');
      container.id = 'loanSystemContainer';
      container.className = 'loan-system-section';
      portfolioSection.parentNode.insertBefore(container, portfolioSection.nextSibling);
    }
  }
  
  // Inicializar sistema de préstamos
  window.loanSystem = new LoanSystem();
  await window.loanSystem.initialize();
  window.loanSystem.renderLoanUI('loanSystemContainer');
});

// Exponer sistema globalmente para acceso fácil desde otros scripts
window.LoanSystem = LoanSystem;