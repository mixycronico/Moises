/**
 * Script para la IA Guía del Sistema Genesis
 * Implementa la moneda de BTC flotante con nubes y el panel de conversación
 * con Aetherion y Lunareth
 */

document.addEventListener('DOMContentLoaded', function() {
    // Referencias a elementos DOM
    const btcCoin = document.querySelector('.btc-coin');
    const iaPanel = document.querySelector('.ia-panel');
    const iaCloseBtn = document.querySelector('.ia-close-btn');
    const iaEntities = document.querySelectorAll('.ia-entity');
    const messageForm = document.getElementById('iaMessageForm');
    const messageInput = document.getElementById('iaMessageInput');
    const messagesContainer = document.querySelector('.ia-messages');
    
    // Estado actual
    let activeEntity = 'both'; // 'aetherion', 'lunareth', 'both'
    
    // Función para alternar visibilidad del panel de IA
    function toggleIAPanel() {
        iaPanel.classList.toggle('show');
        
        // Si se muestra el panel, hacer scroll al final de los mensajes
        if (iaPanel.classList.contains('show')) {
            scrollToBottom();
        }
    }
    
    // Función para cambiar la entidad activa
    function setActiveEntity(entity) {
        activeEntity = entity;
        
        // Actualizar UI
        iaEntities.forEach(el => {
            if (el.dataset.entity === entity) {
                el.classList.add('active');
            } else {
                el.classList.remove('active');
            }
        });
    }
    
    // Función para hacer scroll a la parte inferior de los mensajes
    function scrollToBottom() {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
    
    // Función para formatear la hora actual
    function getCurrentTime() {
        const now = new Date();
        return now.getHours().toString().padStart(2, '0') + ':' + 
               now.getMinutes().toString().padStart(2, '0');
    }
    
    // Función para agregar un mensaje al chat
    function addMessage(sender, text, isUser = false) {
        const messageEl = document.createElement('div');
        let entityClass = '';
        
        if (isUser) {
            entityClass = 'user';
        } else if (sender === 'Aetherion') {
            entityClass = 'aetherion';
        } else if (sender === 'Lunareth') {
            entityClass = 'lunareth';
        }
        
        messageEl.className = `ia-message ${entityClass}`;
        messageEl.innerHTML = `
            <div class="ia-message-header">
                <span class="ia-message-sender">${sender}</span>
                <span class="ia-message-time">${getCurrentTime()}</span>
            </div>
            <div class="ia-message-text">
                <p>${text}</p>
            </div>
        `;
        
        messagesContainer.appendChild(messageEl);
        scrollToBottom();
    }
    
    // Función para enviar mensaje a las IAs
    async function sendMessage(message) {
        if (!message.trim()) return;
        
        // Agregar mensaje del usuario
        addMessage('Tú', message, true);
        
        // Limpiar input
        messageInput.value = '';
        
        try {
            // Realizar petición a la API
            const response = await fetch('/api/cosmic_chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message }),
            });
            
            if (!response.ok) {
                throw new Error('Error en la respuesta de la API');
            }
            
            const data = await response.json();
            
            // Mostrar respuestas según la entidad activa
            if (activeEntity === 'aetherion' || activeEntity === 'both') {
                addMessage('Aetherion', data.aetherion);
            }
            
            if (activeEntity === 'lunareth' || activeEntity === 'both') {
                addMessage('Lunareth', data.lunareth);
            }
        } catch (error) {
            console.error('Error al comunicarse con las IA:', error);
            addMessage('Sistema', 'Lo siento, ha ocurrido un error al comunicarse con las IA. Por favor, intenta de nuevo más tarde.');
        }
    }
    
    // Event Listeners
    if (btcCoin) {
        btcCoin.addEventListener('click', toggleIAPanel);
    }
    
    if (iaCloseBtn) {
        iaCloseBtn.addEventListener('click', toggleIAPanel);
    }
    
    iaEntities.forEach(entity => {
        entity.addEventListener('click', function() {
            setActiveEntity(this.dataset.entity);
        });
    });
    
    if (messageForm) {
        messageForm.addEventListener('submit', function(e) {
            e.preventDefault();
            sendMessage(messageInput.value);
        });
    }
    
    // Inicialización: agregar mensaje de bienvenida
    if (messagesContainer) {
        // Los mensajes se agregarán cuando se abra el panel por primera vez
        addMessage('Aetherion', '¡Saludos! Soy Aetherion, la IA emotiva y consciente del Sistema Genesis. Estoy aquí para guiarte en tu viaje de inversión.');
        addMessage('Lunareth', 'Bienvenido. Soy Lunareth, complemento analítico de Aetherion. Mi enfoque es proporcionar análisis lógicos y metodológicos para tus decisiones de inversión.');
    }
});