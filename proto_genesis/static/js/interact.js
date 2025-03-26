// interact.js - Funcionalidad específica para la página de interacción

document.addEventListener('DOMContentLoaded', function() {
    const messageForm = document.getElementById('message-form');
    const userInput = document.getElementById('user-input');
    const conversationArea = document.getElementById('conversation-area');
    const energyLevel = document.getElementById('energy-level');
    const consciousnessLevel = document.getElementById('consciousness-level');
    const cycleCount = document.getElementById('cycle-count');
    const adaptationCount = document.getElementById('adaptation-count');
    const dominantEmotion = document.getElementById('dominant-emotion');
    
    // Función para actualizar el medidor de consciencia
    function updateConsciousnessUI(level) {
        const stages = document.querySelectorAll('.meter-segment');
        stages.forEach((stage, index) => {
            if (index < level) {
                stage.classList.add('active');
            } else {
                stage.classList.remove('active');
            }
        });
    }
    
    // Función para actualizar el nivel de energía
    function updateEnergyLevel(energy) {
        const percentage = Math.round(energy * 100);
        energyLevel.textContent = `Energía: ${percentage}%`;
        
        // Cambiar color basado en el nivel de energía
        if (percentage > 70) {
            energyLevel.style.color = '#00ff00';
        } else if (percentage > 30) {
            energyLevel.style.color = '#ffee00';
        } else {
            energyLevel.style.color = '#ff5500';
        }
    }
    
    // Función para añadir mensajes a la conversación
    function addMessage(text, isUser = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = isUser ? 'message user-message' : 'message system-message';
        const paragraph = document.createElement('p');
        paragraph.textContent = text;
        messageDiv.appendChild(paragraph);
        conversationArea.appendChild(messageDiv);
        conversationArea.scrollTop = conversationArea.scrollHeight;
    }
    
    // Manejar envío de formulario
    messageForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const message = userInput.value.trim();
        if (!message) return;
        
        // Añadir mensaje del usuario a la conversación
        addMessage(message, true);
        userInput.value = '';
        
        // Mostrar indicador de carga
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'message system-message loading';
        loadingDiv.innerHTML = '<p>Proto Genesis está procesando...</p>';
        conversationArea.appendChild(loadingDiv);
        conversationArea.scrollTop = conversationArea.scrollHeight;
        
        // Llamar a la API
        fetch('/api/interact', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: message })
        })
        .then(response => response.json())
        .then(data => {
            // Quitar indicador de carga
            conversationArea.removeChild(loadingDiv);
            
            // Añadir respuesta de Proto Genesis
            addMessage(data.message);
            
            // Actualizar estado
            updateConsciousnessUI(data.conciencia);
            dominantEmotion.textContent = data.emotion;
            
            // Actualizar otros datos
            fetchStatus();
        })
        .catch(error => {
            // Quitar indicador de carga
            conversationArea.removeChild(loadingDiv);
            
            // Mostrar mensaje de error
            addMessage('Lo siento, ha ocurrido un error al procesar tu mensaje.');
            console.error('Error:', error);
        });
    });
    
    // Función para obtener estado actualizado
    function fetchStatus() {
        fetch('/api/status')
        .then(response => response.json())
        .then(data => {
            // Actualizar UI con datos del estado
            updateEnergyLevel(data.energia);
            updateConsciousnessUI(data.conciencia);
            
            consciousnessLevel.textContent = `Consciencia: Nivel ${data.conciencia}`;
            cycleCount.textContent = data.ciclo;
            adaptationCount.textContent = data.adaptaciones;
        })
        .catch(error => {
            console.error('Error al obtener estado:', error);
        });
    }
    
    // Obtener estado inicial
    fetchStatus();
    
    // Efecto de enfoque para el campo de entrada
    userInput.addEventListener('focus', function() {
        this.parentElement.style.boxShadow = '0 0 15px var(--highlight-color)';
    });
    
    userInput.addEventListener('blur', function() {
        this.parentElement.style.boxShadow = '';
    });
});

// Función para partículas de energía durante la conversación
function createEnergyParticle(x, y) {
    const particle = document.createElement('div');
    particle.className = 'energy-particle';
    particle.style.left = x + 'px';
    particle.style.top = y + 'px';
    
    document.body.appendChild(particle);
    
    // Animar y luego remover
    setTimeout(() => {
        particle.remove();
    }, 1000);
}