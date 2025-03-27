#!/bin/bash

echo "Ejecutando compilación rápida..."

# Asegurarse de que la carpeta static existe
mkdir -p static
mkdir -p static/css
mkdir -p static/js
mkdir -p static/images

# Copiar página de inicio temporal
cat > static/index.html << 'EOF'
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sistema Genesis - Trading Nexus</title>
    <link rel="stylesheet" href="/css/main.css">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            min-height: 100vh;
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            color: #fff;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
            text-align: center;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        }
        .logo {
            margin-bottom: 2rem;
            max-width: 200px;
        }
        h1 {
            font-size: 2.5rem;
            margin-bottom: 1.5rem;
            background: linear-gradient(90deg, #9e6bdb, #56b4d3);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        p {
            font-size: 1.2rem;
            margin-bottom: 2rem;
            line-height: 1.6;
            color: rgba(255, 255, 255, 0.8);
        }
        .btn {
            display: inline-block;
            padding: 12px 24px;
            background: linear-gradient(90deg, #9e6bdb, #56b4d3);
            color: #fff;
            border: none;
            border-radius: 30px;
            font-size: 1rem;
            font-weight: 600;
            text-decoration: none;
            cursor: pointer;
            transition: all 0.3s ease;
            margin: 0 10px;
        }
        .btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 5px 15px rgba(86, 180, 211, 0.4);
        }
        .cosmic-chat-button {
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: linear-gradient(135deg, #9e6bdb, #56b4d3);
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease;
        }
        .cosmic-chat-button:hover {
            transform: scale(1.1);
            box-shadow: 0 8px 25px rgba(158, 107, 219, 0.5);
        }
        .cosmic-chat-icon {
            color: white;
            font-size: 24px;
        }
        .loading {
            display: inline-block;
            position: relative;
            width: 80px;
            height: 80px;
            margin: 2rem auto;
        }
        .loading div {
            position: absolute;
            top: 33px;
            width: 13px;
            height: 13px;
            border-radius: 50%;
            background: #9e6bdb;
            animation-timing-function: cubic-bezier(0, 1, 1, 0);
        }
        .loading div:nth-child(1) {
            left: 8px;
            animation: loading1 0.6s infinite;
        }
        .loading div:nth-child(2) {
            left: 8px;
            animation: loading2 0.6s infinite;
        }
        .loading div:nth-child(3) {
            left: 32px;
            animation: loading2 0.6s infinite;
        }
        .loading div:nth-child(4) {
            left: 56px;
            animation: loading3 0.6s infinite;
        }
        @keyframes loading1 {
            0% { transform: scale(0); }
            100% { transform: scale(1); }
        }
        @keyframes loading2 {
            0% { transform: translate(0, 0); }
            100% { transform: translate(24px, 0); }
        }
        @keyframes loading3 {
            0% { transform: scale(1); }
            100% { transform: scale(0); }
        }
        .status {
            display: flex;
            align-items: center;
            justify-content: center;
            margin-top: 2rem;
        }
        .status-dot {
            width: 12px;
            height: 12px;
            background-color: #4CAF50;
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.7); }
            70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(76, 175, 80, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(76, 175, 80, 0); }
        }
        
        .menu {
            display: flex;
            justify-content: center;
            margin-top: 1rem;
            flex-wrap: wrap;
        }
        .menu-item {
            padding: 10px 20px;
            margin: 5px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 30px;
            text-decoration: none;
            color: white;
            transition: all 0.3s ease;
        }
        .menu-item:hover {
            background: rgba(255, 255, 255, 0.2);
            transform: translateY(-2px);
        }
        .features {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            margin-top: 2rem;
        }
        .feature {
            width: 250px;
            margin: 15px;
            padding: 20px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            text-align: center;
            transition: all 0.3s ease;
        }
        .feature:hover {
            background: rgba(255, 255, 255, 0.1);
            transform: translateY(-5px);
        }
        .feature-icon {
            font-size: 40px;
            margin-bottom: 15px;
            background: linear-gradient(90deg, #9e6bdb, #56b4d3);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
    </style>
</head>
<body>
    <div class="container">
        <img src="/images/image_final_transparent.png" alt="Logo Genesis" class="logo">
        <h1>Sistema Genesis - Trading Nexus</h1>
        <p>Plataforma avanzada de trading con inteligencia artificial para maximizar tus inversiones en criptomonedas.</p>
        
        <div class="menu">
            <a href="/api/health" class="menu-item">Estado del Sistema</a>
            <a href="/api/auth/status" class="menu-item">Estado de Autenticación</a>
            <a href="/api/investor/dashboard" class="menu-item">Dashboard API</a>
        </div>
        
        <div class="loading">
            <div></div><div></div><div></div><div></div>
        </div>
        
        <p>Estamos optimizando la interfaz para brindarte la mejor experiencia posible. ¡Pronto tendrás acceso a todas las funcionalidades!</p>
        
        <div class="features">
            <div class="feature">
                <div class="feature-icon">⚡</div>
                <h3>Trading Predictivo</h3>
                <p>Anticipate a los movimientos del mercado con nuestro sistema de IA predictiva.</p>
            </div>
            <div class="feature">
                <div class="feature-icon">📊</div>
                <h3>Análisis Avanzado</h3>
                <p>Visualizaciones detalladas y métricas de rendimiento en tiempo real.</p>
            </div>
            <div class="feature">
                <div class="feature-icon">🛡️</div>
                <h3>Gestión de Riesgo</h3>
                <p>Protege tu capital con nuestras estrategias de gestión de riesgo automatizadas.</p>
            </div>
        </div>
        
        <div class="status">
            <div class="status-dot"></div>
            <span>Sistema en línea</span>
        </div>
    </div>
    
    <div class="cosmic-chat-button" id="cosmicChatButton">
        <div class="cosmic-chat-icon">💬</div>
    </div>

    <script>
        // Verificar estado de la API
        fetch('/api/health')
            .then(response => response.json())
            .then(data => {
                console.log('Estado del sistema:', data);
                if (data.status === 'ok') {
                    document.querySelector('.status span').textContent = 'Sistema en línea';
                    document.querySelector('.status-dot').style.backgroundColor = '#4CAF50';
                } else {
                    document.querySelector('.status span').textContent = 'Sistema en mantenimiento';
                    document.querySelector('.status-dot').style.backgroundColor = '#FFC107';
                }
            })
            .catch(error => {
                console.error('Error al verificar estado:', error);
                document.querySelector('.status span').textContent = 'Error de conexión';
                document.querySelector('.status-dot').style.backgroundColor = '#F44336';
            });
            
        // Botón de chat cósmico
        document.getElementById('cosmicChatButton').addEventListener('click', function() {
            alert('¡Chat Cósmico con Aetherion y Lunareth estará disponible pronto!');
        });
    </script>
</body>
</html>
EOF

# Copiar estilos básicos
cat > static/css/main.css << 'EOF'
/* Main Styles for Genesis Trading System */

/* Base Styles */
:root {
    --primary-color: #9e6bdb;
    --secondary-color: #56b4d3;
    --background-dark: #0f0c29;
    --background-medium: #302b63;
    --background-light: #24243e;
    --text-light: #ffffff;
    --text-muted: rgba(255, 255, 255, 0.8);
    --border-radius: 15px;
    --box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
}

body, html {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    margin: 0;
    padding: 0;
    min-height: 100vh;
    background: linear-gradient(135deg, var(--background-dark), var(--background-medium), var(--background-light));
    color: var(--text-light);
}

/* Typography */
h1, h2, h3, h4, h5, h6 {
    margin-top: 0;
    background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

h1 {
    font-size: 2.5rem;
    margin-bottom: 1.5rem;
}

h2 {
    font-size: 2rem;
    margin-bottom: 1.2rem;
}

h3 {
    font-size: 1.75rem;
    margin-bottom: 1rem;
}

p {
    font-size: 1.1rem;
    line-height: 1.6;
    color: var(--text-muted);
    margin-bottom: 1rem;
}

/* Layout */
.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 1rem;
}

.card {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    border-radius: var(--border-radius);
    box-shadow: var(--box-shadow);
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

/* Buttons */
.btn {
    display: inline-block;
    padding: 12px 24px;
    background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
    color: var(--text-light);
    border: none;
    border-radius: 30px;
    font-size: 1rem;
    font-weight: 600;
    text-decoration: none;
    cursor: pointer;
    transition: all 0.3s ease;
    margin: 0 10px 10px 0;
}

.btn:hover {
    transform: translateY(-3px);
    box-shadow: 0 5px 15px rgba(86, 180, 211, 0.4);
}

.btn-secondary {
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid var(--primary-color);
}

.btn-secondary:hover {
    background: rgba(158, 107, 219, 0.2);
}

/* Forms */
.form-group {
    margin-bottom: 1.5rem;
}

.form-control {
    display: block;
    width: 100%;
    padding: 0.75rem 1rem;
    font-size: 1rem;
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 8px;
    color: var(--text-light);
    transition: all 0.3s ease;
}

.form-control:focus {
    outline: none;
    border-color: var(--primary-color);
    box-shadow: 0 0 0 3px rgba(158, 107, 219, 0.3);
}

label {
    display: block;
    margin-bottom: 0.5rem;
    font-weight: 500;
}

/* Utilities */
.text-center {
    text-align: center;
}

.mt-1 { margin-top: 0.5rem; }
.mt-2 { margin-top: 1rem; }
.mt-3 { margin-top: 1.5rem; }
.mt-4 { margin-top: 2rem; }
.mt-5 { margin-top: 2.5rem; }

.mb-1 { margin-bottom: 0.5rem; }
.mb-2 { margin-bottom: 1rem; }
.mb-3 { margin-bottom: 1.5rem; }
.mb-4 { margin-bottom: 2rem; }
.mb-5 { margin-bottom: 2.5rem; }

/* Animation Classes */
.fade-in {
    animation: fadeIn 0.5s ease-in;
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

.slide-up {
    animation: slideUp 0.5s ease-out;
}

@keyframes slideUp {
    from { 
        transform: translateY(20px);
        opacity: 0;
    }
    to { 
        transform: translateY(0);
        opacity: 1;
    }
}

/* Responsive Utilities */
@media (max-width: 992px) {
    .container {
        padding: 1rem;
    }
}

@media (max-width: 768px) {
    h1 {
        font-size: 2rem;
    }
    
    h2 {
        font-size: 1.75rem;
    }
    
    h3 {
        font-size: 1.5rem;
    }
    
    .card {
        padding: 1.25rem;
    }
}

@media (max-width: 576px) {
    h1 {
        font-size: 1.75rem;
    }
    
    h2 {
        font-size: 1.5rem;
    }
    
    h3 {
        font-size: 1.25rem;
    }
    
    .btn {
        display: block;
        width: 100%;
        margin-right: 0;
    }
}
EOF

# Copiar la imagen del logo
cp attached_assets/image_final_transparent.png static/images/

echo "Compilación rápida completada con éxito."
echo "Puedes iniciar el servidor Flask para ver los resultados."