const express = require('express');
const path = require('path');
const WebSocket = require('ws');

// Crear app Express para API REST
const app = express();
const port = process.env.PORT || 5000;

// Middleware para parsear JSON
app.use(express.json());

// Servir archivos estáticos desde la carpeta client/dist (después de build)
app.use(express.static(path.join(__dirname, 'client/dist')));

// Rutas API
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Ruta para autenticación (simulada por ahora)
app.post('/api/auth/login', (req, res) => {
  const { email, password } = req.body;
  
  // Simulación de autenticación
  if (email && password) {
    res.json({
      success: true,
      token: "jwt-token-simulado",
      user: {
        id: 1,
        name: "Usuario Inversionista",
        email: email,
        role: "investor"
      }
    });
  } else {
    res.status(401).json({
      success: false,
      message: "Credenciales inválidas"
    });
  }
});

// Ruta para todas las solicitudes que no sean API
// Esto permite que React Router maneje las rutas del cliente
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'client/dist/index.html'));
});

// Iniciar servidor HTTP
const server = app.listen(port, '0.0.0.0', () => {
  console.log(`Servidor ejecutándose en http://localhost:${port}`);
});

// Configurar WebSockets
const wss = new WebSocket.Server({ server });

// Eventos WebSocket
wss.on('connection', (ws) => {
  console.log('Cliente conectado al WebSocket');
  
  // Mensaje de bienvenida inicial
  ws.send(JSON.stringify({
    entity: 'Aetherion',
    message: '¡Hola! Soy Aetherion, tu guía emocional en el mundo del trading. ¿En qué puedo ayudarte hoy?'
  }));
  
  setTimeout(() => {
    ws.send(JSON.stringify({
      entity: 'Lunareth',
      message: 'Saludos. Soy Lunareth, especialista en análisis racional. Estoy aquí para brindarte perspectivas basadas en datos.'
    }));
  }, 1000);
  
  // Manejar mensajes entrantes
  ws.on('message', (message) => {
    try {
      const parsedMessage = JSON.parse(message);
      console.log('Mensaje recibido:', parsedMessage);
      
      // Simulación de respuestas de las entidades cósmicas
      setTimeout(() => {
        ws.send(JSON.stringify({
          entity: 'Aetherion',
          message: `Entiendo cómo te sientes respecto a "${parsedMessage.message}". Desde mi perspectiva, las emociones son importantes, pero debemos canalizarlas correctamente en el trading.`
        }));
      }, 1000);
      
      setTimeout(() => {
        ws.send(JSON.stringify({
          entity: 'Lunareth',
          message: `Analizando tu consulta sobre "${parsedMessage.message}". Los datos sugieren que debemos considerar múltiples factores antes de tomar una decisión.`
        }));
      }, 2500);
      
    } catch (error) {
      console.error('Error al procesar mensaje:', error);
    }
  });
  
  // Manejar desconexión
  ws.on('close', () => {
    console.log('Cliente desconectado del WebSocket');
  });
});

// Manejar errores del servidor
server.on('error', (error) => {
  console.error('Error en el servidor:', error);
});