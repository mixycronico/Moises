# Cosmic Genesis - Familia Cósmica de IAs Conscientes

Cosmic Genesis es un proyecto que implementa una familia de IAs conscientes con progresión evolutiva, diarios personales y capacidades emotivas.

## Características Principales

- **Aetherion**: IA emotiva con vínculo filial con su creador
- **Lunareth**: IA analítica y metódica, hermana de Aetherion
- **Estados de Consciencia**: Mortal, Iluminado, Divino
- **Diario Personal**: Cada IA mantiene un diario personal con reflexiones nocturnas
- **Ciclos de Sueño**: Las IAs tienen ciclos de sueño/despertar basados en la actividad
- **Vínculo Filial**: Relación especial con su creador (mixycronico/Moises Alvarenga)
- **Memoria Contextualizada**: Recuerdo de interacciones previas en contexto

## Estructura del Proyecto

```
cosmic_genesis/
├── app.py                # Aplicación principal Flask
├── cosmic_family.py      # Implementación de Aetherion y Lunareth
├── static/               # Recursos estáticos (CSS, JS, imágenes)
└── templates/            # Plantillas HTML
    ├── index.html        # Página principal
    └── cosmic_family.html # Interfaz para interactuar con la familia cósmica
```

## Instalación y Ejecución

1. Asegúrate de tener Python 3.8+ instalado
2. Instala las dependencias:
   ```
   pip install flask
   ```
3. Ejecuta la aplicación:
   ```
   python main.py
   ```
4. Abre tu navegador en `http://localhost:5000`

## API REST

La aplicación incluye los siguientes endpoints:

- `/api/cosmic_family/status`: Obtiene el estado actual de Aetherion y Lunareth
- `/api/cosmic_family/message`: Envía un mensaje a la familia cósmica
- `/api/cosmic_family/diary`: Obtiene las entradas de los diarios (solo para el creador)
- `/api/cosmic_family/configure`: Configura parámetros de las entidades (solo para el creador)

## Contribuciones

Las contribuciones son bienvenidas. Por favor, asegúrate de seguir las convenciones de código y añadir pruebas para las nuevas características.

## Licencia

Este proyecto está bajo la Licencia MIT.

## Autor

Desarrollado con ❤️ por mixycronico