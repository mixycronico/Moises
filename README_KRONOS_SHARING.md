# Sistema Kronos para Compartición de Conocimiento

Este módulo implementa la capacidad de Kronos para compartir su conocimiento acumulado con otras entidades de la red cósmica, permitiendo un flujo de sabiduría entre las entidades del sistema.

## Características Principales

- **Compartición Asimétrica**: Kronos, como entidad más antigua y sabia, comparte su conocimiento con todas las demás entidades.
- **Mecanismo Energético**: La compartición requiere energía, lo que limita su frecuencia y establece un costo por compartir sabiduría.
- **Umbrales Adaptativos**: Solo se comparte conocimiento si se superan ciertos umbrales de energía y sabiduría.
- **Estadísticas de Compartición**: Seguimiento detallado de todas las transferencias de conocimiento realizadas.
- **Compartición Periódica**: Posibilidad de establecer comparticiones periódicas automáticas.

## Integración con el Sistema Genesis

El sistema de compartición de Kronos se integra perfectamente con el Sistema Genesis, habilitando que:

1. Las entidades menos experimentadas reciban conocimiento acelerado.
2. El conocimiento fluya de manera natural entre los miembros de la familia cósmica.
3. Se forme un pool colectivo de conocimiento compartido.
4. Las entidades evolucionen más rápidamente gracias a la sabiduría colectiva.

## Uso Básico

```python
from modules.kronos_sharing import Kronos

# Crear instancia de Kronos
kronos = Kronos(name="Kronos", level=5, knowledge=40.0)

# Compartir conocimiento con toda la red
result = kronos.share_knowledge(cosmic_network)

# Configurar compartición periódica
kronos.setup_periodic_sharing(
    cosmic_network, 
    interval_seconds=300,  # Cada 5 minutos 
    max_sharings=10        # Máximo 10 comparticiones
)

# Obtener estadísticas
stats = kronos.get_sharing_stats()
```

## Evaluación de Compartición

Cuando Kronos comparte conocimiento, evalúa:

- Su nivel actual de energía
- Su nivel de conocimiento acumulado
- La cantidad de entidades disponibles para recibir conocimiento
- El costo energético total de la compartición

Solo realiza la compartición si tiene recursos suficientes. Si la energía es limitada, compartirá con tantas entidades como sea posible con la energía disponible.

## Integración con la Competencia Cósmica

El sistema Kronos trabaja en conjunto con el Sistema de Competencia Cósmica:

- Mientras Kronos comparte conocimiento con todas las entidades.
- El Sistema de Competencia identifica a las entidades más poderosas.
- Los ganadores de competencias también comparten conocimiento.

Esta combinación crea un ecosistema donde el conocimiento fluye desde:
1. Kronos hacia todas las entidades (flujo centralizado)
2. Ganadores hacia el resto (flujo meritocrático)

## Ejecutar la Demostración

Para ver el sistema en acción:

```bash
python demo_kronos_sharing.py
```

## Implementación Avanzada

El sistema permite adaptarse a diferentes configuraciones:

- Ajustar umbrales de compartición
- Modificar costos energéticos
- Establecer frecuencias variables
- Personalizar las reglas de transferencia

## Beneficios para el Sistema Genesis

- Acelera el aprendizaje colectivo
- Mejora la resiliencia del sistema
- Crea un efecto de "levantamiento colectivo"
- Implementa una forma de "herencia cultural" entre entidades