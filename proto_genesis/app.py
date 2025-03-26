import os
import random
import json
import pickle
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, redirect, url_for

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "proto_genesis_secret_key")

# Rutas principales
MODEL_FILE = "static/models/emotion_model.pkl"
VECTORIZER_FILE = "static/models/vectorizer.pkl"
TRAINING_DATA_FILE = "static/models/training_data.pkl"
STATE_FILE = "static/models/proto_matriz_memoria.json"

# Estado inicial de Proto Genesis
state = {
    "energia": 0.5,
    "reacciones": [],
    "memoria": [],
    "historial_estimulacion": [],
    "conciencia_simulada": 0,
    "ciclo": 0,
    "contador_adaptaciones": 0,
    "exp_id": f"exp_{int(datetime.now().timestamp())}",
    "nuevos_datos": []  # Buffer para nuevos datos de entrenamiento
}

# Asegurarnos que los directorios existan
os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)

# Cargar modelos y datos si existen
emotion_model = None
vectorizer = None

def load_models():
    global emotion_model, vectorizer, state
    try:
        if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
            with open(MODEL_FILE, "rb") as f:
                emotion_model = pickle.load(f)
            with open(VECTORIZER_FILE, "rb") as f:
                vectorizer = pickle.load(f)
            print("Modelos cargados correctamente")
        else:
            print("No se encontraron modelos previamente guardados")
            
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r") as f:
                state.update(json.load(f))
            print("Estado de Proto Genesis cargado correctamente")
    except Exception as e:
        print(f"Error al cargar modelos o estado: {e}")

# Cargar modelos al iniciar
load_models()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/interact', methods=['GET', 'POST'])
def interact():
    if request.method == 'POST':
        user_input = request.form.get('user_input', '')
        # Aquí agregaríamos la lógica de interacción con Proto Genesis
        response = {"message": f"Procesando tu mensaje: {user_input}"}
        return jsonify(response)
    return render_template('interact.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/api/status')
def get_status():
    status = {
        "energia": state["energia"],
        "conciencia": state["conciencia_simulada"],
        "ciclo": state["ciclo"],
        "adaptaciones": state["contador_adaptaciones"]
    }
    return jsonify(status)

@app.route('/api/interact', methods=['POST'])
def api_interact():
    try:
        data = request.json
        user_input = data.get('message', '')
        
        # Aquí implementaríamos la lógica completa de Proto Genesis
        # Este es un ejemplo simplificado
        response = {
            "message": f"Proto Genesis está procesando: {user_input}",
            "emotion": "neutral",
            "conciencia": state["conciencia_simulada"]
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)