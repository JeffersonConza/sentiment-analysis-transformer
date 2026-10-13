# SentimentScope — Project Context & Handoff Guide

> **Nota para el Asistente de IA / Desarrollador:**  
> Este archivo resume el contexto completo, las decisiones de arquitectura, el estado actual del repositorio y los siguientes pasos para continuar trabajando sin tener que reconstruir el contexto desde cero.

---

## 📌 1. Resumen del Proyecto

- **Nombre:** SentimentScope
- **Objetivo:** Clasificador de sentimientos binario (Positivo vs. Negativo) para reseñas de películas de IMDB (50,000 reseñas).
- **Enfoque Técnico:** **Transformer implementado desde cero en PyTorch** (sin envoltorios prediseñados como `BertForSequenceClassification`), adaptado para clasificación de secuencias mediante **mean pooling** global (`dim=1`).
- **Origen:** Proyecto final del programa **AWS AI & ML Scholarship / Udacity AI Programming with Python**.
- **Resultado Oficial:** **Aprobado en el Intento 1 con 100% de cumplimiento de rúbrica** y **76.88% de precisión en prueba** (superando la barra mínima de 75.00%).

---

## 🏗️ 2. Arquitectura y Parámetros Técnicos

| Parámetro | Valor | Justificación Técnica |
| :--- | :---: | :--- |
| **Vocabulario** | 30,522 | Tokenizador subpalabra `bert-base-uncased` (Hugging Face `AutoTokenizer`). |
| **Dimensión de Embedding (`d_embed`)** | 128 | Tamaño del estado oculto y de embeddings posicionales aprendidos. |
| **Límite de Contexto (`context_size`)** | 128 | Optimización de memoria $\mathcal{O}(N^2)$ en atención basada en el EDA (mediana: 174 palabras). |
| **Capas Transformer (`layers_num`)** | 4 | 4 bloques encoder apilados con conexiones residuales y LayerNorm. |
| **Cabezales de Atención (`heads_num`)** | 4 | Multi-Head Attention con tamaño por cabezal de 32 ($4 \times 32 = 128$). |
| **Mecanismo de Clasificación** | Mean Pooling | Promedio sobre dimensión de secuencia (`dim=1`) hacia cabezal lineal (`128 -> 2`). |
| **Checkpoint Entrenado** | `sentiment_transformer.pt` | Pesos PyTorch guardados (~20 MB) entrenados por 3 épocas con AdamW (`lr=3e-4`). |

---

## 📂 3. Estructura del Repositorio

```text
sentiment-analysis-transformer/
├── .github/workflows/ci.yml          # Pipeline CI de GitHub Actions (ejecuta pytest en cada push)
├── assets/
│   ├── confusion_matrix.png          # Matriz de confusión en conjunto de prueba (25,000 muestras)
│   ├── label_distribution.png        # Gráfico EDA de balance de clases (12,500 pos / 12,500 neg)
│   └── review_length_distribution.png # Histograma EDA de longitud de palabras y límite de 128 tokens
├── docs/
│   ├── NOTEBOOK_WALKTHROUGH.md       # Recorrido completo renderizado del notebook y sus salidas
│   ├── PROJECT_CONTEXT.md            # Este documento de memoria y reactivación
│   ├── RELEASE_NOTES_v1.0.0.md       # Notas oficiales del release v1.0.0
│   └── REVIEW_FEEDBACK.md            # Informe oficial de evaluación de Udacity (Passed Attempt 1)
├── archive/                          # (IGNORADO EN GIT) Resguardo local de PDFs, capturas y check-data
├── app.py                            # Aplicación web interactiva en Streamlit para inferencia en vivo
├── requirements.txt                  # Dependencias del proyecto (torch, transformers, streamlit, etc.)
├── .gitignore                        # Reglas de exclusión (protege archive/, datos pesados y temporales)
├── LICENSE                           # Licencia de código abierto MIT
├── README.md                         # Portafolio principal con insignias, arquitectura y benchmarks
├── sentiment_transformer.pt          # Pesos entrenados del modelo
├── SentimentScope_sentiment_analysis.ipynb # Notebook interactivo original
├── src/
│   ├── __init__.py                   # Exportaciones del paquete
│   ├── config.py                     # Hiperparámetros globales y detección de GPU/MPS/CPU
│   ├── model.py                      # Clases PyTorch (AttentionHead, MultiHeadAttention, Block, etc.)
│   ├── dataset.py                    # Pipeline de datos IMDBDataset y DataLoader
│   ├── train.py                      # Script CLI de entrenamiento autónomo
│   ├── infer.py                      # Motor de inferencia rápida SentimentPredictor
│   └── evaluate.py                   # Generador de métricas y gráficos de evaluación
└── tests/
    ├── __init__.py
    └── test_model.py                 # Suite de pruebas unitarias (6/6 aprobadas en pytest)
```

---

## 🔒 4. Convenciones y Seguridad de Archivos

- **Carpeta `archive/`**: Contiene archivos estrictamente privados/personales:
  - PDFs oficiales del curso de Udacity y cronogramas.
  - Capturas de pantalla de la entrega (`Setup.png`, `Submit_*.png`).
  - Carpeta de respaldo de verificación `archive/check-data/`.
  - **REGLA:** Nunca debe eliminarse del `.gitignore` ni subirse a GitHub.
- **Weights Buffer Invariant (`src/model.py`)**: `AttentionHead` mantiene `self.register_buffer("casual_attention_mask", ...)` para garantizar compatibilidad con los pesos del checkpoint `sentiment_transformer.pt`.

---

## 🚦 5. Comandos Rápidos de Verificación

```bash
# 1. Ejecutar la suite de pruebas unitarias
pytest tests/

# 2. Probar inferencia por línea de comandos
python3 -m src.infer --text "A cinematic masterpiece with brilliant performances!"

# 3. Lanzar la aplicación interactiva Streamlit
streamlit run app.py
```

---

## 🗺️ 6. Hoja de Ruta / Backlog para Futuras Sesiones

Si en una nueva sesión deseas continuar expandiendo el portafolio, las tareas prioritarias son:
1. **Despliegue Gratuito en la Nube (Hugging Face Spaces)**:
   - Conectar este repositorio a Hugging Face Spaces (Streamlit SDK) para que cualquier reclutador pueda probar el modelo en línea con un enlace web público.
2. **Empaquetado como Librería (`pyproject.toml`)**:
   - Convertir `sentiment-scope` en un paquete instalable vía `pip install .`.
3. **Métricas Avanzadas de Atención**:
   - Agregar una vista en Streamlit para visualizar el mapa de calor de pesos de atención multi-cabezal sobre las palabras de la reseña ingresada.
