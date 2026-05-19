# Reto 6 — Análisis de Sentimiento de Reseñas

**Unidad 4 · Inteligencia Artificial Moderna · Universidad de Boyacá · 2026**

Clasificador de sentimiento de reseñas en español (1–5 estrellas) construido mediante *fine-tuning* de **BETO** (BERT en español) sobre el dataset Amazon Reviews Multi.

---

## Equipo

- Camilo Andres Ortiz Gonzalez
- Omar Elian Martinez Perez

## Reto asignado

Reto **6 — Análisis de Sentimiento de Reseñas** (caso de uso: Amazon Reviews / IMDB Movie Reviews).

- **Dataset utilizado:** `mteb/amazon_reviews_multi`, subset `es` (español).
- **Tipo de problema:** clasificación multiclase 1–5 estrellas.
- **Arquitectura:** Transformer pre-entrenado con fine-tuning — `dccuchile/bert-base-spanish-wwm-cased` (BETO).

---

## Estructura del repositorio

```
.
├── reto6_sentiment_analysis.ipynb   # Notebook con todo el pipeline ejecutado
├── slides_reto6.pdf                 # Diapositivas de la presentación
└── README.md
```

> **Nota sobre el modelo entrenado:** `beto_amazon_es.zip` pesa ~440 MB y excede el límite de 100 MB por archivo de GitHub. Está alojado en Google Drive y se descarga desde el siguiente enlace:
>
> 🔗 **[Descargar modelo entrenado (Google Drive)](https://drive.google.com/drive/folders/1ebedufVu_d6564Tv_J7plo_o42M5gXjG?usp=drive_link)**

---

## Pipeline implementado

El notebook sigue las 7 fases exigidas por el reto:

1. **Ingesta** — carga del dataset desde Hugging Face Hub.
2. **EDA** — 3 visualizaciones: distribución de clases, longitud de reseñas, palabras frecuentes por clase.
3. **Limpieza** — nulos, duplicados, normalización ligera, manejo de URLs y emails.
4. **Preprocesamiento** — tokenización WordPiece (`max_length=128`), split estratificado train/val/test.
5. **Modelado** — fine-tuning de BETO con cabezal de 5 clases.
6. **Evaluación** — métricas, curvas de aprendizaje, matriz de confusión, comparación con baseline TF-IDF y análisis de errores.
7. **Demostración** — inferencia en vivo sobre reseñas nuevas no presentes en el dataset.

---

## Cómo reproducir el experimento

### Opción A — Google Colab (recomendado)

1. Subir `reto6_sentiment_analysis.ipynb` a Google Colab.
2. **Entorno de ejecución → Cambiar tipo de entorno → GPU (T4)**.
3. **Entorno de ejecución → Ejecutar todas**.
4. Tiempo estimado: **15–25 minutos** (la fase de fine-tuning es la más costosa).

### Opción B — Local con GPU

Requisitos:
- Python 3.10+
- GPU NVIDIA con ≥6 GB VRAM (CUDA 11.8 o 12.x)
- ~5 GB libres en disco

```bash
pip install transformers==4.44.2 datasets==2.21.0 accelerate==0.34.2 \
            evaluate==0.4.3 scikit-learn==1.5.2 matplotlib seaborn jupyter

jupyter notebook reto6_sentiment_analysis.ipynb
```

### Reproducibilidad

Todas las fuentes de aleatoriedad están fijadas con `seed=42` (numpy, random, torch, splits del dataset, semilla del Trainer).

---

## Uso del modelo entrenado (sin re-entrenar)

Una vez descargado y descomprimido `beto_amazon_es.zip`:

```python
from transformers import pipeline

clasificador = pipeline(
    "text-classification",
    model="./beto_amazon_es",
    tokenizer="./beto_amazon_es",
)

resena = "La licuadora hace mucho ruido y la cuchilla se atascó al segundo uso"
print(clasificador(resena))
# → [{'label': '2★', 'score': 0.78}]
```

---

## Resultados finales obtenidos

### Métricas globales sobre TEST (2.000 reseñas nunca vistas)

| Modelo                       | Accuracy | F1 macro |
|------------------------------|---------:|---------:|
| Azar (5 clases balanceadas)  |    0,200 |    0,200 |
| Mayoría (clase balanceada)   |    0,200 |    0,067 |
| TF-IDF + Logistic Regression |   0,5215 |   0,5175 |
| **BETO fine-tuned**          | **0,5505** | **0,5544** |

### Métricas por clase (BETO sobre TEST)

| Clase | Precision | Recall | F1     |
|------:|----------:|-------:|-------:|
| 1★    |    0,7203 | 0,6375 | 0,6764 |
| 2★    |    0,4683 | 0,5175 | 0,4917 |
| 3★    |    0,4444 | 0,4700 | 0,4569 |
| 4★    |    0,4846 | 0,5125 | 0,4982 |
| 5★    |    0,6872 | 0,6150 | 0,6491 |

### Distribución de la distancia de error

- distancia = 0 (correcto): **1.101 (55,0 %)**
- distancia = 1 (clase vecina): 797 (39,9 %)
- distancia = 2: 94 (4,7 %)
- distancia = 3: 6 (0,3 %)
- distancia = 4: 2 (0,1 %)

**El 94,9 % de las predicciones cae en la clase correcta o la adyacente.** Solo el 0,4 % de los errores se desvía más de 2 estrellas.

---

## Decisiones técnicas clave

- **¿Por qué Transformer?** El texto natural tiene dependencias largas y matices (negación, ironía) que las arquitecturas CNN/LSTM capturan peor. Los Transformers resuelven esto con *self-attention*.
- **¿Por qué BETO y no mBERT?** BETO está entrenado únicamente en español, con vocabulario optimizado para el idioma; obtiene mejor rendimiento documentado en tareas en español que multilingual-BERT.
- **¿Por qué fine-tuning?** Entrenar un Transformer desde cero requiere terabytes de texto y semanas de cómputo. El fine-tuning aprovecha las representaciones lingüísticas ya aprendidas por BETO y solo ajusta los pesos para la tarea específica.
- **¿Por qué 5 clases y no clasificación binaria?** Mayor utilidad para el negocio: permite priorizar atención (1★ urgente, 3★ feedback, 4–5★ celebrar) en lugar de una simple dicotomía positivo/negativo.
- **¿Por qué 20.000 muestras y no 200.000?** Restricciones de la GPU gratuita en Colab. Escalar a 200.000 muestras mejora aproximadamente 3–5 puntos de F1 sin cambiar la arquitectura.

---

## Limitaciones conocidas

- **Dominio:** entrenado con reseñas de Amazon España; el léxico LATAM ("chévere", "bacano", "padrísimo") tiene cobertura desigual.
- **Sarcasmo e ironía:** sigue siendo un problema abierto en NLP.
- **Truncamiento:** reseñas mayores a 128 tokens pierden la cola, que a veces contiene la conclusión.
- **Submuestreo:** se usaron 20.000 de 200.000 muestras disponibles.

---

## Entregables del reto (checklist)

- [x] Notebook (`.ipynb`) con pipeline ejecutado y celdas Markdown explicativas.
- [x] Archivo del modelo entrenado (`beto_amazon_es.zip`, enlazado vía Google Drive).
- [x] Preprocesadores serializados — el tokenizer se guarda dentro del zip del modelo.
- [x] Slides de la presentación (`slides_reto6.pdf`).
- [x] README.md con nombres del equipo, reto e instrucciones de reproducción.

---

## Créditos

- **Dataset:** [mteb/amazon_reviews_multi](https://huggingface.co/datasets/mteb/amazon_reviews_multi) (Amazon, 2015–2019).
- **Modelo base:** [BETO](https://github.com/dccuchile/beto) — Cañete et al., 2020.
- **Stack:** [🤗 Transformers](https://huggingface.co/docs/transformers), PyTorch, scikit-learn, Hugging Face Datasets.
