# Predicción de riesgo de ataque al corazón

App de Streamlit que despliega un modelo **Random Forest** de clasificación:
captura los datos del paciente y devuelve la predicción (`Yes` / `No`) con su
probabilidad.

## Requisitos

- Python 3.10+ (probado en 3.13)
- El archivo `modelo-cla.pkl` en la raíz del proyecto

## Instalación

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt   # macOS / Linux
```

> `scikit-learn` está fijado a `1.6.1` porque es la versión con la que se
> entrenó el pickle. Con otra versión sklearn avisa que los resultados pueden
> ser inválidos.

## Ejecutar

```bash
.venv/Scripts/streamlit run despliegue_validacioncruz_clasif.py   # Windows
# streamlit run despliegue_validacioncruz_clasif.py               # con el venv activado
```

Abre [http://localhost:8501](http://localhost:8501). La predicción se recalcula en vivo al mover
cualquier control; no hay botón que pulsar.

Para parar el servidor: `Ctrl+C` en la terminal.

Otro puerto: `streamlit run despliegue_validacioncruz_clasif.py --server.port 8502`

## Estructura

| Archivo                                 | Qué es                                                    |
| --------------------------------------- | ---------------------------------------------------------- |
| `despliegue_validacioncruz_clasif.py` | App completa: carga, UI, preparación y predicción        |
| `modelo-cla.pkl`                      | Tupla`(modelo, labelencoder, variables, min_max_scaler)` |
| `requirements.txt`                    | Dependencias                                               |
| `.streamlit/config.toml`              | Tema (colores) y toolbar en modo mínimo                   |

## Cómo predice

1. Los 6 campos del formulario arman un `DataFrame` con los nombres originales.
2. `pd.get_dummies(..., drop_first=False)` sobre las categóricas.
3. `reindex(columns=variables, fill_value=0)` para alinear con el entrenamiento.
4. `modelo.predict(...)` + `labelencoder.inverse_transform(...)` → `Yes` / `No`.

No se normaliza: el modelo final es Random Forest (árboles), así que
`min_max_scaler` se carga pero no se aplica.

## Notas

- El modelo tiene un error del 8% (MAPE). Es un ejercicio académico, no un
  diagnóstico médico.
