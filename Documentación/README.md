# CreditGuard - Machine Learning Credit Risk Dashboard

Este repositorio contiene el desarrollo del **Proyecto 1** de la asignatura **Organización de Lenguajes y Compiladores 2** (Escuela de Vacaciones, Junio 2026). El sistema es una aplicación web integral y autónoma enfocada en la analítica predictiva del riesgo financiero y de incumplimiento crediticio, utilizando técnicas avanzadas de Inteligencia Artificial y Machine Learning sin dependencias de APIs externas.

---

## 🛠️ Arquitectura de Software y Decisiones de Diseño

El sistema implementa una arquitectura desacoplada basada en el patrón de comunicación Cliente-Servidor mediante una API REST:

* **Backend (Python + Flask):** Encargado de la persistencia de datos en memoria local, procesamiento estadístico, entrenamiento secuencial del clasificador de Machine Learning y exposición de los endpoints asíncronos.
* **Frontend (HTML5 + CSS3 + JavaScript ES6):** Diseñado con un enfoque Single Page Application (SPA). Consume la API del backend mediante peticiones asíncronas (`fetch`), gestiona el estado de las vistas dinámicamente y renderiza los componentes visuales e interactivos requeridos por la rúbrica.

---

## 📊 Proceso de Limpieza y Transformación de Datos (Pandas)

Para garantizar la estabilidad matemática del modelo predictivo y evitar sesgos de sobreajuste o valores atípicos corruptos, la data masiva cargada (en formatos `.csv` o `.txt`) es sometida a un pipeline estricto de saneamiento en el Backend utilizando la biblioteca **Pandas**:

1.  **Detección e Imputación de Valores Nulos:** Los campos vacíos (`NaN`) dentro de las columnas numéricas son detectados y sustituidos automáticamente utilizando la **mediana** de dicha variable, asegurando la consistencia dimensional sin alterar la distribución de la muestra original.
2.  **Eliminación de Duplicidad:** Se realiza un escaneo de registros idénticos a nivel de filas, aplicando `.drop_duplicates()` para evitar redundancia de pesos durante el entrenamiento.
3.  **Sanitización de Inconsistencias Lógicas:** En variables financieras de carácter estrictamente no negativo (`ingresos_mensuales`, `deuda_activa`, `monto_solicitado`), cualquier registro de valor menor a cero ($< 0$) es reemplazado por la mediana estadística calculada de la columna.
4.  **Acotamiento de Rangos Porcentuales:** Las variables que representan proporciones o tasas (`historial_pagos`, `utilizacion_credito`) son normalizadas mediante la función `.clip(0, 100)` para restringir matemáticamente los valores al rango porcentual válido de 0 a 100.

---

## 🤖 Selección del Modelo y Justificación Técnica

El modelo algorítmico seleccionado para la toma de decisiones predictivas es el **Random Forest Classifier** (Bosques Aleatorios) provisto por la biblioteca especializada **Scikit-Learn**:

* **Justificación:** Al tratarse de un problema de clasificación binaria (determinar si un aplicante está `En Riesgo [1]` o `Sin Riesgo [0]`), los árboles de decisión ofrecen una excelente capacidad para capturar interacciones no lineales entre variables financieras. Al combinar múltiples árboles mediante técnicas de *bagging*, Random Forest reduce drásticamente la varianza y mitiga el riesgo de *overfitting* en comparación con un único árbol de decisión.
* **División del Dataset:** El conjunto de datos saneado es fragmentado de forma metodológica bajo la proporción clásica de **80% para el conjunto de entrenamiento** (*Train Set*) y **20% para el conjunto de validación y prueba** (*Test Set*), garantizando una evaluación objetiva sobre datos no vistos previamente.
* **Configuración Base:** * `n_estimators` = 100 (Cantidad de árboles robustos en paralelo)
    * `max_depth` = 10 (Nivel máximo de profundidad para mitigar sobreajuste)
    * `max_leaf_nodes` = 50 (Número máximo de nodos hoja finales)

---

## 📈 Evaluación del Rendimiento e Hiperparámetros

El sistema expone de forma obligatoria las métricas de evaluación de rendimiento calculadas estrictamente sobre el set de pruebas del 20%:
* **Exactitud (Accuracy):** Proporción de clasificaciones correctas totales.
* **Precisión (Precision):** Capacidad del modelo para no clasificar como positivo un caso negativo.
* **Exhaustividad (Recall):** Eficacia del modelo para detectar todos los casos reales de riesgo de impago.
* **F1-Score:** Media armónica que balancea la precisión y el recall.
* **Matriz de Confusión:** Desglose en cuadrantes reales de Verdaderos Negativos (TN), Falsos Positivos (FP), Falsos Negativos (FN) y Verdaderos Positivos (TP) inyectados dinámicamente en el DOM.

Adicionalmente, el módulo de **Ajuste de Hiperparámetros** permite al usuario alterar la estructura del bosque desde el frontend. La aplicación recalcula las métricas en tiempo real mediante una petición `POST` hacia la API del backend y genera tablas comparativas matemáticas que muestran la variación porcentual incremental o decremental del rendimiento analítico.