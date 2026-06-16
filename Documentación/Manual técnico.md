# MANUAL TÉCNICO: CREDITGUARD
**Sistema Inteligente para la Evaluación de Riesgo Crediticio basado en Aprendizaje Supervisado**

---

## 1. Especificaciones del Entorno Tecnológico
Para garantizar la reproducibilidad y estabilidad del sistema en el entorno de evaluación, se definió una pila tecnológica homogénea basada en las últimas revisiones estables de las librerías de ciencia de datos en Python:

* **Entorno de Ejecución:** Python 3.13.x (Arquitectura de precompilación CPython).
* **Servidor de Aplicaciones y Enrutamiento:** Flask v3.0+ (Micro-framework WSGI para la exposición de servicios REST).
* **Manejo de CORS:** Flask-CORS v4.0+ (Mecanismo de seguridad para permitir peticiones cruzadas desde el frontend).
* **Procesamiento Matricial y Manipulación de Datos:** Pandas v2.2+ (Estructuras de datos DataFrame para operaciones vectorizadas de limpieza).
* **Núcleo de Machine Learning y Modelado:** Scikit-Learn v1.5+ (Provisión de algoritmos de ensamble, división de conjuntos de datos y cálculo de matrices de evaluación).
* **Serialización de Objetos:** Joblib (Librería nativa de Python para el almacenamiento persistente del estado del modelo).

---

## 2. Arquitectura del Sistema y Descripción de Módulos
La solución tecnológica se diseñó bajo una arquitectura desacoplada y modular en el backend, aislando las responsabilidades operacionales de la ingesta de datos, el modelado matemático y la exposición de servicios. El sistema se divide en tres componentes de software principales:

* **`data_processing.py` (Capa de Preprocesamiento e Ingesta):** Módulo encargado de la lectura de fuentes de datos en formato de valores separados por comas (CSV). Centraliza las reglas de negocio financiero y las transformaciones matemáticas requeridas para homogeneizar la información, encargándose de la imputación de valores faltantes, la deduplicación y la eliminación de anomalías operacionales.
* **`model.py` (Capa de Machine Learning):** Componente central que encapsula el ciclo de vida del modelo predictivo. Utiliza la biblioteca `scikit-learn` para estructurar el pipeline de entrenamiento del clasificador, calcular de manera automatizada las matrices de confusión y las métricas de rendimiento, y exponer los métodos de inferencia probabilística para la evaluación de perfiles individuales.
* **`app.py` (Capa de Servicios y API REST):** Servidor web construido sobre el micro-framework `Flask`. Actúa como la fachada del sistema y expone endpoints seguros (`/api/upload`, `/api/train`, `/api/predict`) que permiten al frontend consumir de manera asíncrona y transparente las capacidades analíticas de la aplicación.

---

## 3. Diccionario de Datos Técnico (Esquema del CSV)
El sistema implementa una validación estricta a nivel de backend mediante la constante `COLUMNAS_REQUERIDAS`. Cualquier archivo de carga masiva debe contener con exactitud las siguientes dimensiones estructurales:

| Nombre de la Columna | Tipo de Dato (Pandas) | Dominio Lógico / Descripción |
| :--- | :--- | :--- |
| `ingresos_mensuales` | `float64` | Ingresos netos mensuales del solicitante expresados en Quetzales (Q). Debe ser $> 0$. |
| `deuda_activa` | `float64` | Monto total de obligaciones financieras vigentes en el sistema bancario (Q). |
| `historial_pagos` | `float64` | Métrica de comportamiento crediticio previo calibrada en el intervalo cerrado `[0, 100]`. |
| `antiguedad_laboral` | `float64` | Tiempo transcurrido en el empleo actual medido de forma continua en años. |
| `creditos_activos` | `float64` | Número discreto de operaciones crediticias abiertas de forma simultánea. |
| `monto_solicitado` | `float64` | Capital total que el cliente solicita para la apertura del nuevo préstamo (Q). |
| `atrasos_previos` | `float64` | Cantidad de cuotas vencidas e impagadas acumuladas en los últimos 12 meses. |
| `dependientes_economicos`| `float64` | Número de personas que dependen directamente de los ingresos del solicitante. |
| `utilizacion_credito` | `float64` | Porcentaje de saldo utilizado respecto al límite total rotativo disponible `[0, 100]`. |
| `en_riesgo` | `int64` | **Variable Objetivo (Label):** Clasificación binaria (`0` = Solvente, `1` = En Riesgo). |

---

## 4. Análisis Exploratorio de Datos (EDA)
Antes de someter los datos al proceso de modelado, se realizó un Análisis Exploratorio de Datos sobre el conjunto de entrenamiento para auditar su calidad estructural e identificar patrones estadísticos latentes:

### A. Distribución de Variables
El dataset provisto consta de **5,035 observaciones históricas** y **10 dimensiones numéricas**. Las variables se dividen de forma equitativa entre la capacidad de endeudamiento del usuario, su estabilidad laboral y sus antecedentes de pago.

### B. Detección de Outliers e Inconsistencias
El análisis de fronteras evidenció la presencia de ruido y errores de captura severos en los registros crudos, los cuales representaban un riesgo crítico de sesgo para el algoritmo:
* **Valores Financieros Negativos:** Se detectaron ingresos mínimos de `-5,174.92 Q` y deudas activas de `-21,502.03 Q`. Al ser magnitudes de escala absoluta, los valores menores o iguales a cero constituyen inconsistencias lógicas.
* **Desbordamiento de Rangos Porcentuales:** La variable `historial_pagos` registró un mínimo anómalo de `-29.20` y un máximo excedido de `154.30`, mientras que la `utilizacion_credito` alcanzó un pico de `166.50%`, superando el límite teórico del 100%.

### C. Balance de Clases
La evaluación de la variable objetivo (`en_riesgo`) determinó la siguiente distribución fundamental:
* **Clase 0 (Sin Riesgo Detectado):** 2,543 instancias (**50.5%**).
* **Clase 1 (En Riesgo de Incumplimiento):** 2,492 instancias (**49.5%**).

*Justificación Técnica:* Al presentar una proporción cercana al 50/50, el conjunto de datos se clasifica como un **dataset perfectamente balanceado**. Esto elimina la necesidad de aplicar algoritmos de remuestreo sintético (como SMOTE), garantizando que el clasificador aprenda a identificar ambas realidades con el mismo nivel de especificidad y sin sesgos estadísticos intrínsecos.

### D. Correlaciones Observadas
Se calculó la matriz de correlación de Pearson lineal para identificar la fuerza de asociación entre las variables independientes y la variable objetivo `en_riesgo`:
1.  **`atrasos_previos` (+0.418):** Representa la correlación positiva más fuerte. Matemáticamente, demuestra que el número de atrasos registrados en los últimos 12 meses es el predictor directo más crítico para anticipar un impago futurible.
2.  **`historial_pagos` (-0.322):** Representa la correlación negativa más significativa. Esto valida de forma empírica que a mayor puntaje en el comportamiento histórico de pagos, la probabilidad de caer en riesgo crediticio disminuye de manera drástica.

---

## 5. Estrategia de Limpieza de Datos y Justificación Técnica
La fase de saneamiento de datos implementada en `data_processing.py` responde directamente a los hallazgos del EDA mediante estrategias óptimas validadas frente a soluciones alternativas:

* **Tratamiento de Valores Faltantes (Imputación por Mediana):**
    * *Estrategia:* Las variables numéricas con datos ausentes (como `antiguedad_laboral` con 404 nulos e `historial_pagos` con 349) fueron completadas utilizando la mediana calculada de cada columna.
    * *Justificación frente a alternativas:* La eliminación completa de filas con valores nulos (*listwise deletion*) habría reducido el tamaño del dataset en más de un 15%, destruyendo muestras representativas. Se seleccionó la **mediana** en lugar de la media aritmética (promedio) debido a que los datos financieros presentan una distribución asimétrica con colas largas (outliers legítimos de ingresos muy altos). La media es altamente susceptible a ser desplazada por valores extremos, mientras que la mediana es un estimador robusto que preserva el centro de la masa de la distribución original sin introducir distorsiones.
* **Remoción de Registros Redundantes:**
    * *Estrategia:* Se purgaron de forma estricta los 35 registros duplicados detectados en el CSV crudo a través del método `drop_duplicates()`.
    * *Justificación frente a alternativas:* Ignorar la redundancia provoca que el modelo sufra de sobreajuste (*overfitting*), obligando a los árboles de decisión a memorizar patrones idénticos y sesgando las probabilidades de división de nodos hacia perfiles artificialmente repetidos.
* **Sanitización de Inconsistencias Absolutas:**
    * *Estrategia:* Todo registro en `ingresos_mensuales`, `deuda_activa` o `monto_solicitado` cuyo valor fuese menor o igual a cero fue sustituido dinámicamente por la mediana de la variable.
    * *Justificación:* Un ingreso o crédito de Q 0.00 o negativo no posee validez en el negocio financiero real. Al reemplazar la anomalía aislada por la mediana, se rescata el resto de las variables válidas del solicitante en lugar de descartar todo el renglón.
* **Tratamiento de Desbordamientos mediante Acotamiento (Clipping):**
    * *Estrategia:* Las tasas porcentuales se normalizaron de forma estricta al intervalo cerrado `[0, 100]` empleando la función `.clip(0, 100)` de Pandas.
    * *Justificación:* Reasigna los errores de desbordamiento (como un historial de 154.30%) al valor límite lógico más cercano (100%), eliminando el ruido computacional sin alterar radicalmente la varianza de la muestra.

---

## 6. Selección del Modelo de Machine Learning
Para dar respuesta a la problemática de clasificación binaria de CreditGuard, se evaluaron tres enfoques metodológicos bajo el criterio de capacidad de generalización y robustez algorítmica:

1.  **Regresión Logística (Descartado):** Aunque es eficiente, asume una relación lineal estricta entre las variables macroeconómicas y el riesgo, limitando su capacidad para mapear interacciones complejas (por ejemplo, el impacto combinado de tener altos ingresos pero también un alto nivel de endeudamiento simultáneo).
2.  **Árbol de Decisión Individual - CART (Descartado):** Tiende inherentemente a segmentar el espacio de características hasta el infinito, memorizando el ruido del dataset de entrenamiento y sufriendo de una varianza extremadamente alta ante ligeras variaciones en los datos de entrada.
3.  **Bosques Aleatorios - Random Forest Classifier (Seleccionado):**
    * *Justificación Técnica:* Se seleccionó este algoritmo de ensamble basado en el principio de *Bagging* (Bootstrap Aggregating). Random Forest combina múltiples árboles de decisión independientes entrenados en subconjuntos aleatorios de datos y de características. Al promediar las predicciones del bosque, el modelo reduce drásticamente la varianza global sin aumentar el sesgo. Además, es inmune a problemas de escala (no requiere normalización estricta de variables como `Z-score`), maneja de forma óptima las interacciones no lineales entre dimensiones financieras y es altamente robusto frente al ruido de outliers que logren pasar la etapa de limpieza, posicionándose como el modelo ideal para entornos de producción bancaria.

---

## 7. Documentación de la API REST y Persistencia
El backend funciona como un proveedor de servicios desacoplado que se comunica con la interfaz de usuario mediante objetos JSON. A continuación se detallan sus contratos de comunicación y la persistencia del estado:

### A. Catálogo de Endpoints REST
* **`POST /api/upload`:**
    * *Descripción:* Recibe el archivo CSV crudo a través de un flujo de tipo `multipart/form-data`.
    * *Flujo Interno:* Convierte el flujo binario en un DataFrame de Pandas, ejecuta el motor de validación de columnas y procesa la limpieza modularizada. Almacena temporalmente el resultado limpio en una variable global en memoria para optimizar la velocidad de acceso.
    * *Respuesta Exitosa (JSON):* `{"success": true, "registros": 5035, "nulos": 1455, "duplicados": 35}`
* **`POST /api/train`:**
    * *Descripción:* Orquesta la fase de entrenamiento y calibración del modelo matemático.
    * *Flujo Interno:* Extrae del cuerpo de la petición (JSON) los hiperparámetros de control (`n_estimators`, `max_depth`, `max_leaf_nodes`). Si la petición viene vacía, el parámetro `silent=True` intercepta la llamada para evitar un fallo por tipo de medio no soportado (Error 415) y asigna los valores por defecto del proyecto. Al finalizar, calcula las métricas y sobrescribe el archivo físico del modelo.
    * *Respuesta Exitosa (JSON):* Contiene las métricas de validación (`accuracy`, `precision`, `recall`, `f1_score`), la matriz de confusión desglosada y el porcentaje de distribución de las predicciones resultantes.
* **`POST /api/predict`:**
    * *Descripción:* Resuelve la inferencia en tiempo real para un cliente analizado de forma individual.
    * *Flujo Interno:* Recibe un objeto JSON con las claves simplificadas del formulario del frontend (`ingresos`, `deuda`, `historial`, etc.). Realiza un mapeo semántico hacia los nombres de columnas que el estimador requiere y ejecuta el método `.predict()` sobre el bosque cargado.
    * *Respuesta Exitosa (JSON):* `{"success": true, "prediction": 0}` (Donde 0 indica perfil aprobado y 1 denota riesgo de incumplimiento).

### B. Persistencia del Modelo de Aprendizaje (Pipeline de Guardado y Memoria)
Una vez que el algoritmo de ensamble completa el entrenamiento, no es eficiente volver a procesar miles de filas cada vez que un analista financiero requiera una predicción individual. Para resolver esto y garantizar tiempos de respuesta óptmos, el sistema implementa una estrategia híbrida de persistencia física y almacenamiento en memoria local:

* **Mecanismo de Persistencia Física:** Serialización binaria optimizada mediante la librería `joblib`. Se seleccionó esta biblioteca sobre la nativa `pickle` debido a su alta eficiencia y rendimiento superior al manipular objetos de Scikit-Learn que contienen grandes matrices numéricas y estructuras de datos pesadas (como los múltiples estimadores y nodos de un Bosque Aleatorio).
* **Ruta de Almacenamiento:** El estado interno de las ramas, pesos de decisión e índices de corte de los árboles entrenados se exporta de forma persistente a través de la función `joblib.dump()` en la ruta estructurada `backend/data/credit_model.pkl`.
* **Persistencia en Memoria Local (Inferencia de Alta Velocidad):** Al finalizar con éxito la fase de ajuste en el endpoint `/api/train`, el backend almacena la instancia del clasificador directamente en la variable global `modelo_base` dentro de `app.py`.
* **Operación de Inferencia:** Cuando se invoca el endpoint de predicción individual `/api/predict`, el servidor Flask ejecuta el método `.predict()` directamente sobre la variable global `modelo_base` alojada en la memoria local del hilo de ejecución. Esta arquitectura evita el impacto en rendimiento y la latencia por operaciones de Entrada/Salida (I/O) que significaría leer el archivo físico `.pkl` desde el disco en cada petición web entrante, permitiendo resolver la inferencia en milisegundos y manteniendo el archivo en disco como respaldo permanente ante reinicios del servidor.

---

## 8. Entrenamiento, Evaluación e Hiperparámetros

### A. Proceso de Entrenamiento y Validación
El conjunto de datos unificado se dividió de forma clásica utilizando una partición del **80% para el conjunto de entrenamiento** (ajuste de pesos y divisiones de nodos) y un **20% para el conjunto de prueba** (validación ciega con datos no vistos por el modelo).

### B. Métricas de Rendimiento Obtenidas (Conjunto de Validación)
El clasificador final calibrado arrojó las siguientes métricas de precisión y fiabilidad:
* **Exactitud (Accuracy):** **78.6%** — Mide la proporción total de predicciones correctas (tanto sanas como en riesgo) sobre el total de evaluaciones.
* **Precisión:** **77.4%** — Determina que de cada 100 clientes que el sistema etiqueta con "Riesgo de Incumplimiento", aproximadamente 77 están correctamente asignados, minimizando los falsos positivos (evita rechazar de forma injusta a clientes solventes).
* **Exhaustividad (Recall / Sensibilidad):** **79.0%** — Mide la capacidad del sistema para detectar a los verdaderos morosos. De cada 100 perfiles de alto riesgo reales en el dataset, el modelo captura exitosamente a 79 de ellos, reduciendo al mínimo los falsos negativos (aprobar créditos que se convertirán en pérdidas).
* **F1-Score:** **78.2%** — Representa la media armónica balanceada entre la Precisión y el Recall, sirviendo como el indicador definitivo de estabilidad algorítmica.

### C. Espacio de Ajuste Dinámico de Hiperparámetros
Para permitir la optimización del modelo según las necesidades cambiantes de cada institución financiera, el backend expone tres palancas de control dinámico procesadas a través del endpoint `/api/train`:
1.  **`n_estimators` (Cantidad de árboles de decisión):** Espacio de búsqueda recomendado: `[50 - 200]`. Un mayor volumen de estimadores incrementa la estabilidad de la frontera de decisión y reduce la varianza, requiriendo un mayor coste de tiempo de cómputo en el servidor.
2.  **`max_depth` (Profundidad máxima de los árboles):** Espacio de búsqueda recomendado: `[10 - 20]`. Controla el límite de bifurcaciones permitidas para cada árbol; valores demasiado altos provocan sobreajuste al ajustarse perfectamente al ruido del entrenamiento.
3.  **`max_leaf_nodes` (Número máximo de hojas por árbol):** Espacio de búsqueda recomendado: `[25 - 50]`. Modula el crecimiento horizontal de las ramas de decisión, controlando de manera fina la complejidad estructural general de los estimadores individuales.

