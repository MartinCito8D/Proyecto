// ==========================================
// CONTROL DE VISTAS (PESTAÑAS)
// ==========================================
function cambiarPestana(pestana) {
    const vistas = ['carga', 'rendimiento', 'hiperparametros', 'predicciones'];
    vistas.forEach(v => {
        const section = document.getElementById(`vista-${v}`);
        if (section) section.classList.add('hidden');
    });
    
    const vistaActiva = document.getElementById(`vista-${pestana}`);
    if (vistaActiva) vistaActiva.classList.remove('hidden');

    const links = document.querySelectorAll('.nav-link');
    links.forEach(l => l.classList.remove('active'));
    
    if (event && event.currentTarget) {
        event.currentTarget.classList.add('active');
    }
}

// ==========================================
// ASIGNACIÓN DE COMPONENTES DEL DOM
// ==========================================
const csvFileInput = document.getElementById('csvFile');
const btnLimpiar = document.getElementById('btnLimpiar');
const btnEntrenarBase = document.getElementById('btnEntrenarBase');
const fileNameDisplay = document.getElementById('fileNameDisplay');

window.metricsIniciales = null;

// Disparador del explorador de archivos local
csvFileInput.addEventListener('change', function() {
    if (csvFileInput.files.length > 0) {
        fileNameDisplay.textContent = csvFileInput.files[0].name;
        btnLimpiar.removeAttribute('disabled'); // Desbloquea Paso 2
    }
});

// ==========================================
// PASO 2: ENVÍO Y CONFIGURACIÓN PANDAS
// ==========================================
function subirYLimpiar() {
    const file = csvFileInput.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    fetch('http://127.0.0.1:5000/api/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            const alertBox = document.getElementById('alertLimpieza');
            alertBox.textContent = `✔️ Procesamiento Exitoso: ${data.registros} filas indexadas, ${data.nulos} nulos corregidos y ${data.duplicados} registros duplicados eliminados.`;
            alertBox.classList.remove('hidden');
            
            // Desbloqueo inmediato del disparador del Paso 3
            btnEntrenarBase.removeAttribute('disabled');
        } else {
            alert('Error al limpiar los datos: ' + data.message);
        }
    })
    .catch(error => {
        console.error('Error de comunicación:', error);
        alert('No se pudo conectar con Flask en el puerto 5000.');
    });
}

// ==========================================
// PASO 3: ENTRENAMIENTO E INYECCIÓN DE MÉTRICAS
// ==========================================
function entrenarModeloBase() {
    fetch('http://127.0.0.1:5000/api/train', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            document.getElementById('alertEntrenamiento').classList.remove('hidden');

            // 1. Inyección directa en tarjetas globales (Evaluación de Rendimiento)
            document.getElementById('m-accuracy').textContent = (data.metrics.accuracy * 100).toFixed(1) + '%';
            document.getElementById('m-precision').textContent = (data.metrics.precision * 100).toFixed(1) + '%';
            document.getElementById('m-recall').textContent = (data.metrics.recall * 100).toFixed(1) + '%';
            document.getElementById('m-f1').textContent = (data.metrics.f1_score * 100).toFixed(1) + '%';

            // 2. Mapeo de valores de la Matriz de Confusión obligatoria
            document.getElementById('cell-tn').textContent = data.matrix.tn;
            document.getElementById('cell-fp').textContent = data.matrix.fp;
            document.getElementById('cell-fn').textContent = data.matrix.fn;
            document.getElementById('cell-tp').textContent = data.matrix.tp;

            // 3. Distribución porcentual predictiva
            document.getElementById('dist-sin').textContent = data.distribution.sin_riesgo;
            document.getElementById('dist-con').textContent = data.distribution.en_riesgo;

            // Almacenar el estado base en caché global del navegador para comparativas
            window.metricsIniciales = data.metrics;

            // Inyectar en la tabla estática de la vista de hiperparámetros
            document.getElementById('c-acc-ant').textContent = (data.metrics.accuracy * 100).toFixed(1) + '%';
            document.getElementById('c-pre-ant').textContent = (data.metrics.precision * 100).toFixed(1) + '%';
            document.getElementById('c-rec-ant').textContent = (data.metrics.recall * 100).toFixed(1) + '%';
            document.getElementById('c-f1-ant').textContent = (data.metrics.f1_score * 100).toFixed(1) + '%';
        } else {
            alert('Falla en entrenamiento base: ' + data.message);
        }
    })
    .catch(error => console.error('Error en fetch de entrenamiento:', error));
}

// ==========================================
// MODIFICACIÓN DE HIPERPARAMETROS (REENTRENAR)
// ==========================================
function reentrenarHiperparametros() {
    if (!window.metricsIniciales) {
        alert('Debes entrenar el modelo inicial antes de realizar comparativas experimentales.');
        return;
    }

    const estimators = document.getElementById('param-estimators').value;
    const depth = document.getElementById('param-depth').value;
    const leaves = document.getElementById('param-leaves').value;

    fetch('http://127.0.0.1:5000/api/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            n_estimators: parseInt(estimators),
            max_depth: parseInt(depth),
            max_leaf_nodes: parseInt(leaves)
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            const n = data.metrics;
            const v = window.metricsIniciales;

            // Pintar columna del modelo modificado
            document.getElementById('c-acc-des').textContent = (n.accuracy * 100).toFixed(1) + '%';
            document.getElementById('c-pre-des').textContent = (n.precision * 100).toFixed(1) + '%';
            document.getElementById('c-rec-des').textContent = (n.recall * 100).toFixed(1) + '%';
            document.getElementById('c-f1-des').textContent = (n.f1_score * 100).toFixed(1) + '%';

            // Computar variaciones marginales
            const dAcc = (n.accuracy - v.accuracy) * 100;
            const dPre = (n.precision - v.precision) * 100;
            const dRec = (n.recall - v.recall) * 100;
            const dF1  = (n.f1_score - v.f1_score) * 100;

            const renderDiff = (id, val) => {
                const target = document.getElementById(id);
                if (val > 0) {
                    target.innerHTML = `<span style="color:#22c55e; font-weight:bold;">+${val.toFixed(1)}% 📈</span>`;
                } else if (val < 0) {
                    target.innerHTML = `<span style="color:#ef4444; font-weight:bold;">${val.toFixed(1)}% 📉</span>`;
                } else {
                    target.innerHTML = `<span style="color:#94a3b8;">0.0% =</span>`;
                }
            };

            renderDiff('c-acc-dif', dAcc);
            renderDiff('c-pre-dif', dPre);
            renderDiff('c-rec-dif', dRec);
            renderDiff('c-f1-dif', dF1);

            alert('¡Modelo reentrenado con éxito con los nuevos hiperparámetros!');
        } else {
            alert('Error en el reentrenamiento modular: ' + data.message);
        }
    })
    .catch(error => console.error('Error en optimización:', error));
}

// ==========================================
// PREDICCIÓN INDIVIDUAL EN TIEMPO REAL
// ==========================================
function predecirIndividual(event) {
    event.preventDefault();

    const payload = {
        ingresos: document.getElementById('f-ingresos').value,
        deuda: document.getElementById('f-deuda').value,
        historial: document.getElementById('f-historial').value,
        antiguedad: document.getElementById('f-antiguedad').value,
        creditos: document.getElementById('f-creditos').value,
        monto: document.getElementById('f-monto').value,
        atrasos: document.getElementById('f-atrasos').value,
        dependientes: document.getElementById('f-dependientes').value,
        utilizacion: document.getElementById('f-utilizacion').value
    };

    fetch('http://127.0.0.1:5000/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            const box = document.getElementById('resultBox');
            const txt = document.getElementById('resultText');
            box.classList.remove('hidden');

            if (data.prediction === 1) {
                txt.innerHTML = "❌ <span style='color:#ef4444; font-weight:bold;'>EN RIESGO DE INCUMPLIMIENTO</span>";
                box.style.borderLeft = "6px solid #ef4444";
                box.style.backgroundColor = "rgba(239, 68, 68, 0.05)";
            } else {
                txt.innerHTML = "🟢 <span style='color:#22c55e; font-weight:bold;'>SIN RIESGO DETECTADO</span>";
                box.style.borderLeft = "6px solid #22c55e";
                box.style.backgroundColor = "rgba(34, 197, 94, 0.05)";
            }
            box.scrollIntoView({ behavior: 'smooth' });
        } else {
            alert('Falla en la consulta del aplicante: ' + data.message);
        }
    })
    .catch(error => console.error('Error de inferencia individual:', error));
}