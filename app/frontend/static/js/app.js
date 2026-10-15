async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  });

  if (!response.ok) {
    throw new Error('Erreur de requête');
  }

  return response.json();
}

if ('scrollRestoration' in window.history) {
  window.history.scrollRestoration = 'manual';
}
window.addEventListener('pageshow', () => window.scrollTo(0, 0), { once: true });

async function loadSummary() {
  try {
    const data = await fetchJson('/api/summary');
    const metrics = data.metrics || {};

    const metricValues = {
      accuracyValue: metrics.accuracy,
      precisionValue: metrics.precision,
      recallValue: metrics.recall,
      f1Value: metrics.f1,
      aucValue: metrics.auc_roc,
    };
    Object.entries(metricValues).forEach(([id, value]) => {
      const node = document.getElementById(id);
      if (node) node.textContent = value == null ? '--' : `${(value * 100).toFixed(2)}%`;
    });

    const modelStatus = document.getElementById('modelStatus');
    const statusText = data.model_mode === 'production' ? 'Modèle prêt' : 'Mode démonstration';
    if (modelStatus) {
      modelStatus.innerHTML = `<i class="fa-solid ${data.model_mode === 'production' ? 'fa-circle-check' : 'fa-flask'}"></i><span>${statusText}</span>`;
    }

    if (document.getElementById('testRecords')) {
      renderConfusionMatrix(metrics.confusion_matrix);
    }
    const testRecords = document.getElementById('testRecords');
    if (testRecords) testRecords.textContent = metrics.records ?? 0;
  } catch (error) {
    console.error(error);
  }
}

function renderConfusionMatrix(confusion = [[0, 0], [0, 0]]) {
  const matrixNode = document.getElementById('confusionMatrix');
  if (!matrixNode) return;
  matrixNode.replaceChildren();

  confusion.forEach((row, rowIndex) => {
    row.forEach((cell, colIndex) => {
      const div = document.createElement('div');
      div.className = 'matrix-cell';
      div.innerHTML = `<span>${rowIndex === 0 ? 'Vrai 0' : 'Vrai 1'} / ${colIndex === 0 ? 'Préd. 0' : 'Préd. 1'}</span><strong>${cell}</strong>`;
      matrixNode.appendChild(div);
    });
  });
}

async function loadFeatureOptions() {
  if (!document.getElementById('predictionForm')) return;
  try {
    const options = await fetchJson('/api/options');
    Object.entries(options).forEach(([name, values]) => {
      const select = document.querySelector(`#predictionForm [name="${name}"]`);
      if (!select) return;
      select.replaceChildren(...values.map((value) => {
        const option = document.createElement('option');
        option.value = value;
        option.textContent = value;
        return option;
      }));
    });
  } catch (error) {
    console.error(error);
  }
}

async function submitPrediction(event) {
  event.preventDefault();

  const form = event.target;
  const data = Object.fromEntries(new FormData(form).entries());
  const resultBox = document.getElementById('predictionResult');
  resultBox.classList.remove('risk-high', 'risk-low');

  try {
    const result = await fetchJson('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    const isHighRisk = result.prediction_code === 1;
    resultBox.classList.add(isHighRisk ? 'risk-high' : 'risk-low');
    const icon = isHighRisk ? 'fa-triangle-exclamation' : 'fa-circle-check';
    resultBox.innerHTML = `<p>Résultat de la prédiction</p><h3><i class="fa-solid ${icon}" aria-hidden="true"></i> ${result.decision}</h3><p>Probabilité: ${result.probability}%</p>`;
  } catch (error) {
    resultBox.innerHTML = '<p>La prédiction n’a pas pu être calculée.</p>';
    console.error(error);
  }
}

async function handleHistoryUpload(event) {
  const file = event.target.files?.[0];
  if (!file) return;

  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch('/api/test-historical', { method: 'POST', body: formData });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || result.error || 'Erreur durant le test historique.');
    document.getElementById('historyRecords').textContent = result.records ?? 0;
    document.getElementById('historyAccuracy').textContent = `${((result.accuracy ?? 0) * 100).toFixed(1)}%`;
    renderConfusionMatrix(result.confusion_matrix);
  } catch (error) {
    document.getElementById('historyAccuracy').textContent = error.message;
  }
}

document.getElementById('predictionForm')?.addEventListener('submit', submitPrediction);
document.getElementById('historyFile')?.addEventListener('change', handleHistoryUpload);
loadSummary();
loadFeatureOptions();
