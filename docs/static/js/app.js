// AI-assisted (Claude Code, claude.ai) — https://claude.ai
/* nocapchicken — app.js
 * Browser-side inference: onnxruntime-web + pre-computed restaurant lookup.
 * No backend required — model and data ship with the site.
 */

// ort is loaded globally from cdn.jsdelivr.net/npm/onnxruntime-web (see index.html)
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';

// ── Module-level state (loaded once) ────────────────────────────

let restaurantLookup = null;   // { "pizza hut": { name, features, rating, … }, … }
let ortSession       = null;
let loadError        = null;

// Feature order must match ONNX model's float_input and export_browser_data.py
const N_FEATURES = 6;

// ── Asset loading ─────────────────────────────────────────────

const assetsReady = (async () => {
  const [lookupRes, sessionRes] = await Promise.allSettled([
    fetch('static/data/restaurants.json').then(r => r.json()),
    ort.InferenceSession.create('models/random_forest.onnx', {
      executionProviders: ['wasm'],
    }),
  ]);

  if (lookupRes.status === 'rejected') {
    loadError = 'Failed to load restaurant database.';
    return;
  }
  if (sessionRes.status === 'rejected') {
    loadError = 'Failed to load inference model.';
    return;
  }

  restaurantLookup = lookupRes.value;
  ortSession       = sessionRes.value;
})();

// ── DOM refs ──────────────────────────────────────────────────

const form            = document.getElementById('searchForm');
const btnSearch       = form.querySelector('.btn-search');
const resultSec       = document.getElementById('resultSection');
const resultCard      = document.getElementById('resultCard');
const resultError     = document.getElementById('resultError');
const resultSkeleton  = document.getElementById('resultSkeleton');
const errorMsg        = document.getElementById('errorMessage');
const restaurantInput = document.getElementById('restaurantName');
const suggestionList  = document.getElementById('restaurant-suggestions');

// ── Suggestions (client-side) ─────────────────────────────────

let suggestTimer = null;

restaurantInput.addEventListener('input', () => {
  clearTimeout(suggestTimer);
  suggestTimer = setTimeout(() => {
    const query = restaurantInput.value.trim().toLowerCase();
    if (query.length < 2 || !restaurantLookup) { suggestionList.innerHTML = ''; return; }

    const exact = [];
    const fuzzy = [];

    for (const [key, entry] of Object.entries(restaurantLookup)) {
      if (key.includes(query)) {
        exact.push(entry.name);
      } else if (bigramSimilarity(query, key) >= 0.5) {
        fuzzy.push(entry.name);
      }
      if (exact.length >= 5) break;
    }

    const names = [...exact, ...fuzzy].slice(0, 5);
    suggestionList.innerHTML = names.map(n => `<option value="${escHtml(n)}">`).join('');
  }, 200);
});

// Jaccard similarity on character bigrams
function bigramSimilarity(a, b) {
  const bigrams = s => {
    const set = new Set();
    for (let i = 0; i < s.length - 1; i++) set.add(s[i] + s[i + 1]);
    return set;
  };
  const sa = bigrams(a);
  const sb = bigrams(b);
  let intersection = 0;
  for (const bg of sa) { if (sb.has(bg)) intersection++; }
  const union = sa.size + sb.size - intersection;
  return union === 0 ? 0 : intersection / union;
}

// ── Form submit ───────────────────────────────────────────────

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const name = restaurantInput.value.trim();
  if (!name) return;

  setLoading(true);
  hideAll();
  resultSkeleton.hidden = false;
  resultSec.hidden      = false;

  try {
    await assetsReady;

    if (loadError) { showError(loadError); return; }

    const data = await predict(name);
    data.error ? showError(data.error) : renderResult(data);
  } catch (err) {
    showError('Inference error: ' + err.message);
  } finally {
    setLoading(false);
  }
});

// ── Inference ─────────────────────────────────────────────────

async function predict(name) {
  const entry = findRestaurant(name);
  if (!entry) {
    return { error: 'No review data found for this restaurant — prediction unavailable.' };
  }

  const tensor = new ort.Tensor('float32', new Float32Array(entry.features), [1, N_FEATURES]);

  const t0      = performance.now();
  const results = await ortSession.run({ float_input: tensor });
  const inferMs = performance.now() - t0;

  // zipmap:false → outputs are 'label' (int64[1]) and 'probabilities' (float32[1,2])
  const proba      = Array.from(results['probabilities'].data);  // [prob_A, prob_flagged]
  const predClass  = proba[1] > proba[0] ? 1 : 0;
  const confidence = proba[predClass];
  const grade      = predClass === 0 ? 'A' : 'Flagged';

  return {
    restaurant_name:     entry.name,
    location:            entry.location || '',
    predicted_grade:     grade,
    grade_color:         predClass === 0 ? 'green' : 'red',
    confidence,
    google_rating:       entry.rating,
    google_review_count: entry.review_count,
    top_shap_features:   [],   // SHAP not available in browser
    divergence_warning:  grade === 'Flagged' && entry.rating != null && entry.rating >= 4.0,
    actual_grade:        entry.actual_grade  || null,
    actual_score:        entry.actual_score  || null,
    actual_date:         entry.actual_date   || null,
    sample_reviews:      entry.sample_reviews || [],
    // Inference trace
    features:            entry.features,
    probabilities:       proba,
    infer_ms:            inferMs,
  };
}

function findRestaurant(name) {
  if (!restaurantLookup) return null;
  const key = name.trim().toLowerCase();
  if (restaurantLookup[key]) return restaurantLookup[key];

  let bestKey = null, bestScore = 0;
  for (const candidate of Object.keys(restaurantLookup)) {
    const score = candidate.includes(key) || key.includes(candidate)
      ? 0.9
      : bigramSimilarity(key, candidate);
    if (score > bestScore) { bestScore = score; bestKey = candidate; }
  }

  return bestScore >= 0.5 ? restaurantLookup[bestKey] : null;
}

// ── Render result ─────────────────────────────────────────────

function renderResult(d) {
  const pill = document.getElementById('gradePill');
  pill.textContent = d.predicted_grade;
  pill.className = `grade-pill grade-${d.predicted_grade === '?' ? 'unknown' : d.predicted_grade}`;

  document.getElementById('gradeName').textContent     = d.restaurant_name;
  document.getElementById('gradeLocation').textContent = d.location || '';
  document.getElementById('confidenceValue').textContent =
    d.confidence > 0 ? `${Math.round(d.confidence * 100)}%` : '—';

  const actualWrap    = document.getElementById('actualGradeWrap');
  const actualPill    = document.getElementById('actualGradePill');
  const actualScoreEl = document.getElementById('actualScore');
  const actualDateEl  = document.getElementById('actualDate');
  if (d.actual_grade) {
    actualPill.textContent    = d.actual_grade;
    actualPill.className      = `grade-pill ${d.actual_grade === 'A' ? 'grade-A' : 'grade-Flagged'}`;
    actualScoreEl.textContent = d.actual_score ? `${d.actual_score} / 100` : '';
    actualDateEl.textContent  = d.actual_date ? `inspected ${d.actual_date}` : '';
    actualWrap.hidden = false;
  } else {
    actualWrap.hidden = true;
  }

  document.getElementById('divergenceAlert').hidden = !d.divergence_warning;

  renderPlatform(d.google_rating, d.google_review_count);
  renderShap(d.top_shap_features);
  renderReviews(d.sample_reviews);
  renderTrace(d);

  resultSkeleton.hidden = true;
  resultCard.hidden     = false;
  resultSec.hidden      = false;
  resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

const FEATURE_LABELS = [
  'Google rating',
  'log(review count)',
  'Review word count',
  'Avg word length',
  'Safety keyword hits',
  'Negative phrase hits',
];

function renderTrace(d) {
  if (!d.features) return;

  const featureRows = d.features.map((v, i) =>
    `<tr><td>${FEATURE_LABELS[i]}</td><td>${Number(v).toFixed(4)}</td></tr>`
  ).join('');
  document.getElementById('featureTableBody').innerHTML = featureRows;

  const probaRows = [
    `<tr><td>A (safe)</td><td>${d.probabilities[0].toFixed(6)}</td></tr>`,
    `<tr><td>Flagged</td><td>${d.probabilities[1].toFixed(6)}</td></tr>`,
  ].join('');
  document.getElementById('probaTableBody').innerHTML = probaRows;

  document.getElementById('traceTiming').textContent =
    `ONNX session.run(): ${d.infer_ms.toFixed(2)} ms`;

  document.getElementById('lastTrace').hidden = false;

  // Pre-fill manual inputs with values from this prediction
  d.features.forEach((v, i) => {
    const input = document.getElementById(`fi${i}`);
    if (input) input.value = Number(v).toFixed(input.step && input.step < 1 ? 2 : 1);
  });
}

function renderPlatform(rating, count) {
  const starsEl = document.getElementById('googleStars');
  const countEl = document.getElementById('googleCount');
  if (rating != null) {
    starsEl.innerHTML   = `${rating.toFixed(1)} <span class="star-glyph">★</span>`;
    countEl.textContent = count ? `${count.toLocaleString()} reviews` : '';
  } else {
    starsEl.textContent = 'N/A';
    countEl.textContent = 'Not found on Google';
  }
}

function renderShap(features) {
  const section = document.getElementById('shapSection');
  const list    = document.getElementById('shapList');
  list.innerHTML = '';

  if (!features || features.length === 0) { section.hidden = true; return; }

  const maxAbs = Math.max(...features.map(f => Math.abs(f.impact)), 0.001);
  features.forEach(f => {
    const pct   = Math.min(Math.abs(f.impact) / maxAbs * 100, 100).toFixed(1);
    const dir   = f.impact >= 0 ? 'positive' : 'negative';
    const sign  = f.impact >= 0 ? '+' : '';
    const label = f.feature.replace(/_/g, ' ').replace(/ log$/, ' (log)');
    list.insertAdjacentHTML('beforeend', `
      <li class="shap-item" aria-label="${label}: ${dir} impact, ${sign}${f.impact.toFixed(3)}">
        <span class="shap-feature" aria-hidden="true">${label}</span>
        <div class="shap-bar-wrap" aria-hidden="true">
          <div class="shap-bar ${dir}" style="width:${pct}%"></div>
        </div>
        <span class="shap-impact" aria-hidden="true">${sign}${f.impact.toFixed(3)}</span>
      </li>
    `);
  });
  section.hidden = false;
}

function renderReviews(reviews) {
  const section = document.getElementById('reviewsSection');
  const list    = document.getElementById('reviewsList');
  list.innerHTML = '';

  if (!reviews || reviews.length === 0) { section.hidden = true; return; }

  reviews.forEach(text => {
    if (!text) return;
    const excerpt = text.length > 220 ? text.slice(0, 217) + '…' : text;
    list.insertAdjacentHTML('beforeend', `<li class="review-item">${escHtml(excerpt)}</li>`);
  });
  section.hidden = false;
}

// ── UI helpers ────────────────────────────────────────────────

function showError(msg) {
  resultSkeleton.hidden = true;
  errorMsg.textContent  = msg;
  resultError.hidden    = false;
  resultSec.hidden      = false;
}

function hideAll() {
  resultCard.hidden     = true;
  resultError.hidden    = true;
  resultSkeleton.hidden = true;
  document.getElementById('divergenceAlert').hidden = true;
  document.getElementById('actualGradeWrap').hidden = true;
  const trace = document.getElementById('traceSection');
  if (trace) trace.removeAttribute('open');
}

function setLoading(on) {
  btnSearch.classList.toggle('loading', on);
  btnSearch.disabled = on;
  btnSearch.setAttribute('aria-label', on ? 'Loading results' : 'Check it');
  resultSec.setAttribute('aria-busy', on ? 'true' : 'false');
}

// ── Theme toggle ──────────────────────────────────────────────

const themeBtn = document.getElementById('themeToggle');
if (themeBtn) {
  const syncThemeBtn = () => {
    const dark = document.documentElement.dataset.theme === 'dark';
    themeBtn.setAttribute('aria-pressed', String(dark));
    themeBtn.setAttribute('aria-label', dark ? 'Disable dark mode' : 'Enable dark mode');
  };
  if (document.documentElement.dataset.theme === 'dark') syncThemeBtn();
  themeBtn.addEventListener('click', () => {
    const nowDark = document.documentElement.dataset.theme !== 'dark';
    document.documentElement.dataset.theme = nowDark ? 'dark' : 'light';
    localStorage.setItem('theme', nowDark ? 'dark' : 'light');
    syncThemeBtn();
  });
}

// Example chips
document.querySelectorAll('.example-chip').forEach(chip => {
  chip.addEventListener('click', () => {
    restaurantInput.value = chip.dataset.name;
    form.requestSubmit();
  });
});

// ── Console API ───────────────────────────────────────────────
// Exposed for manual verification from DevTools:
//
//   await ncap.predict([4.5, 5.30, 300, 4.1, 2, 0])
//   await ncap.lookup('cosmic cantina')
//
// Features order: [google_rating, log(review_count), word_count, avg_word_len,
//                  safety_keyword_hits, negative_phrase_hits]

window.ncap = {
  async predict(features) {
    await assetsReady;
    const tensor  = new ort.Tensor('float32', new Float32Array(features), [1, N_FEATURES]);
    const t0      = performance.now();
    const results = await ortSession.run({ float_input: tensor });
    const ms      = performance.now() - t0;
    const proba   = Array.from(results['probabilities'].data);
    const label   = proba[1] > proba[0] ? 'Flagged' : 'A';
    console.table({ 'P(A)': proba[0].toFixed(6), 'P(Flagged)': proba[1].toFixed(6), label, 'ms': ms.toFixed(2) });
    return { label, proba, ms };
  },
  async lookup(name) {
    await assetsReady;
    const entry = findRestaurant(name);
    if (!entry) { console.warn('Not found:', name); return null; }
    console.log('Features:', FEATURE_LABELS.map((l, i) => `${l}: ${entry.features[i]}`).join(', '));
    return entry;
  },
};

// ── Manual trace run ──────────────────────────────────────

document.getElementById('traceRunBtn').addEventListener('click', async () => {
  const btn    = document.getElementById('traceRunBtn');
  const output = document.getElementById('traceRunOutput');
  const values = [0, 1, 2, 3, 4, 5].map(i => parseFloat(document.getElementById(`fi${i}`).value) || 0);

  btn.disabled  = true;
  output.textContent = 'Running…';

  try {
    await assetsReady;
    const tensor  = new ort.Tensor('float32', new Float32Array(values), [1, N_FEATURES]);
    const t0      = performance.now();
    const results = await ortSession.run({ float_input: tensor });
    const ms      = performance.now() - t0;
    const proba   = Array.from(results['probabilities'].data);
    output.textContent =
      `P(A) ${proba[0].toFixed(4)}  ·  P(Flagged) ${proba[1].toFixed(4)}  ·  ${ms.toFixed(1)} ms`;
  } catch (err) {
    output.textContent = 'Error: ' + err.message;
  } finally {
    btn.disabled = false;
  }
});

function escHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
