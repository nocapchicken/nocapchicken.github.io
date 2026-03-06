// AI-assisted (Claude Code, claude.ai) — https://claude.ai
/* nocapchicken — app.js */

(function () {
  'use strict';

  const form            = document.getElementById('searchForm');
  const btnSearch       = form.querySelector('.btn-search');
  const resultSec       = document.getElementById('resultSection');
  const resultCard      = document.getElementById('resultCard');
  const resultError     = document.getElementById('resultError');
  const resultSkeleton  = document.getElementById('resultSkeleton');
  const errorMsg        = document.getElementById('errorMessage');
  const restaurantInput = document.getElementById('restaurantName');
  const suggestionList  = document.getElementById('restaurant-suggestions');

  // ── Restaurant suggestions ──────────────────────────────────

  let suggestTimer = null;

  restaurantInput.addEventListener('input', () => {
    clearTimeout(suggestTimer);
    suggestTimer = setTimeout(async () => {
      const name = restaurantInput.value.trim();
      if (name.length < 2) { suggestionList.innerHTML = ''; return; }
      try {
        const params = new URLSearchParams({ name });
        const res   = await fetch(`/api/suggest?${params}`);
        const names = await res.json();
        suggestionList.innerHTML = names.map(n => `<option value="${escHtml(n)}">`).join('');
      } catch (err) {
        console.error('Suggestion fetch failed:', err);
      }
    }, 300);
  });

  // ── Form submit ─────────────────────────────────────────────

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const name = restaurantInput.value.trim();
    if (!name) return;

    setLoading(true);
    hideAll();
    resultSkeleton.hidden = false;
    resultSec.hidden = false;

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
      });
      const data = await res.json();
      if (!res.ok || data.error) {
        showError(data.error || 'Something went wrong. Please try again.');
      } else {
        renderResult(data);
      }
    } catch (err) {
      console.error('Prediction request failed:', err);
      showError('Network error. Please check your connection and try again.');
    } finally {
      setLoading(false);
    }
  });

  // ── Render result ───────────────────────────────────────────

  function renderResult(result) {
    // Grade pill
    const pill = document.getElementById('gradePill');
    pill.textContent = result.predicted_grade;
    pill.className = `grade-pill grade-${result.predicted_grade === '?' ? 'unknown' : result.predicted_grade}`;

    // Restaurant meta
    document.getElementById('gradeName').textContent = result.restaurant_name;
    document.getElementById('gradeLocation').textContent = result.location || '';
    document.getElementById('confidenceValue').textContent =
      result.confidence > 0 ? `${Math.round(result.confidence * 100)}%` : '—';

    // Divergence warning
    const divAlert = document.getElementById('divergenceAlert');
    divAlert.hidden = !result.divergence_warning;

    // Platform ratings
    renderPlatform(result.google_rating, result.google_review_count);

    // SHAP
    renderShap(result.top_shap_features);

    // Sample reviews
    renderReviews(result.sample_reviews);

    resultSkeleton.hidden = true;
    resultCard.hidden = false;
    resultSec.hidden  = false;
    resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  function renderPlatform(rating, count) {
    const starsEl = document.getElementById('googleStars');
    const countEl = document.getElementById('googleCount');

    if (rating != null) {
      starsEl.innerHTML = `${rating.toFixed(1)} <span class="star-glyph">★</span>`;
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

    if (!features || features.length === 0) {
      section.hidden = true;
      return;
    }

    const maxAbs = Math.max(...features.map(f => Math.abs(f.impact)), 0.001);

    features.forEach(f => {
      const pct  = Math.min(Math.abs(f.impact) / maxAbs * 100, 100).toFixed(1);
      const dir  = f.impact >= 0 ? 'positive' : 'negative';
      const sign = f.impact >= 0 ? '+' : '';
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

    if (!reviews || reviews.length === 0) {
      section.hidden = true;
      return;
    }

    reviews.forEach(text => {
      if (!text) return;
      const excerpt = text.length > 220 ? text.slice(0, 217) + '…' : text;
      list.insertAdjacentHTML('beforeend',
        `<li class="review-item">${escHtml(excerpt)}</li>`
      );
    });

    section.hidden = false;
  }

  // ── UI helpers ──────────────────────────────────────────────

  function showError(msg) {
    resultSkeleton.hidden = true;
    errorMsg.textContent = msg;
    resultError.hidden = false;
    resultSec.hidden   = false;
  }

  function hideAll() {
    resultCard.hidden     = true;
    resultError.hidden    = true;
    resultSkeleton.hidden = true;
    document.getElementById('divergenceAlert').hidden = true;
  }

  function setLoading(on) {
    btnSearch.classList.toggle('loading', on);
    btnSearch.disabled = on;
    btnSearch.setAttribute('aria-label', on ? 'Loading results' : 'Check it');
    resultSec.setAttribute('aria-busy', on ? 'true' : 'false');
  }

  // ── Theme toggle ─────────────────────────────────────────

  const themeBtn = document.getElementById('themeToggle');
  if (themeBtn) {
    const syncThemeBtn = () => {
      const dark = document.documentElement.dataset.theme === 'dark';
      themeBtn.setAttribute('aria-pressed', String(dark));
      themeBtn.setAttribute('aria-label', dark ? 'Disable dark mode' : 'Enable dark mode');
    };
    // Sync only when dark mode was restored; HTML defaults cover the light-mode case
    if (document.documentElement.dataset.theme === 'dark') syncThemeBtn();
    themeBtn.addEventListener('click', () => {
      const nowDark = document.documentElement.dataset.theme !== 'dark';
      document.documentElement.dataset.theme = nowDark ? 'dark' : 'light';
      localStorage.setItem('theme', nowDark ? 'dark' : 'light');
      syncThemeBtn();
    });
  }

  // Example chips — click to pre-fill and submit
  document.querySelectorAll('.example-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      restaurantInput.value = chip.dataset.name;
      form.requestSubmit();
    });
  });

  function escHtml(str) {
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }
})();
