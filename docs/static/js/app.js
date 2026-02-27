// AI-assisted (Claude Code, claude.ai) — https://claude.ai
(function () {
  'use strict';

  const API_BASE = 'https://nocapchicken-github-io.onrender.com';

  const form        = document.getElementById('searchForm');
  const btnSearch   = form.querySelector('.btn-search');
  const resultSec   = document.getElementById('resultSection');
  const resultCard  = document.getElementById('resultCard');
  const resultError = document.getElementById('resultError');
  const errorMsg    = document.getElementById('errorMessage');

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const name = document.getElementById('restaurantName').value.trim();
    const city = document.getElementById('city').value.trim();
    if (!name || !city) return;

    setLoading(true);
    hideAll();

    try {
      const res = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, city }),
      });
      const data = await res.json();
      if (!res.ok || data.error) {
        showError(data.error || 'Something went wrong. Please try again.');
      } else {
        renderResult(data);
      }
    } catch (_) {
      showError('Network error. Please check your connection and try again.');
    } finally {
      setLoading(false);
    }
  });

  function renderResult(d) {
    const pill = document.getElementById('gradePill');
    pill.textContent = d.predicted_grade;
    pill.className = `grade-pill grade-${d.predicted_grade === '?' ? 'unknown' : d.predicted_grade}`;

    document.getElementById('gradeName').textContent = d.restaurant_name;
    document.getElementById('gradeLocation').textContent = `${d.location}, NC`;
    document.getElementById('confidenceValue').textContent =
      d.confidence > 0 ? `${Math.round(d.confidence * 100)}%` : '—';

    const divAlert = document.getElementById('divergenceAlert');
    divAlert.hidden = !d.divergence_warning;

    renderPlatform('yelp', d.yelp_rating, d.yelp_review_count);
    renderPlatform('google', d.google_rating, d.google_review_count);

    const deltaCard = document.getElementById('deltaCard');
    const deltaVal  = document.getElementById('deltaValue');
    if (d.rating_delta != null) {
      deltaVal.textContent = d.rating_delta.toFixed(1);
      deltaCard.style.display = '';
    } else {
      deltaCard.style.display = 'none';
    }

    renderShap(d.top_shap_features);
    renderReviews(d.sample_reviews);

    resultCard.hidden = false;
    resultSec.hidden  = false;
    resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  function renderPlatform(platform, rating, count) {
    const platformNames = { yelp: 'Yelp', google: 'Google' };
    const starsEl = document.getElementById(`${platform}Stars`);
    const countEl = document.getElementById(`${platform}Count`);

    if (rating != null) {
      starsEl.textContent = `${rating.toFixed(1)} ★`;
      countEl.textContent = count ? `${count.toLocaleString()} reviews` : '';
    } else {
      starsEl.textContent = 'N/A';
      countEl.textContent = `Not found on ${platformNames[platform] || platform}`;
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
        <li class="shap-item">
          <span class="shap-feature">${label}</span>
          <div class="shap-bar-wrap">
            <div class="shap-bar ${dir}" style="width:${pct}%"></div>
          </div>
          <span class="shap-impact">${sign}${f.impact.toFixed(3)}</span>
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

  function showError(msg) {
    errorMsg.textContent = msg;
    resultError.hidden = false;
    resultSec.hidden   = false;
  }

  function hideAll() {
    resultCard.hidden  = true;
    resultError.hidden = true;
  }

  function setLoading(on) {
    btnSearch.classList.toggle('loading', on);
    btnSearch.disabled = on;
    btnSearch.style.position = on ? 'relative' : '';
  }

  function escHtml(str) {
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }
})();
