// Netflix-style recommendations grid and modal

(function () {
    // Utility
    function qs(sel, root = document) { return root.querySelector(sel); }
    function qsa(sel, root = document) { return Array.from(root.querySelectorAll(sel)); }
    function escapeHTML(str) { return str.replace(/[&<>"']/g, s => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[s])); }
    function getQueryParam(name) {
        const m = window.location.search.match(new RegExp(`[?&]${name}=([^&]+)`));
        return m ? decodeURIComponent(m[1]) : null;
    }
    function setQueryParam(name, value) {
        const url = new URL(window.location);
        url.searchParams.set(name, value);
        window.history.pushState({}, '', url);
    }
    function removeQueryParam(name) {
        const url = new URL(window.location);
        url.searchParams.delete(name);
        window.history.pushState({}, '', url);
    }
    function lockBodyScroll() { document.body.style.overflow = 'hidden'; }
    function unlockBodyScroll() { document.body.style.overflow = ''; }

    // Focus trap
    function trapFocus(modal) {
        const focusable = qsa('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])', modal)
            .filter(el => !el.hasAttribute('disabled') && el.offsetParent !== null);
        if (focusable.length === 0) return;
        let first = focusable[0], last = focusable[focusable.length - 1];
        function handler(e) {
            if (e.key === 'Tab') {
                if (e.shiftKey && document.activeElement === first) {
                    e.preventDefault(); last.focus();
                } else if (!e.shiftKey && document.activeElement === last) {
                    e.preventDefault(); first.focus();
                }
            }
        }
        modal.addEventListener('keydown', handler);
        return () => modal.removeEventListener('keydown', handler);
    }

    // Loading skeleton for posters
    function posterSkeleton() {
        const div = document.createElement('div');
        div.className = 'poster-skeleton';
        return div;
    }

    // Loading spinner for modal
    function modalLoading(text) {
        const div = document.createElement('div');
        div.className = 'recommend-modal-loading';
        div.setAttribute('aria-live', 'polite');
        div.textContent = text || 'Loading...';
        return div;
    }

    // Render grid
    function renderGrid(movies, enriched) {
        const grid = qs('#recommend-grid');
        grid.innerHTML = '';
        movies.forEach((rec, i) => {
            const data = enriched[rec.tmdb_id] || {};
            const card = document.createElement('div');
            card.className = 'card recommend-card';
            card.setAttribute('tabindex', '0');
            card.setAttribute('role', 'button');
            card.setAttribute('aria-label', `${data.title || rec.name}${data.year ? ' (' + data.year + ')' : ''}`);
            card.dataset.tmdbId = data.tmdb_id || '';
            card.dataset.index = i;
            card.dataset.title = data.title || rec.name;
            card.dataset.genre = data.genres ? data.genres.join(', ') : '';
            card.dataset.year = data.year || '';
            card.dataset.pred = rec.pred || '';
            card.dataset.posterUrl = data.posterUrl || '';
            // Poster
            const posterWrap = document.createElement('div');
            posterWrap.className = 'poster-wrap';
            if (data.posterUrl) {
                const img = document.createElement('img');
                img.className = 'poster-img';
                img.src = data.posterUrl;
                img.alt = `${data.title || rec.name} poster`;
                posterWrap.appendChild(img);
            } else {
                posterWrap.appendChild(posterSkeleton());
            }
            card.appendChild(posterWrap);
            // Info
            const info = document.createElement('div');
            info.className = 'card-info';
            info.innerHTML = `
                <div class="movie-title">${escapeHTML(data.title || rec.name)}</div>
                ${data.year ? `<div class="movie-year">${data.year}</div>` : ''}
                ${data.genres ? `<div class="movie-genre">${escapeHTML(data.genres.join(', '))}</div>` : ''}
                ${data.rating ? `<div class="movie-rating">★ ${data.rating}</div>` : ''}
            `;
            card.appendChild(info);
            grid.appendChild(card);
        });
    }

    // Modal rendering
    let lastFocusedCard = null;
    function openModal(data, card) {
        lastFocusedCard = card;
        setQueryParam('movie', data.tmdb_id);
        lockBodyScroll();
        const root = qs('#recommend-modal-root');
        root.innerHTML = `
            <div class="recommend-modal-backdrop active" tabindex="-1"></div>
            <div class="recommend-modal-panel active" role="dialog" aria-modal="true" aria-labelledby="modal-title">
                <button class="recommend-modal-close" aria-label="Close dialog">&times;</button>
                <div class="recommend-modal-poster">
                    ${data.posterUrl ? `<img src="${data.posterUrl}" alt="${escapeHTML(data.title)} poster">` : `<div class="poster-skeleton"></div>`}
                </div>
                <div class="recommend-modal-content">
                    <div id="modal-title" class="recommend-modal-title">${escapeHTML(data.title || '')}</div>
                    <div class="recommend-modal-meta">
                        ${data.year ? `${data.year}` : ''} 
                        ${data.runtimeMinutes ? ` • ${data.runtimeMinutes} min` : ''} 
                        ${data.rating ? ` • ★ ${data.rating}` : ''}
                    </div>
                    ${data.genres ? `<div class="recommend-modal-genres">${escapeHTML(data.genres.join(', '))}</div>` : ''}
                    ${data.overview ? `<div class="recommend-modal-overview">${escapeHTML(data.overview)}</div>` : ''}
                    <div class="recommend-modal-providers">
                        <div class="recommend-modal-providers-title">Where to watch</div>
                        <div class="recommend-modal-provider-list">
                            ${Array.isArray(data.providers) && data.providers.length ? data.providers.map(p => `
                                <a class="recommend-modal-provider" href="${p.url || '#'}" target="_blank" rel="noopener">
                                    ${p.logoUrl ? `<img class="recommend-modal-provider-logo" src="${p.logoUrl}" alt="${escapeHTML(p.name)} logo">` : ''}
                                    <span>${escapeHTML(p.name)}</span>
                                </a>
                            `).join('') : `<span style="color:#bbb;">No streaming info for NL</span>`}
                        </div>
                    </div>
                </div>
            </div>
        `;
        // Focus trap
        const panel = qs('.recommend-modal-panel', root);
        panel.focus();
        const removeTrap = trapFocus(panel);
        // Close logic
        function closeModal() {
            unlockBodyScroll();
            root.innerHTML = '';
            removeTrap && removeTrap();
            removeQueryParam('movie');
            if (lastFocusedCard) lastFocusedCard.focus();
        }
        qs('.recommend-modal-close', root).onclick = closeModal;
        qs('.recommend-modal-backdrop', root).onclick = closeModal;
        root.onkeydown = e => { if (e.key === 'Escape') closeModal(); };
        // Prevent tabbing out of modal
        panel.tabIndex = -1;
        panel.focus();
    }

    // Deep-linking: open modal if ?movie=<tmdb_id>
    function openModalById(tmdb_id, enriched) {
        const data = enriched[tmdb_id];
        if (data) {
            // Find card to restore focus
            const card = qs(`.card[data-tmdb-id="${tmdb_id}"]`);
            openModal(data, card);
        }
    }

    // Enrichment API
    async function enrichRecommendations(recs) {
        // POST to /api/enrich, limit concurrency to 5
        const resp = await fetch('/api/enrich', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(recs)
        });
        if (!resp.ok) throw new Error('Enrichment failed');
        const arr = await resp.json();
        // Return as { [tmdb_id]: enrichedObj }
        const out = {};
        arr.forEach(obj => { if (obj.tmdb_id) out[obj.tmdb_id] = obj; });
        return out;
    }

    // After renderGrid, attach modal handlers:
    function attachModalHandlers() {
        const modal = document.getElementById('movie-modal');
        const modalBody = modal.querySelector('.movie-modal-body');
        const closeBtn = modal.querySelector('.movie-modal-close');
        const backdrop = modal.querySelector('.movie-modal-backdrop');

        function fetchMovieInfo(movieId, cb) {
            fetch(`/api/movie_info/${movieId}`)
                .then(r => r.json())
                .then(data => cb(data))
                .catch(() => cb(null));
        }

        function openModal(card) {
            const movieId = card.dataset.movieId;
            if (!movieId) {
                modalBody.innerHTML = `<div>Could not load info.<br>No movie ID found.</div>`;
                modal.style.display = 'flex';
                document.body.style.overflow = 'hidden';
                return;
            }
            modalBody.innerHTML = `<div>Loading...</div>`;
            modal.style.display = 'flex';
            document.body.style.overflow = 'hidden';
            fetchMovieInfo(movieId, function (data) {
                if (!data || data.error) {
                    modalBody.innerHTML = `<div>Could not load info.</div>`;
                    return;
                }
                modalBody.innerHTML = `
                    ${data.Poster && data.Poster !== "N/A" ? `<img src="${data.Poster}" alt="${data.Title} poster">` : ''}
                    <div class="movie-modal-title">${data.Title}</div>
                    <div class="movie-modal-meta">${data.Year} • ${data.Genre}</div>
                    <div class="movie-modal-score">IMDB: ${data.imdbRating}</div>
                    <div class="movie-modal-desc">${data.Plot}</div>
                    ${data.Website && data.Website !== "N/A" ? `<div><a href="${data.Website}" target="_blank">Official Site</a></div>` : ''}
                `;
            });
        }

        function closeModal() {
            modal.style.display = 'none';
            document.body.style.overflow = '';
        }

        closeBtn.addEventListener('click', closeModal);
        backdrop.addEventListener('click', closeModal);

        document.querySelectorAll('.recommend-card').forEach(card => {
            card.addEventListener('click', () => openModal(card));
            card.addEventListener('keydown', e => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    openModal(card);
                }
            });
        });
    }

    // Main
    document.addEventListener('DOMContentLoaded', async function () {
        if (!window.recommendations || !Array.isArray(window.recommendations)) {
            qs('#recommend-grid').innerHTML = '<div style="color:#ff6b6b;">No recommendations found.</div>';
            return;
        }
        qs('#recommend-grid').innerHTML = '<div class="recommend-modal-loading" aria-live="polite">Loading recommendations...</div>';
        let enriched = {};
        try {
            enriched = await enrichRecommendations(window.recommendations);
        } catch (e) {
            qs('#recommend-grid').innerHTML = '<div style="color:#ff6b6b;">Failed to load movie details.</div>';
            return;
        }
        renderGrid(window.recommendations, enriched);
        attachModalHandlers(); // <-- Attach handlers after rendering cards
        // Deep-link modal
        const movieId = getQueryParam('movie');
        if (movieId && enriched[movieId]) openModalById(movieId, enriched);
    });
})();