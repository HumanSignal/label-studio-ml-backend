/**
 * ReID UI - Pairwise comparison interface for identity clustering.
 *
 * Shows pairs of crops for the user to judge: "Are they the same person?"
 * Mixes ambiguous pairs with calibration pairs (confident same/different).
 *
 * Relies on the global API object (from app.js) for HTTP requests
 * and showToast() for notifications.
 */

/* exported ReIDUI */
/* global API, showToast, AppState, navigate */

class ReIDUI {
    constructor(container) {
        this.container = container;
        this.pairs = [];
        this.currentPairIndex = 0;
        this.resolutions = {};       // {pair_id: "same"|"different"|"unsure"}
        this.sessionId = null;
        this.clusters = {};
        this.nIdentities = 0;
        this.totalPairs = 0;
        this.onComplete = null;      // callback(mergeResult) when all pairs resolved

        this._boundKeyHandler = this._handleKeyDown.bind(this);
    }

    // ------------------------------------------------------------------
    // Lifecycle
    // ------------------------------------------------------------------

    async init(sessionId) {
        this.sessionId = sessionId;
        this.resolutions = {};
        this.currentPairIndex = 0;

        try {
            const data = await API.get('/reid/clusters', {
                session_id: sessionId,
            });
            this.pairs = data.unresolved_pairs || [];
            this.clusters = data.clusters || {};
            this.nIdentities = data.n_identities || 0;
            this.totalPairs = data.total_pairs || 0;
        } catch (err) {
            showToast('Failed to load ReID clusters: ' + err.message, 'error');
            return;
        }

        document.addEventListener('keydown', this._boundKeyHandler);
        this._render();

        if (this.pairs.length === 0) {
            this._renderSummary();
        } else {
            this._showPair(0);
        }
    }

    destroy() {
        document.removeEventListener('keydown', this._boundKeyHandler);
    }

    // ------------------------------------------------------------------
    // Top-level render
    // ------------------------------------------------------------------

    _render() {
        this.container.innerHTML = '';

        var root = document.createElement('div');
        root.className = 'reid-panel';

        // Left column: two frame panels stacked
        var frameTop = document.createElement('div');
        frameTop.className = 'reid-frame-top frame-viewer';
        frameTop.id = 'reid-frame-a';

        var frameBottom = document.createElement('div');
        frameBottom.className = 'reid-frame-bottom frame-viewer';
        frameBottom.id = 'reid-frame-b';

        // Right column: crops + verdict + merge status
        var comparison = document.createElement('div');
        comparison.className = 'reid-comparison';
        comparison.id = 'reid-comparison';

        root.appendChild(frameTop);
        root.appendChild(frameBottom);
        root.appendChild(comparison);

        this.container.appendChild(root);
    }

    // ------------------------------------------------------------------
    // Show a specific pair
    // ------------------------------------------------------------------

    async _showPair(index) {
        if (index < 0 || index >= this.pairs.length) return;
        this.currentPairIndex = index;

        var pair = this.pairs[index];

        // Fetch pair frame data from backend
        var pairData;
        try {
            pairData = await API.get('/reid/pair/' + pair.pair_id + '/frames', {
                session_id: this.sessionId,
            });
        } catch (err) {
            showToast('Failed to load pair data: ' + err.message, 'error');
            return;
        }

        var cropA = pairData.crop_a;
        var cropB = pairData.crop_b;
        if (!cropA || !cropB) {
            showToast('Missing crop data for this pair.', 'warning');
            return;
        }

        // -- Frame A with highlighted box --
        this._renderFrame('reid-frame-a', cropA, 'A');

        // -- Frame B with highlighted box --
        this._renderFrame('reid-frame-b', cropB, 'B');

        // -- Comparison panel --
        this._renderComparison(pair, cropA, cropB);
    }

    _renderFrame(containerId, crop, label) {
        var container = document.getElementById(containerId);
        container.innerHTML = '';

        var wrapper = document.createElement('div');
        wrapper.style.position = 'relative';
        wrapper.style.display = 'inline-block';
        wrapper.style.maxWidth = '100%';
        wrapper.style.maxHeight = '100%';

        var img = document.createElement('img');
        img.src = '/interview/api/detect/frame/' + crop.frame_idx +
            '?session_id=' + encodeURIComponent(this.sessionId);
        img.alt = 'Frame ' + crop.frame_idx;
        img.style.display = 'block';
        img.style.maxWidth = '100%';
        img.style.maxHeight = '100%';
        img.style.objectFit = 'contain';

        // Draw highlight box after image loads
        img.onload = function () {
            var scaleX = img.clientWidth / img.naturalWidth;
            var scaleY = img.clientHeight / img.naturalHeight;

            var box = document.createElement('div');
            box.className = 'overlay-box active';
            box.style.position = 'absolute';
            box.style.left = (crop.xyxy[0] * scaleX) + 'px';
            box.style.top = (crop.xyxy[1] * scaleY) + 'px';
            box.style.width = ((crop.xyxy[2] - crop.xyxy[0]) * scaleX) + 'px';
            box.style.height = ((crop.xyxy[3] - crop.xyxy[1]) * scaleY) + 'px';
            box.style.borderColor = label === 'A' ? '#3498db' : '#e67e22';
            box.style.borderWidth = '3px';
            box.style.borderStyle = 'solid';
            box.style.pointerEvents = 'none';
            wrapper.appendChild(box);
        };

        // Frame label badge
        var badge = document.createElement('span');
        badge.className = 'card-badge';
        badge.style.position = 'absolute';
        badge.style.top = '8px';
        badge.style.left = '8px';
        badge.style.fontSize = '0.75rem';
        badge.style.padding = '2px 8px';
        badge.style.zIndex = '10';
        badge.textContent = label + ' - Frame ' + crop.frame_idx;

        wrapper.appendChild(img);
        wrapper.appendChild(badge);
        container.appendChild(wrapper);
    }

    _renderComparison(pair, cropA, cropB) {
        var container = document.getElementById('reid-comparison');
        container.innerHTML = '';

        // -- Progress bar --
        var progressWrapper = document.createElement('div');
        progressWrapper.className = 'progress-bar-wrapper';

        var track = document.createElement('div');
        track.className = 'progress-bar-track';
        var fill = document.createElement('div');
        fill.className = 'progress-bar-fill';
        var resolved = Object.keys(this.resolutions).length;
        var pct = this.pairs.length > 0
            ? ((resolved / this.pairs.length) * 100)
            : 0;
        fill.style.width = pct + '%';
        track.appendChild(fill);

        var text = document.createElement('div');
        text.className = 'progress-text';

        var ambiguousRemaining = 0;
        for (var i = 0; i < this.pairs.length; i++) {
            if (!this.resolutions[this.pairs[i].pair_id] && this.pairs[i].pool === 'ambiguous') {
                ambiguousRemaining++;
            }
        }

        text.innerHTML =
            '<span>Pair ' + (this.currentPairIndex + 1) + '/' + this.pairs.length + '</span>' +
            '<span>Ambiguous remaining: ' + ambiguousRemaining + '</span>';

        progressWrapper.appendChild(track);
        progressWrapper.appendChild(text);
        container.appendChild(progressWrapper);

        // -- Crops side by side --
        var cropsRow = document.createElement('div');
        cropsRow.className = 'reid-crops';

        var cropImgA = document.createElement('img');
        cropImgA.src = '/interview/api/detect/crop/' + cropA.crop_id +
            '/image?session_id=' + encodeURIComponent(this.sessionId);
        cropImgA.alt = 'Crop A';
        cropImgA.style.border = '3px solid #3498db';

        var cropImgB = document.createElement('img');
        cropImgB.src = '/interview/api/detect/crop/' + cropB.crop_id +
            '/image?session_id=' + encodeURIComponent(this.sessionId);
        cropImgB.alt = 'Crop B';
        cropImgB.style.border = '3px solid #e67e22';

        cropsRow.appendChild(cropImgA);
        cropsRow.appendChild(cropImgB);
        container.appendChild(cropsRow);

        // -- Pool indicator (subtle) --
        var poolIndicator = document.createElement('div');
        poolIndicator.style.display = 'flex';
        poolIndicator.style.alignItems = 'center';
        poolIndicator.style.justifyContent = 'center';
        poolIndicator.style.gap = '6px';
        poolIndicator.style.padding = '4px 0';

        var dot = document.createElement('span');
        dot.style.display = 'inline-block';
        dot.style.width = '8px';
        dot.style.height = '8px';
        dot.style.borderRadius = '50%';

        var poolLabel = document.createElement('span');
        poolLabel.style.fontSize = '0.65rem';
        poolLabel.style.color = 'var(--text-muted)';

        if (pair.pool === 'ambiguous') {
            dot.style.background = '#3498db';
            poolLabel.textContent = 'ambiguous';
        } else if (pair.pool === 'confident_same') {
            dot.style.background = '#2ecc71';
            poolLabel.textContent = 'calibration (likely same)';
        } else {
            dot.style.background = '#e67e22';
            poolLabel.textContent = 'calibration (likely different)';
        }

        poolIndicator.appendChild(dot);
        poolIndicator.appendChild(poolLabel);
        container.appendChild(poolIndicator);

        // -- Similarity score --
        if (typeof pair.similarity === 'number') {
            var simRow = document.createElement('div');
            simRow.style.textAlign = 'center';
            simRow.style.fontSize = '0.75rem';
            simRow.style.color = 'var(--text-secondary)';
            simRow.textContent = 'Similarity: ' + pair.similarity.toFixed(3);
            container.appendChild(simRow);
        }

        // -- Verdict buttons --
        var verdictRow = document.createElement('div');
        verdictRow.className = 'reid-verdict';

        var btnYes = document.createElement('button');
        btnYes.className = 'btn btn-accept';
        btnYes.innerHTML = 'Yes &mdash; Same Person';
        btnYes.addEventListener('click', function () {
            this._resolvePair('same');
        }.bind(this));

        var btnNo = document.createElement('button');
        btnNo.className = 'btn btn-reject';
        btnNo.innerHTML = 'No &mdash; Different';
        btnNo.addEventListener('click', function () {
            this._resolvePair('different');
        }.bind(this));

        var btnUnsure = document.createElement('button');
        btnUnsure.className = 'btn btn-unsure';
        btnUnsure.textContent = 'Unsure';
        btnUnsure.addEventListener('click', function () {
            this._resolvePair('unsure');
        }.bind(this));

        verdictRow.appendChild(btnYes);
        verdictRow.appendChild(btnNo);
        verdictRow.appendChild(btnUnsure);
        container.appendChild(verdictRow);

        // -- Keyboard hints --
        var hints = document.createElement('div');
        hints.className = 'keyboard-hints';
        hints.innerHTML =
            '<kbd>Y</kbd> Same ' +
            '<kbd>N</kbd> Different ' +
            '<kbd>U</kbd> Unsure ' +
            '<kbd>&larr;</kbd> Previous';
        container.appendChild(hints);

        // -- Merge status panel --
        this._renderMergeStatus(container);
    }

    // ------------------------------------------------------------------
    // Resolve a pair
    // ------------------------------------------------------------------

    async _resolvePair(resolution) {
        var pair = this.pairs[this.currentPairIndex];
        if (!pair) return;

        this.resolutions[pair.pair_id] = resolution;

        // Submit to backend
        var resolutionPayload = {};
        resolutionPayload[pair.pair_id] = resolution;

        try {
            var result = await API.post('/reid/resolve', {
                session_id: this.sessionId,
                resolutions: resolutionPayload,
            });

            if (result.n_identities !== undefined) {
                this.nIdentities = result.n_identities;
            }
            if (result.clusters) {
                this.clusters = result.clusters;
            }
        } catch (err) {
            showToast('Failed to submit resolution: ' + err.message, 'error');
            // Keep the local resolution anyway to not block the user
        }

        // Move to next unresolved pair
        var nextIndex = this._findNextUnresolved(this.currentPairIndex + 1);
        if (nextIndex !== -1) {
            this._showPair(nextIndex);
        } else {
            // All pairs resolved
            this._renderSummary();
        }
    }

    _findNextUnresolved(startIndex) {
        for (var i = startIndex; i < this.pairs.length; i++) {
            if (!this.resolutions[this.pairs[i].pair_id]) {
                return i;
            }
        }
        // Wrap around to check before startIndex
        for (var j = 0; j < startIndex && j < this.pairs.length; j++) {
            if (!this.resolutions[this.pairs[j].pair_id]) {
                return j;
            }
        }
        return -1;
    }

    // ------------------------------------------------------------------
    // Merge status panel
    // ------------------------------------------------------------------

    _renderMergeStatus(container) {
        var statusPanel = document.createElement('div');
        statusPanel.style.flex = '1';
        statusPanel.style.overflowY = 'auto';
        statusPanel.style.padding = '12px 0';
        statusPanel.style.borderTop = '1px solid var(--border-default)';

        var heading = document.createElement('div');
        heading.style.fontSize = '0.75rem';
        heading.style.fontWeight = '600';
        heading.style.color = 'var(--text-secondary)';
        heading.style.marginBottom = '8px';
        heading.textContent = 'Merge Evidence';
        statusPanel.appendChild(heading);

        // Build a map of cluster-pair evidence from resolved pairs
        var clusterPairEvidence = {};
        for (var i = 0; i < this.pairs.length; i++) {
            var p = this.pairs[i];
            var res = this.resolutions[p.pair_id];
            if (!res) continue;

            var key = Math.min(p.cluster_a, p.cluster_b) + '-' +
                      Math.max(p.cluster_a, p.cluster_b);

            if (!clusterPairEvidence[key]) {
                clusterPairEvidence[key] = {
                    a: Math.min(p.cluster_a, p.cluster_b),
                    b: Math.max(p.cluster_a, p.cluster_b),
                    same: 0,
                    different: 0,
                    unsure: 0,
                };
            }
            clusterPairEvidence[key][res]++;
        }

        var keys = Object.keys(clusterPairEvidence);
        if (keys.length === 0) {
            var noEvidence = document.createElement('div');
            noEvidence.style.fontSize = '0.7rem';
            noEvidence.style.color = 'var(--text-muted)';
            noEvidence.textContent = 'No pair evidence yet. Resolve pairs to see merge status.';
            statusPanel.appendChild(noEvidence);
        } else {
            for (var k = 0; k < keys.length; k++) {
                var ev = clusterPairEvidence[keys[k]];
                var row = document.createElement('div');
                row.style.display = 'flex';
                row.style.alignItems = 'center';
                row.style.gap = '8px';
                row.style.marginBottom = '4px';
                row.style.fontSize = '0.7rem';
                row.style.color = 'var(--text-primary)';

                var label = 'Cluster ' + ev.a + ' <-> ' + ev.b + ': ';

                if (ev.different > 0) {
                    label += 'VETOED (different)';
                    row.style.color = 'var(--color-rejected)';
                } else if (ev.same >= 2) {
                    label += 'MERGED (' + ev.same + ' confirmations)';
                    row.style.color = 'var(--color-accepted)';
                } else {
                    label += ev.same + ' same, ' + ev.unsure + ' unsure';
                }

                row.textContent = label;
                statusPanel.appendChild(row);
            }
        }

        container.appendChild(statusPanel);
    }

    // ------------------------------------------------------------------
    // Summary (all pairs resolved)
    // ------------------------------------------------------------------

    _renderSummary() {
        this.container.innerHTML = '';

        var panel = document.createElement('div');
        panel.className = 'seeding-panel';
        panel.style.marginTop = '40px';

        var box = document.createElement('div');
        box.className = 'session-setup';
        box.style.maxWidth = '600px';

        var h2 = document.createElement('h2');
        h2.textContent = 'ReID Complete';
        box.appendChild(h2);

        var resolved = Object.keys(this.resolutions).length;

        var stats = document.createElement('div');
        stats.style.marginBottom = '20px';
        stats.innerHTML =
            '<p style="margin-bottom:8px;color:var(--text-secondary)">' +
            'Resolved <strong>' + resolved + '</strong> of ' +
            '<strong>' + this.pairs.length + '</strong> pairs.' +
            '</p>' +
            '<p style="margin-bottom:8px;color:var(--text-secondary)">' +
            'Final identity count: <strong>' + this.nIdentities + '</strong>' +
            '</p>';
        box.appendChild(stats);

        // Cluster summary table
        var clusterKeys = Object.keys(this.clusters);
        if (clusterKeys.length > 0) {
            var table = document.createElement('table');
            table.className = 'seed-table';
            table.style.marginBottom = '20px';
            table.innerHTML =
                '<thead><tr>' +
                '<th>Identity</th>' +
                '<th>Crops</th>' +
                '</tr></thead>';

            var tbody = document.createElement('tbody');
            for (var c = 0; c < clusterKeys.length; c++) {
                var cluster = this.clusters[clusterKeys[c]];
                var tr = document.createElement('tr');
                tr.innerHTML =
                    '<td>Cluster ' + clusterKeys[c] + '</td>' +
                    '<td>' + cluster.count + '</td>';
                tbody.appendChild(tr);
            }
            table.appendChild(tbody);
            box.appendChild(table);
        }

        // Action buttons
        var actions = document.createElement('div');
        actions.className = 'session-actions';

        var btnProceed = document.createElement('button');
        btnProceed.className = 'btn btn-primary';
        btnProceed.textContent = 'Proceed to Seeding';
        btnProceed.addEventListener('click', function () {
            if (typeof this.onComplete === 'function') {
                this.onComplete({
                    n_identities: this.nIdentities,
                    resolutions: this.resolutions,
                });
            }
        }.bind(this));
        actions.appendChild(btnProceed);

        box.appendChild(actions);
        panel.appendChild(box);
        this.container.appendChild(panel);

        this.destroy();
    }

    // ------------------------------------------------------------------
    // Keyboard handler
    // ------------------------------------------------------------------

    _handleKeyDown(e) {
        // Ignore if user is typing in an input
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

        var key = e.key.toLowerCase();
        if (key === 'y') {
            e.preventDefault();
            this._resolvePair('same');
        } else if (key === 'n') {
            e.preventDefault();
            this._resolvePair('different');
        } else if (key === 'u') {
            e.preventDefault();
            this._resolvePair('unsure');
        } else if (key === 'arrowleft') {
            e.preventDefault();
            this._goBack();
        }
    }

    _goBack() {
        if (this.currentPairIndex > 0) {
            // Find previous pair (even if resolved, let user revisit)
            this._showPair(this.currentPairIndex - 1);
        }
    }
}

/**
 * Global bridge function called by app.js renderReID().
 * Instantiates ReIDUI, inits with current session, and wires
 * onComplete to navigate to the seeding phase.
 */
function renderReIDPhase(container) {
    var ui = new ReIDUI(container);
    // Register for cleanup on navigation (prevents keydown listener leak)
    if (typeof AppState !== 'undefined' && AppState._components) {
        AppState._components.reidUI = ui;
    }
    ui.onComplete = function () {
        if (typeof navigate === 'function') {
            navigate('seeding');
        }
    };
    ui.init(AppState.sessionId);
}
