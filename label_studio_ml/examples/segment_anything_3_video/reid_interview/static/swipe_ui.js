/**
 * SwipeUI — Tinder-style swipe card interface for person ReID.
 *
 * Controls: F=Same(accept), J=Different(reject), Space=Unsure
 *           ArrowLeft=Previous, ArrowRight=Skip/Next
 *
 * Supports touch/mouse drag with spring snap-back.
 */

/* global API, AppState, showToast, updateAccuracyBadge, updateClusterStats, Confetti, ClusterMap */

var SwipeUI = (function () {
    'use strict';

    var _container = null;
    var _sessionId = null;
    var _pairs = [];
    var _currentIndex = 0;
    var _onAllResolved = null;
    var _resolved = {};
    var _totalPairs = 0;
    var _resolvedCount = 0;

    // Drag state
    var _dragging = false;
    var _startX = 0;
    var _currentX = 0;
    var _cardEl = null;
    var _threshold = 100; // px to trigger swipe

    // ------------------------------------------------------------------
    // Init
    // ------------------------------------------------------------------

    function init(containerId, sessionId, onAllResolved) {
        _container = document.getElementById(containerId);
        _sessionId = sessionId;
        _onAllResolved = onAllResolved;
        _resolved = {};
        _currentIndex = 0;

        document.addEventListener('keydown', _handleKeyDown);
        _loadPairs();
    }

    function destroy() {
        document.removeEventListener('keydown', _handleKeyDown);
    }

    // ------------------------------------------------------------------
    // Load pairs
    // ------------------------------------------------------------------

    function _loadPairs() {
        API.get('/pairs/next', {
            session_id: _sessionId,
            batch_size: 50,
        }).then(function (data) {
            _pairs = data.pairs || [];
            _totalPairs = data.total_pairs || 0;
            _resolvedCount = data.resolved_count || 0;

            if (_pairs.length === 0) {
                _renderAllDone();
                return;
            }
            _showPair(0);
        }).catch(function (err) {
            showToast('Failed to load pairs: ' + err.message, 'error');
        });
    }

    // ------------------------------------------------------------------
    // Render pair
    // ------------------------------------------------------------------

    function _showPair(index) {
        if (index < 0 || index >= _pairs.length) return;
        _currentIndex = index;
        var pair = _pairs[index];

        _container.innerHTML = '';

        // Progress bar
        var progress = document.createElement('div');
        progress.className = 'swipe-progress';
        var pct = _totalPairs > 0 ? ((_resolvedCount / _totalPairs) * 100) : 0;
        progress.innerHTML =
            '<div class="swipe-progress-bar">' +
                '<div class="swipe-progress-track">' +
                    '<div class="swipe-progress-fill" style="width:' + pct + '%"></div>' +
                '</div>' +
                '<span class="swipe-progress-text">' +
                    _resolvedCount + '/' + _totalPairs +
                '</span>' +
            '</div>';
        _container.appendChild(progress);

        // Card container
        var cardContainer = document.createElement('div');
        cardContainer.className = 'swipe-card-container';

        // Card
        var card = document.createElement('div');
        card.className = 'swipe-card';
        _cardEl = card;

        // Swipe labels
        var labelSame = document.createElement('div');
        labelSame.className = 'swipe-label same';
        labelSame.textContent = 'SAME';
        labelSame.id = 'swipe-label-same';

        var labelDiff = document.createElement('div');
        labelDiff.className = 'swipe-label diff';
        labelDiff.textContent = 'DIFF';
        labelDiff.id = 'swipe-label-diff';

        card.appendChild(labelSame);
        card.appendChild(labelDiff);

        // Crops comparison
        var crops = document.createElement('div');
        crops.className = 'crops-comparison';
        crops.innerHTML =
            '<div class="crop-cell">' +
                '<img src="' + API.base + '/crop/' + pair.crop_id_a + '/image?session_id=' + _sessionId + '" alt="Crop A">' +
                '<div class="crop-badge">A &mdash; ' + pair.track_a + '</div>' +
            '</div>' +
            '<div class="crop-cell">' +
                '<img src="' + API.base + '/crop/' + pair.crop_id_b + '/image?session_id=' + _sessionId + '" alt="Crop B">' +
                '<div class="crop-badge">B &mdash; ' + pair.track_b + '</div>' +
            '</div>';
        card.appendChild(crops);

        // Frame context
        var frameA = parseInt(pair.crop_id_a.split('_f').pop()) || 0;
        var frameB = parseInt(pair.crop_id_b.split('_f').pop()) || 0;
        var frames = document.createElement('div');
        frames.className = 'frames-context';
        frames.innerHTML =
            '<div class="frame-cell">' +
                '<img src="' + API.base + '/frame/' + frameA + '?session_id=' + _sessionId + '" alt="Frame ' + frameA + '">' +
                '<span class="frame-badge">Frame ' + frameA + '</span>' +
            '</div>' +
            '<div class="frame-cell">' +
                '<img src="' + API.base + '/frame/' + frameB + '?session_id=' + _sessionId + '" alt="Frame ' + frameB + '">' +
                '<span class="frame-badge">Frame ' + frameB + '</span>' +
            '</div>';
        card.appendChild(frames);

        // Pair info
        var info = document.createElement('div');
        info.className = 'pair-info';

        var simClass = pair.similarity > 0.6 ? 'color:var(--color-accepted)' :
                       pair.similarity < 0.3 ? 'color:var(--color-rejected)' : '';
        var stars = pair.difficulty === 0 ? '\u2605' :
                    pair.difficulty === 1 ? '\u2605\u2605' : '\u2605\u2605\u2605';
        var poolLabel = pair.pool === 'warmup' ? 'Warm-up' :
                        pair.pool === 'calibration' ? 'Calibration' : 'Challenge';

        info.innerHTML =
            '<span class="similarity-badge" style="' + simClass + '">' +
                'Sim: ' + pair.similarity.toFixed(3) +
            '</span>' +
            '<span>' + poolLabel + '</span>' +
            '<span class="difficulty-stars">' + stars + '</span>';
        card.appendChild(info);

        // Verdict buttons
        var verdict = document.createElement('div');
        verdict.className = 'verdict-row';
        verdict.innerHTML =
            '<button class="btn btn-reject" id="btn-different">J &mdash; Different</button>' +
            '<button class="btn btn-unsure" id="btn-unsure">Space &mdash; Unsure</button>' +
            '<button class="btn btn-accept" id="btn-same">F &mdash; Same</button>';
        card.appendChild(verdict);

        // Keyboard hints
        var hints = document.createElement('div');
        hints.className = 'keyboard-hints';
        hints.innerHTML =
            '<kbd>F</kbd> Same &nbsp; ' +
            '<kbd>J</kbd> Different &nbsp; ' +
            '<kbd>Space</kbd> Unsure &nbsp; ' +
            '<kbd>&larr;</kbd> Back &nbsp; ' +
            '<kbd>&rarr;</kbd> Skip';
        card.appendChild(hints);

        cardContainer.appendChild(card);
        _container.appendChild(cardContainer);

        // Button handlers
        document.getElementById('btn-same').addEventListener('click', function () { _resolve('same'); });
        document.getElementById('btn-different').addEventListener('click', function () { _resolve('different'); });
        document.getElementById('btn-unsure').addEventListener('click', function () { _resolve('unsure'); });

        // Drag handlers
        card.addEventListener('pointerdown', _onPointerDown);
        card.addEventListener('pointermove', _onPointerMove);
        card.addEventListener('pointerup', _onPointerUp);
        card.addEventListener('pointercancel', _onPointerUp);
    }

    // ------------------------------------------------------------------
    // Drag/Swipe handling
    // ------------------------------------------------------------------

    function _onPointerDown(e) {
        if (e.target.tagName === 'BUTTON') return;
        _dragging = true;
        _startX = e.clientX;
        _currentX = e.clientX;
        _cardEl.setPointerCapture(e.pointerId);
        _cardEl.style.transition = 'none';
    }

    function _onPointerMove(e) {
        if (!_dragging) return;
        _currentX = e.clientX;
        var dx = _currentX - _startX;
        var rotate = dx * 0.05;
        _cardEl.style.transform = 'translateX(' + dx + 'px) rotate(' + rotate + 'deg)';

        // Show swipe labels
        var sameLbl = document.getElementById('swipe-label-same');
        var diffLbl = document.getElementById('swipe-label-diff');
        if (dx > 30) {
            _cardEl.classList.add('swiping-right');
            _cardEl.classList.remove('swiping-left');
            if (sameLbl) sameLbl.style.opacity = Math.min(1, (dx - 30) / 70);
            if (diffLbl) diffLbl.style.opacity = 0;
        } else if (dx < -30) {
            _cardEl.classList.add('swiping-left');
            _cardEl.classList.remove('swiping-right');
            if (diffLbl) diffLbl.style.opacity = Math.min(1, (-dx - 30) / 70);
            if (sameLbl) sameLbl.style.opacity = 0;
        } else {
            _cardEl.classList.remove('swiping-right', 'swiping-left');
            if (sameLbl) sameLbl.style.opacity = 0;
            if (diffLbl) diffLbl.style.opacity = 0;
        }
    }

    function _onPointerUp(e) {
        if (!_dragging) return;
        _dragging = false;
        var dx = _currentX - _startX;

        _cardEl.style.transition = 'transform 0.3s ease-out, opacity 0.3s ease-out';

        if (dx > _threshold) {
            _animateOut('right');
            setTimeout(function () { _resolve('same'); }, 300);
        } else if (dx < -_threshold) {
            _animateOut('left');
            setTimeout(function () { _resolve('different'); }, 300);
        } else {
            // Snap back
            _cardEl.style.transform = '';
            _cardEl.classList.remove('swiping-right', 'swiping-left');
            var sameLbl = document.getElementById('swipe-label-same');
            var diffLbl = document.getElementById('swipe-label-diff');
            if (sameLbl) sameLbl.style.opacity = 0;
            if (diffLbl) diffLbl.style.opacity = 0;
        }
    }

    function _animateOut(direction) {
        if (!_cardEl) return;
        if (direction === 'right') _cardEl.classList.add('exit-right');
        else if (direction === 'left') _cardEl.classList.add('exit-left');
        else _cardEl.classList.add('exit-up');
    }

    // ------------------------------------------------------------------
    // Resolve pair
    // ------------------------------------------------------------------

    function _resolve(resolution) {
        var pair = _pairs[_currentIndex];
        if (!pair || _resolved[pair.pair_id]) return;
        _resolved[pair.pair_id] = resolution;

        API.post('/pairs/resolve', {
            session_id: _sessionId,
            pair_id: pair.pair_id,
            resolution: resolution,
        }).then(function (data) {
            _resolvedCount = data.resolved_count || _resolvedCount;

            // Update accuracy badge
            if (data.accuracy !== undefined && data.accuracy !== null) {
                AppState.accuracy = data.accuracy;
                updateAccuracyBadge(data.accuracy);
            }

            // Update cluster map
            if (typeof ClusterMap !== 'undefined') {
                ClusterMap.refresh();
            }
            updateClusterStats();

            // Small confetti on resolve
            if (typeof Confetti !== 'undefined') {
                var rect = _container.getBoundingClientRect();
                Confetti.burst(rect.left + rect.width / 2, rect.top + rect.height / 2, 15);
            }

            // Next pair
            var next = _findNextUnresolved(_currentIndex + 1);
            if (next !== -1) {
                _showPair(next);
            } else if (data.remaining <= 0) {
                _renderAllDone();
            } else {
                // Load more pairs
                _loadPairs();
            }
        }).catch(function (err) {
            showToast('Resolution failed: ' + err.message, 'error');
        });
    }

    function _findNextUnresolved(startIndex) {
        for (var i = startIndex; i < _pairs.length; i++) {
            if (!_resolved[_pairs[i].pair_id]) return i;
        }
        return -1;
    }

    function _renderAllDone() {
        _container.innerHTML = '';

        var done = document.createElement('div');
        done.style.textAlign = 'center';
        done.style.padding = '40px';
        done.innerHTML =
            '<h2 style="color:var(--color-accepted);margin-bottom:16px">All Pairs Resolved!</h2>' +
            '<p style="color:var(--text-secondary);margin-bottom:24px">' +
                'Resolved ' + _resolvedCount + ' of ' + _totalPairs + ' pairs.' +
            '</p>' +
            '<button class="btn btn-primary" style="max-width:300px;margin:0 auto" id="btn-proceed-writeback">' +
                'Proceed to Write-Back' +
            '</button>';
        _container.appendChild(done);

        document.getElementById('btn-proceed-writeback').addEventListener('click', function () {
            if (typeof _onAllResolved === 'function') _onAllResolved();
        });

        // Big confetti burst
        if (typeof Confetti !== 'undefined') {
            Confetti.burst(window.innerWidth / 2, window.innerHeight / 3, 60);
        }

        destroy();
    }

    // ------------------------------------------------------------------
    // Keyboard handler
    // ------------------------------------------------------------------

    function _handleKeyDown(e) {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
        if (AppState.phase !== 'reviewing') return;

        var key = e.key;

        if (key === 'f' || key === 'F') {
            e.preventDefault();
            _animateOut('right');
            setTimeout(function () { _resolve('same'); }, 200);
        } else if (key === 'j' || key === 'J') {
            e.preventDefault();
            _animateOut('left');
            setTimeout(function () { _resolve('different'); }, 200);
        } else if (key === ' ') {
            e.preventDefault();
            _animateOut('up');
            setTimeout(function () { _resolve('unsure'); }, 200);
        } else if (key === 'ArrowLeft') {
            e.preventDefault();
            _goBack();
        } else if (key === 'ArrowRight') {
            e.preventDefault();
            _goForward();
        }
    }

    function _goBack() {
        if (_currentIndex > 0) {
            _showPair(_currentIndex - 1);
        }
    }

    function _goForward() {
        var next = _findNextUnresolved(_currentIndex + 1);
        if (next !== -1) _showPair(next);
    }

    return {
        init: init,
        destroy: destroy,
    };
})();
