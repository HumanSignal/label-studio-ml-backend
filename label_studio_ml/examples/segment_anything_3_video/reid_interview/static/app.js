/**
 * ReID Interview — Main SPA
 *
 * Phases: landing → loading → reviewing → writeback → complete
 * API base: /ReID-Interview/api
 */

/* global SwipeUI, ClusterMap, Confetti */

// ===========================================================================
// API Client
// ===========================================================================

var API = {
    base: '/ReID-Interview/api',

    post: function (path, body) {
        return fetch(API.base + path, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        }).then(function (r) {
            if (!r.ok) return r.json().then(function (d) { throw new Error(d.error || r.statusText); });
            return r.json();
        });
    },

    get: function (path, params) {
        var url = API.base + path;
        if (params) {
            var qs = Object.keys(params).map(function (k) {
                return encodeURIComponent(k) + '=' + encodeURIComponent(params[k]);
            }).join('&');
            url += '?' + qs;
        }
        return fetch(url).then(function (r) {
            if (!r.ok) return r.json().then(function (d) { throw new Error(d.error || r.statusText); });
            return r.json();
        });
    },
};

// ===========================================================================
// App State
// ===========================================================================

var AppState = {
    phase: 'landing',
    sessionId: null,
    projectId: null,
    taskId: null,
    annotationId: null,
    accuracy: null,
};

// ===========================================================================
// Toast
// ===========================================================================

function showToast(message, type) {
    type = type || 'info';
    var container = document.getElementById('toast-container');
    var toast = document.createElement('div');
    toast.className = 'toast ' + type;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(function () {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        toast.style.transition = 'all 0.3s';
        setTimeout(function () { toast.remove(); }, 300);
    }, 3000);
}

// ===========================================================================
// Phase Indicator
// ===========================================================================

function updatePhaseIndicator(phase) {
    var dots = document.querySelectorAll('.phase-dot');
    var phaseOrder = ['load', 'review', 'writeback'];
    var currentIdx = -1;

    if (phase === 'landing' || phase === 'loading') currentIdx = 0;
    else if (phase === 'reviewing') currentIdx = 1;
    else if (phase === 'writeback') currentIdx = 2;
    else if (phase === 'complete') currentIdx = 3;

    for (var i = 0; i < dots.length; i++) {
        dots[i].classList.remove('active', 'completed');
        if (i < currentIdx) dots[i].classList.add('completed');
        else if (i === currentIdx) dots[i].classList.add('active');
    }
}

// ===========================================================================
// Job Polling
// ===========================================================================

function pollJob(jobId, onProgress, onComplete, onError) {
    var poll = function () {
        API.get('/job/' + jobId + '/progress').then(function (data) {
            if (onProgress) onProgress(data);
            if (data.status === 'completed') {
                if (onComplete) onComplete(data);
            } else if (data.status === 'failed') {
                if (onError) onError(data.error || 'Job failed');
            } else {
                setTimeout(poll, 800);
            }
        }).catch(function (err) {
            if (onError) onError(err.message);
        });
    };
    poll();
}

// ===========================================================================
// Render: Landing Page
// ===========================================================================

function renderLanding() {
    AppState.phase = 'landing';
    updatePhaseIndicator('landing');

    var app = document.getElementById('app');
    app.innerHTML = '';

    var page = document.createElement('div');
    page.className = 'landing-page';

    var card = document.createElement('div');
    card.className = 'landing-card';

    card.innerHTML =
        '<h2>Person Re-Identification</h2>' +
        '<p>Load an existing Label Studio annotation to identify and resolve person tracks across video frames.</p>' +
        '<div class="form-group">' +
            '<label for="project-id">Project ID *</label>' +
            '<input type="number" id="project-id" placeholder="e.g. 1" min="1" required>' +
        '</div>' +
        '<div class="form-group">' +
            '<label for="task-id">Task ID *</label>' +
            '<input type="number" id="task-id" placeholder="e.g. 42" min="1" required>' +
        '</div>' +
        '<div class="form-group">' +
            '<label for="annotation-id">Annotation ID *</label>' +
            '<input type="number" id="annotation-id" placeholder="e.g. 100" min="1" required>' +
        '</div>' +
        '<button class="btn btn-primary" id="btn-load">Load &amp; Analyze</button>' +
        '<div class="progress-wrapper" id="load-progress" style="display:none">' +
            '<div class="progress-label" id="progress-label">Initializing...</div>' +
            '<div class="progress-track"><div class="progress-fill" id="progress-fill"></div></div>' +
        '</div>';

    page.appendChild(card);
    app.appendChild(page);

    document.getElementById('btn-load').addEventListener('click', handleLoad);

    // Allow Enter key to submit
    card.addEventListener('keydown', function (e) {
        if (e.key === 'Enter') handleLoad();
    });
}

function handleLoad() {
    var projectId = document.getElementById('project-id').value;
    var taskId = document.getElementById('task-id').value;
    var annotationId = document.getElementById('annotation-id').value;

    if (!projectId || !taskId || !annotationId) {
        showToast('All three fields are required.', 'error');
        return;
    }

    AppState.projectId = parseInt(projectId);
    AppState.taskId = parseInt(taskId);
    AppState.annotationId = parseInt(annotationId);

    var btn = document.getElementById('btn-load');
    btn.disabled = true;
    btn.textContent = 'Loading...';

    var progressDiv = document.getElementById('load-progress');
    progressDiv.style.display = 'block';

    // Step 1: Init session
    API.post('/session/init', {
        project_id: AppState.projectId,
        task_id: AppState.taskId,
        annotation_id: AppState.annotationId,
    }).then(function (data) {
        AppState.sessionId = data.session_id;
        updatePhaseIndicator('loading');

        // Step 2: Start pipeline
        return API.post('/load', { session_id: AppState.sessionId });
    }).then(function (data) {
        // Step 3: Poll job
        pollJob(data.job_id,
            function onProgress(p) {
                var label = document.getElementById('progress-label');
                var fill = document.getElementById('progress-fill');
                if (label) label.textContent = p.step || 'Processing...';
                if (fill) fill.style.width = (p.percent || 0) + '%';
            },
            function onComplete() {
                showToast('Analysis complete! Starting review.', 'success');
                renderReview();
            },
            function onError(err) {
                showToast('Pipeline failed: ' + err, 'error');
                btn.disabled = false;
                btn.textContent = 'Load & Analyze';
            }
        );
    }).catch(function (err) {
        showToast('Error: ' + err.message, 'error');
        btn.disabled = false;
        btn.textContent = 'Load & Analyze';
    });
}

// ===========================================================================
// Render: Review Phase (Swipe + Cluster Map)
// ===========================================================================

function renderReview() {
    AppState.phase = 'reviewing';
    updatePhaseIndicator('reviewing');

    var badge = document.getElementById('accuracy-badge');
    badge.style.display = 'block';

    var app = document.getElementById('app');
    app.innerHTML = '';

    var layout = document.createElement('div');
    layout.className = 'review-layout';

    var swipePanel = document.createElement('div');
    swipePanel.className = 'swipe-panel';
    swipePanel.id = 'swipe-panel';

    var clusterPanel = document.createElement('div');
    clusterPanel.className = 'cluster-panel';

    var clusterHeader = document.createElement('div');
    clusterHeader.className = 'cluster-panel-header';
    clusterHeader.textContent = 'Identity Clusters';

    var clusterMapContainer = document.createElement('div');
    clusterMapContainer.className = 'cluster-map-container';
    clusterMapContainer.id = 'cluster-map-container';

    var clusterStats = document.createElement('div');
    clusterStats.className = 'cluster-stats';
    clusterStats.id = 'cluster-stats';

    clusterPanel.appendChild(clusterHeader);
    clusterPanel.appendChild(clusterMapContainer);
    clusterPanel.appendChild(clusterStats);

    layout.appendChild(swipePanel);
    layout.appendChild(clusterPanel);
    app.appendChild(layout);

    // Init swipe UI
    if (typeof SwipeUI !== 'undefined') {
        SwipeUI.init('swipe-panel', AppState.sessionId, function onAllResolved() {
            renderWriteback();
        });
    }

    // Init cluster map
    if (typeof ClusterMap !== 'undefined') {
        ClusterMap.init('cluster-map-container', AppState.sessionId);
    }

    // Update cluster stats
    updateClusterStats();
}

function updateClusterStats() {
    if (!AppState.sessionId) return;
    API.get('/clusters', { session_id: AppState.sessionId }).then(function (data) {
        var stats = document.getElementById('cluster-stats');
        if (!stats) return;
        stats.innerHTML =
            '<div class="cluster-stats-grid">' +
                '<div class="stat-item">' +
                    '<div class="stat-value">' + data.n_clusters + '</div>' +
                    '<div class="stat-label">Identities</div>' +
                '</div>' +
                '<div class="stat-item">' +
                    '<div class="stat-value">' + data.nodes.length + '</div>' +
                    '<div class="stat-label">Crops</div>' +
                '</div>' +
            '</div>';
    });
}

function updateAccuracyBadge(accuracy) {
    var el = document.getElementById('accuracy-value');
    if (el) el.textContent = accuracy;
}

// ===========================================================================
// Render: Write-back Phase
// ===========================================================================

function renderWriteback() {
    AppState.phase = 'writeback';
    updatePhaseIndicator('writeback');

    var app = document.getElementById('app');
    app.innerHTML = '';

    var page = document.createElement('div');
    page.className = 'writeback-page';

    var card = document.createElement('div');
    card.className = 'writeback-card';
    card.innerHTML =
        '<h2>Write-Back to Label Studio</h2>' +
        '<p style="color:var(--text-secondary);margin-bottom:16px">Review planned changes before applying them to your annotation.</p>' +
        '<div id="mutation-list-container">Loading preview...</div>' +
        '<div class="writeback-toggle" id="mode-toggle">' +
            '<button class="toggle-btn active" data-mode="prediction">Create Prediction</button>' +
            '<button class="toggle-btn" data-mode="update">Update Annotation</button>' +
        '</div>' +
        '<div class="writeback-actions">' +
            '<button class="btn btn-primary" id="btn-writeback">Apply Changes</button>' +
        '</div>' +
        '<div class="progress-wrapper" id="wb-progress" style="display:none">' +
            '<div class="progress-label" id="wb-progress-label">Writing...</div>' +
            '<div class="progress-track"><div class="progress-fill" id="wb-progress-fill"></div></div>' +
        '</div>';

    page.appendChild(card);
    app.appendChild(page);

    var writebackMode = 'prediction';

    // Mode toggle
    var toggleBtns = document.querySelectorAll('#mode-toggle .toggle-btn');
    for (var i = 0; i < toggleBtns.length; i++) {
        toggleBtns[i].addEventListener('click', function () {
            for (var j = 0; j < toggleBtns.length; j++) toggleBtns[j].classList.remove('active');
            this.classList.add('active');
            writebackMode = this.getAttribute('data-mode');
        });
    }

    // Load preview
    API.get('/writeback/preview', { session_id: AppState.sessionId }).then(function (data) {
        var container = document.getElementById('mutation-list-container');
        if (data.mutations.length === 0) {
            container.innerHTML = '<p style="color:var(--text-muted)">No structural changes needed. Only ID labels will be applied.</p>';
            return;
        }
        var ul = document.createElement('ul');
        ul.className = 'mutation-list';
        for (var k = 0; k < data.mutations.length; k++) {
            var m = data.mutations[k];
            var li = document.createElement('li');
            li.className = 'mutation-item';
            var iconClass = 'label';
            var iconChar = 'L';
            if (m.type === 'fragment_merge') { iconClass = 'merge'; iconChar = 'M'; }
            else if (m.type === 'track_split') { iconClass = 'split'; iconChar = 'S'; }
            else if (m.type === 'id_swap') { iconClass = 'swap'; iconChar = 'X'; }
            else if (m.type === 'keyframe_move') { iconClass = 'move'; iconChar = 'K'; }
            li.innerHTML =
                '<span class="mutation-icon ' + iconClass + '">' + iconChar + '</span>' +
                '<span>' + m.description + '</span>';
            ul.appendChild(li);
        }
        container.innerHTML = '';
        container.appendChild(ul);
    }).catch(function (err) {
        document.getElementById('mutation-list-container').innerHTML =
            '<p style="color:var(--color-rejected)">Failed to load preview: ' + err.message + '</p>';
    });

    // Writeback button
    document.getElementById('btn-writeback').addEventListener('click', function () {
        var btn = this;
        btn.disabled = true;
        btn.textContent = 'Applying...';
        var progressDiv = document.getElementById('wb-progress');
        progressDiv.style.display = 'block';

        API.post('/writeback', {
            session_id: AppState.sessionId,
            mode: writebackMode,
        }).then(function (data) {
            pollJob(data.job_id,
                function onProgress(p) {
                    var label = document.getElementById('wb-progress-label');
                    var fill = document.getElementById('wb-progress-fill');
                    if (label) label.textContent = p.step || 'Writing...';
                    if (fill) fill.style.width = (p.percent || 0) + '%';
                },
                function onComplete() {
                    showToast('Write-back complete!', 'success');
                    renderComplete();
                },
                function onError(err) {
                    showToast('Write-back failed: ' + err, 'error');
                    btn.disabled = false;
                    btn.textContent = 'Apply Changes';
                }
            );
        }).catch(function (err) {
            showToast('Error: ' + err.message, 'error');
            btn.disabled = false;
            btn.textContent = 'Apply Changes';
        });
    });
}

// ===========================================================================
// Render: Complete
// ===========================================================================

function renderComplete() {
    AppState.phase = 'complete';
    updatePhaseIndicator('complete');

    var app = document.getElementById('app');
    app.innerHTML = '';

    var page = document.createElement('div');
    page.className = 'complete-page';
    page.innerHTML =
        '<div class="complete-icon">&#10003;</div>' +
        '<h2>ReID Complete</h2>' +
        '<p style="color:var(--text-secondary)">Identity labels have been applied to your Label Studio annotation.</p>' +
        '<button class="btn btn-primary" style="max-width:200px" onclick="renderLanding()">Start New Session</button>';
    app.appendChild(page);

    // Victory confetti
    if (typeof Confetti !== 'undefined') {
        Confetti.burst(window.innerWidth / 2, window.innerHeight / 3, 80);
    }
}

// ===========================================================================
// Init
// ===========================================================================

document.addEventListener('DOMContentLoaded', renderLanding);
