/* ==========================================================================
   SAM3 Interview UI - Main Application
   Vanilla JS SPA: hash-based routing, API client, state management,
   phase renderers, keyboard shortcuts, job polling, toast notifications.
   No frameworks, no imports -- everything attaches to the global scope.
   ========================================================================== */

'use strict';

// ---------------------------------------------------------------------------
// API Client
// ---------------------------------------------------------------------------

const API = {
    base: '/interview/api',

    /**
     * POST JSON to an API endpoint.
     * @param {string} path - Relative path (e.g. '/session/init').
     * @param {Object} data - JSON body.
     * @returns {Promise<Object>} Parsed JSON response.
     */
    async post(path, data) {
        try {
            const res = await fetch(`${this.base}${path}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data),
            });
            const json = await res.json();
            if (!res.ok) {
                const msg = json.error || `Request failed (${res.status})`;
                showToast(msg, 'error');
                throw new Error(msg);
            }
            return json;
        } catch (err) {
            if (!err.message.includes('Request failed')) {
                showToast(`Network error: ${err.message}`, 'error');
            }
            throw err;
        }
    },

    /**
     * GET from an API endpoint with optional query parameters.
     * @param {string} path - Relative path (e.g. '/detect/crops').
     * @param {Object} [params] - Query string parameters.
     * @returns {Promise<Object>} Parsed JSON response.
     */
    async get(path, params) {
        try {
            let url = `${this.base}${path}`;
            if (params) {
                const qs = new URLSearchParams(params).toString();
                url += `?${qs}`;
            }
            const res = await fetch(url);
            const json = await res.json();
            if (!res.ok) {
                const msg = json.error || `Request failed (${res.status})`;
                showToast(msg, 'error');
                throw new Error(msg);
            }
            return json;
        } catch (err) {
            if (!err.message.includes('Request failed')) {
                showToast(`Network error: ${err.message}`, 'error');
            }
            throw err;
        }
    },

    /**
     * PUT JSON to an API endpoint.
     * @param {string} path - Relative path.
     * @param {Object} data - JSON body.
     * @returns {Promise<Object>} Parsed JSON response.
     */
    async put(path, data) {
        try {
            const res = await fetch(`${this.base}${path}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data),
            });
            const json = await res.json();
            if (!res.ok) {
                const msg = json.error || `Request failed (${res.status})`;
                showToast(msg, 'error');
                throw new Error(msg);
            }
            return json;
        } catch (err) {
            if (!err.message.includes('Request failed')) {
                showToast(`Network error: ${err.message}`, 'error');
            }
            throw err;
        }
    },
};

// ---------------------------------------------------------------------------
// Application State
// ---------------------------------------------------------------------------

const AppState = {
    sessionId: null,
    phase: 'init',       // init, detection, classification, reid, seeding
    projectId: null,
    taskId: null,
    annotationId: null,

    // Video metadata
    videoWidth: 0,
    videoHeight: 0,
    framesCount: 0,
    fps: 30,
    sampledFrames: [],

    // Detection / Classification
    crops: [],
    currentCropIndex: 0,
    currentFrameIdx: 0,
    currentFrameInSampled: 0,
    drawMode: false,
    sortBy: 'uncertainty',
    filterLabel: 'all',

    // Session stats (from backend)
    stats: {},

    // Cached options from session/init
    cacheOptions: null,

    // Active components (for cleanup)
    _components: {},
};

// ---------------------------------------------------------------------------
// Toast Notifications
// ---------------------------------------------------------------------------

/**
 * Show a toast message.
 * @param {string} message - Toast text.
 * @param {string} [type='info'] - 'info', 'success', 'error', 'warning'.
 * @param {number} [duration=4000] - Auto-dismiss in ms.
 */
function showToast(message, type = 'info', duration = 4000) {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    const dismiss = () => {
        toast.classList.add('dismissing');
        setTimeout(() => {
            if (toast.parentNode) toast.parentNode.removeChild(toast);
        }, 300);
    };

    toast.addEventListener('click', dismiss);
    if (duration > 0) {
        setTimeout(dismiss, duration);
    }
}

// ---------------------------------------------------------------------------
// Phase Indicator
// ---------------------------------------------------------------------------

const PHASE_ORDER = ['init', 'detection', 'classification', 'reid', 'seeding'];

/**
 * Update the nav bar phase dots to reflect the current phase.
 * @param {string} activePhase
 */
function updatePhaseIndicator(activePhase) {
    const dots = document.querySelectorAll('.phase-dot');
    const activeIdx = PHASE_ORDER.indexOf(activePhase);

    dots.forEach((dot) => {
        const phase = dot.dataset.phase;
        const phaseIdx = PHASE_ORDER.indexOf(phase);

        dot.classList.remove('active', 'complete');

        if (phaseIdx < activeIdx) {
            dot.classList.add('complete');
        } else if (phaseIdx === activeIdx) {
            dot.classList.add('active');
        }
    });
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

/**
 * Navigate to a phase and render it.
 * @param {string} phase - Phase name.
 */
function navigate(phase) {
    window.location.hash = `/${phase}`;
    AppState.phase = phase;
    renderPhase(phase);
    updatePhaseIndicator(phase);
}

/**
 * Render the appropriate phase into #app.
 * @param {string} phase
 */
function renderPhase(phase) {
    // Cleanup existing components
    _cleanupComponents();

    const app = document.getElementById('app');
    app.innerHTML = '';

    switch (phase) {
        case 'init':
        case 'setup':
            renderSetup(app);
            break;
        case 'detection':
        case 'classification':
            renderDetection(app);
            break;
        case 'reid':
            renderReID(app);
            break;
        case 'seeding':
            renderSeeding(app);
            break;
        default:
            renderSetup(app);
            break;
    }
}

/** Destroy all tracked component instances. */
function _cleanupComponents() {
    for (const key of Object.keys(AppState._components)) {
        const comp = AppState._components[key];
        if (comp && typeof comp.destroy === 'function') {
            comp.destroy();
        }
        delete AppState._components[key];
    }
}

/**
 * Handle hash-change events for browser back/forward navigation.
 */
function _onHashChange() {
    const hash = window.location.hash.replace(/^#\/?/, '') || 'setup';
    // Only re-render if phase actually changed
    if (hash !== AppState.phase) {
        AppState.phase = hash;
        renderPhase(hash);
        updatePhaseIndicator(hash);
    }
}

// ---------------------------------------------------------------------------
// Job Polling
// ---------------------------------------------------------------------------

/**
 * Poll a background job until it completes or fails.
 * @param {string} jobId
 * @param {Function} onProgress - Called with progress object on each poll.
 * @param {Function} onComplete - Called with final progress when done.
 * @param {number} [interval=500] - Polling interval in ms.
 * @returns {Function} A cancel function to stop polling.
 */
function pollJob(jobId, onProgress, onComplete, interval = 500) {
    let cancelled = false;

    const poll = setInterval(async () => {
        if (cancelled) {
            clearInterval(poll);
            return;
        }
        try {
            const progress = await API.get(`/job/${jobId}/progress`);
            if (onProgress) onProgress(progress);

            if (progress.status === 'completed' || progress.status === 'failed') {
                clearInterval(poll);
                if (onComplete) onComplete(progress);
            }
        } catch (err) {
            // Silently retry on transient errors
            console.warn('Poll error for job', jobId, err);
        }
    }, interval);

    return () => {
        cancelled = true;
        clearInterval(poll);
    };
}

// ---------------------------------------------------------------------------
// Phase Renderer: Setup
// ---------------------------------------------------------------------------

/**
 * Render the session setup form (init phase).
 * @param {HTMLElement} app
 */
function renderSetup(app) {
    const wrap = document.createElement('div');
    wrap.className = 'session-setup';

    wrap.innerHTML = `
        <h2>Session Setup</h2>
        <div class="form-group">
            <label for="setup-project-id">Project ID</label>
            <input type="number" id="setup-project-id" placeholder="e.g. 42"
                   value="${AppState.projectId || ''}">
        </div>
        <div class="form-group">
            <label for="setup-task-id">Task ID</label>
            <input type="number" id="setup-task-id" placeholder="e.g. 101"
                   value="${AppState.taskId || ''}">
        </div>
        <div class="form-group">
            <label for="setup-annotation-id">Annotation ID (optional)</label>
            <input type="number" id="setup-annotation-id" placeholder="Leave blank for new"
                   value="${AppState.annotationId || ''}">
        </div>
        <div class="form-group">
            <label for="setup-prompt">Detection Prompt</label>
            <input type="text" id="setup-prompt" placeholder="e.g. person"
                   value="person">
        </div>
        <div id="setup-cache-info" class="hidden" style="margin-bottom:16px;
             font-size:0.8rem;color:var(--text-secondary);"></div>
        <div id="setup-actions" class="session-actions">
            <button id="setup-check-btn" class="btn btn-secondary">Check Cache</button>
        </div>
        <div id="setup-progress" class="hidden" style="margin-top:16px;"></div>
    `;

    app.appendChild(wrap);

    // Bind check-cache button
    document.getElementById('setup-check-btn').addEventListener('click', _onCheckCache);
}

/** Handle "Check Cache" -- calls session/init to discover cache options. */
async function _onCheckCache() {
    const projectId = document.getElementById('setup-project-id').value.trim();
    const taskId = document.getElementById('setup-task-id').value.trim();
    const annotationId = document.getElementById('setup-annotation-id').value.trim();

    if (!projectId || !taskId) {
        showToast('Project ID and Task ID are required', 'warning');
        return;
    }

    AppState.projectId = parseInt(projectId, 10);
    AppState.taskId = parseInt(taskId, 10);
    AppState.annotationId = annotationId ? parseInt(annotationId, 10) : null;

    try {
        const result = await API.post('/session/init', {
            project_id: AppState.projectId,
            task_id: AppState.taskId,
            annotation_id: AppState.annotationId,
        });

        AppState.cacheOptions = result;
        _renderCacheOptions(result);
    } catch (err) {
        // Error already toasted by API client
    }
}

/**
 * Show cache options (Resume / Build On / Fresh) and start buttons.
 * @param {Object} result - Response from session/init.
 */
function _renderCacheOptions(result) {
    const infoEl = document.getElementById('setup-cache-info');
    const actionsEl = document.getElementById('setup-actions');

    infoEl.classList.remove('hidden');
    actionsEl.innerHTML = '';

    if (result.has_cache) {
        infoEl.textContent = 'Existing cache found for this task.';

        const resumeBtn = document.createElement('button');
        resumeBtn.className = 'btn btn-secondary';
        resumeBtn.textContent = 'Resume';
        resumeBtn.addEventListener('click', () => _startSession('resume'));
        actionsEl.appendChild(resumeBtn);

        const buildBtn = document.createElement('button');
        buildBtn.className = 'btn btn-secondary';
        buildBtn.textContent = 'Build On';
        buildBtn.addEventListener('click', () => _startSession('build_on'));
        actionsEl.appendChild(buildBtn);

        const freshBtn = document.createElement('button');
        freshBtn.className = 'btn btn-primary';
        freshBtn.textContent = 'Fresh Start';
        freshBtn.addEventListener('click', async () => {
            const confirmed = await Modal.confirm(
                'Fresh Start',
                'This will delete the existing cache. Continue?'
            );
            if (confirmed) _startSession('fresh');
        });
        actionsEl.appendChild(freshBtn);
    } else if (result.other_caches && result.other_caches.length > 0) {
        infoEl.textContent = 'Other task caches found in this project.';

        result.other_caches.forEach((cache) => {
            const btn = document.createElement('button');
            btn.className = 'btn btn-secondary';
            btn.textContent = `Use from Task ${cache.task_id}`;
            btn.addEventListener('click', () => _startSession(`use_from_${cache.task_id}`));
            actionsEl.appendChild(btn);
        });

        const freshBtn = document.createElement('button');
        freshBtn.className = 'btn btn-primary';
        freshBtn.textContent = 'Fresh Start';
        freshBtn.addEventListener('click', () => _startSession('fresh'));
        actionsEl.appendChild(freshBtn);
    } else {
        infoEl.textContent = 'No existing cache. Starting fresh.';

        const startBtn = document.createElement('button');
        startBtn.className = 'btn btn-primary';
        startBtn.textContent = 'Start';
        startBtn.addEventListener('click', () => _startSession('fresh'));
        actionsEl.appendChild(startBtn);
    }
}

/**
 * Create/resume session, fetch video info, run detection, then navigate.
 * @param {string} mode - 'resume', 'build_on', 'fresh', or 'use_from_<id>'.
 */
async function _startSession(mode) {
    const actionsEl = document.getElementById('setup-actions');
    const progressEl = document.getElementById('setup-progress');

    // Disable buttons
    actionsEl.querySelectorAll('.btn').forEach((b) => (b.disabled = true));
    progressEl.classList.remove('hidden');
    progressEl.textContent = 'Initializing session...';

    try {
        // 1) Resume or create session
        const sessionResult = await API.post('/session/resume', {
            project_id: AppState.projectId,
            task_id: AppState.taskId,
            annotation_id: AppState.annotationId,
            mode,
        });

        AppState.sessionId = sessionResult.session_id;
        AppState.stats = sessionResult;

        // If resuming into an advanced phase, go directly there
        if (mode === 'resume' && sessionResult.phase && sessionResult.phase !== 'init') {
            _applyStats(sessionResult);
            showToast('Session resumed', 'success');
            navigate(sessionResult.phase);
            return;
        }

        // 2) Fetch video info (background job)
        progressEl.textContent = 'Fetching video info...';
        const videoJob = await API.post(`/session/${AppState.sessionId}/video_info`, {});

        await new Promise((resolve, reject) => {
            pollJob(
                videoJob.job_id,
                (p) => {
                    progressEl.textContent = p.step || 'Fetching video info...';
                },
                (p) => {
                    if (p.status === 'failed') {
                        reject(new Error(p.error || 'Video info fetch failed'));
                    } else {
                        resolve(p);
                    }
                }
            );
        });

        // 3) Get updated session status
        const status = await API.get(`/session/${AppState.sessionId}/status`);
        _applyStats(status);

        // 4) Run detection
        progressEl.textContent = 'Running detection...';
        const prompt = document.getElementById('setup-prompt').value.trim() || 'person';
        const detectJob = await API.post('/detect/start', {
            session_id: AppState.sessionId,
            prompt,
        });

        await new Promise((resolve, reject) => {
            pollJob(
                detectJob.job_id,
                (p) => {
                    const pct = p.percent > 0 ? ` (${Math.round(p.percent)}%)` : '';
                    progressEl.textContent = (p.step || 'Detecting...') + pct;
                },
                (p) => {
                    if (p.status === 'failed') {
                        reject(new Error(p.error || 'Detection failed'));
                    } else {
                        resolve(p);
                    }
                }
            );
        });

        showToast('Detection complete', 'success');
        navigate('detection');
    } catch (err) {
        showToast(`Setup failed: ${err.message}`, 'error');
        actionsEl.querySelectorAll('.btn').forEach((b) => (b.disabled = false));
        progressEl.textContent = '';
    }
}

/**
 * Apply session stats to AppState.
 * @param {Object} stats
 */
function _applyStats(stats) {
    AppState.stats = stats;
    if (stats.video_frames) AppState.framesCount = stats.video_frames;
    if (stats.sampled_frames != null) {
        // Note: sampled_frames from stats is a count, not the array.
        // We may need to fetch actual sampled frame list from crops.
    }
}

// ---------------------------------------------------------------------------
// Phase Renderer: Detection / Classification
// ---------------------------------------------------------------------------

/**
 * Render the detection/classification work area.
 * Split panel: left = frame viewer, right = crop labeler + grid.
 * @param {HTMLElement} app
 */
function renderDetection(app) {
    if (!AppState.sessionId) {
        showToast('No active session. Redirecting to setup.', 'warning');
        navigate('setup');
        return;
    }

    // Create split panel
    const split = new SplitPanel(app, 0.6);
    AppState._components.splitPanel = split;

    const leftPanel = split.getLeft();
    const rightPanel = split.getRight();

    // Toolbar at top of left panel
    const toolbar = new Toolbar(leftPanel);
    AppState._components.toolbar = toolbar;

    // Frame viewer
    const frameViewer = new FrameViewer(leftPanel, {
        width: AppState.videoWidth || 1920,
        height: AppState.videoHeight || 1080,
    });
    AppState._components.frameViewer = frameViewer;

    // Progress overlay on the left panel
    const progress = new ProgressOverlay(leftPanel);
    AppState._components.progressOverlay = progress;

    // Crop labeler at top of right panel
    const cropLabeler = new CropLabeler(rightPanel);
    AppState._components.cropLabeler = cropLabeler;

    // Crop grid below the labeler in right panel
    const cropGrid = new CropGrid(rightPanel);
    AppState._components.cropGrid = cropGrid;

    // Wire up frame viewer draw mode
    frameViewer.onBoxDrawn(async (box) => {
        try {
            const result = await API.post('/detect/draw', {
                session_id: AppState.sessionId,
                frame_idx: frameViewer.getCurrentFrame(),
                xyxy: [box.x1, box.y1, box.x2, box.y2],
            });
            showToast('Box added', 'success');
            AppState.stats = result;
            frameViewer.reload(AppState.sessionId);
            await _refreshCrops();
        } catch (err) {
            // Already toasted
        }
    });

    // Wire up crop labeler callbacks
    cropLabeler.onAccept((crop) => _labelCrop(crop, 'accepted'));
    cropLabeler.onReject((crop) => _labelCrop(crop, 'rejected'));

    // Wire up crop grid selection
    cropGrid.onCropSelect((crop, index) => {
        AppState.currentCropIndex = index;
        cropLabeler.showCrop(crop, AppState.sessionId);
        // Navigate frame viewer to the crop's frame
        frameViewer.loadFrame(crop.frame_idx, AppState.sessionId);
    });

    // Render toolbar
    _renderToolbar();

    // Load initial data
    _loadDetectionData();
}

/** Refresh the toolbar with current state. */
function _renderToolbar() {
    const toolbar = AppState._components.toolbar;
    if (!toolbar) return;

    toolbar.render({
        drawMode: AppState.drawMode,
        sortBy: AppState.sortBy,
        filterLabel: AppState.filterLabel,
        stats: AppState.stats,
        recallStrategies: ['multi_prompt', 'feature_search'],

        onDrawToggle: () => {
            AppState.drawMode = !AppState.drawMode;
            const fv = AppState._components.frameViewer;
            if (fv) {
                if (AppState.drawMode) {
                    fv.enableDrawMode();
                } else {
                    fv.disableDrawMode();
                }
            }
            _renderToolbar();
        },

        onTrain: _onTrain,
        onRecall: _onRecall,

        onPrevFrame: () => _navigateFrame(-1),
        onNextFrame: () => _navigateFrame(1),

        onSortChange: (sort) => {
            AppState.sortBy = sort;
            _refreshCrops();
        },

        onFilterChange: (filter) => {
            AppState.filterLabel = filter;
            _refreshCrops();
        },

        onAdvancePhase: async () => {
            const confirmed = await Modal.confirm(
                'Advance to ReID',
                'Move to the Re-Identification phase? You can return to labeling later.'
            );
            if (confirmed) {
                navigate('reid');
            }
        },
    });
}

/** Load crops and video info for detection phase. */
async function _loadDetectionData() {
    try {
        // Get session status for video dimensions
        const status = await API.get(`/session/${AppState.sessionId}/status`);
        _applyStats(status);

        // Update frame viewer dimensions if we have them
        const fv = AppState._components.frameViewer;
        if (fv && AppState.videoWidth && AppState.videoHeight) {
            fv.setVideoDimensions(AppState.videoWidth, AppState.videoHeight);
        }

        await _refreshCrops();

        // Load first sampled frame
        if (AppState.crops.length > 0) {
            const firstFrame = AppState.crops[0].frame_idx;
            AppState.currentFrameIdx = firstFrame;
            if (fv) fv.loadFrame(firstFrame, AppState.sessionId);
        }
    } catch (err) {
        showToast(`Failed to load detection data: ${err.message}`, 'error');
    }
}

/** Refresh the crop list from the backend. */
async function _refreshCrops() {
    try {
        const result = await API.get('/detect/crops', {
            session_id: AppState.sessionId,
            sort: AppState.sortBy,
            filter: AppState.filterLabel,
            limit: 200,
        });
        AppState.crops = result.crops || [];

        const grid = AppState._components.cropGrid;
        if (grid) {
            grid.render(AppState.crops, AppState.sessionId);

            // Re-select current crop
            if (AppState.currentCropIndex >= 0 && AppState.currentCropIndex < AppState.crops.length) {
                grid.select(AppState.currentCropIndex);
                const labeler = AppState._components.cropLabeler;
                if (labeler) {
                    labeler.showCrop(AppState.crops[AppState.currentCropIndex], AppState.sessionId);
                }
            } else if (AppState.crops.length > 0) {
                AppState.currentCropIndex = 0;
                grid.select(0);
                const labeler = AppState._components.cropLabeler;
                if (labeler) {
                    labeler.showCrop(AppState.crops[0], AppState.sessionId);
                }
            }
        }

        // Refresh stats in toolbar
        const statusResult = await API.get(`/session/${AppState.sessionId}/status`);
        _applyStats(statusResult);
        _renderToolbar();
    } catch (err) {
        // Already toasted
    }
}

/**
 * Label a crop and advance to the next one.
 * @param {Object} crop
 * @param {string} label - 'accepted' or 'rejected'.
 */
async function _labelCrop(crop, label) {
    try {
        const result = await API.post('/detect/label', {
            session_id: AppState.sessionId,
            labels: { [crop.crop_id]: label },
        });
        AppState.stats = result;

        // Update local crop state
        crop.label = label;

        // Update grid card in place
        const grid = AppState._components.cropGrid;
        if (grid) {
            grid.updateCardLabel(AppState.currentCropIndex, label);
        }

        // Refresh the frame viewer (annotations changed)
        const fv = AppState._components.frameViewer;
        if (fv) fv.reload(AppState.sessionId);

        // Auto-advance to next pending crop
        _advanceToNextPending();

        // Update toolbar stats
        _renderToolbar();
    } catch (err) {
        // Already toasted
    }
}

/** Move to the next pending (unlabeled) crop. */
function _advanceToNextPending() {
    const crops = AppState.crops;
    const start = AppState.currentCropIndex;

    // Search forward from current position
    for (let i = start + 1; i < crops.length; i++) {
        if (crops[i].label === 'pending') {
            _selectCropByIndex(i);
            return;
        }
    }
    // Wrap around
    for (let i = 0; i < start; i++) {
        if (crops[i].label === 'pending') {
            _selectCropByIndex(i);
            return;
        }
    }
    // No pending crops left -- just move to next
    if (start + 1 < crops.length) {
        _selectCropByIndex(start + 1);
    }
}

/**
 * Select a crop by index, updating grid, labeler, and frame viewer.
 * @param {number} index
 */
function _selectCropByIndex(index) {
    if (index < 0 || index >= AppState.crops.length) return;
    AppState.currentCropIndex = index;

    const crop = AppState.crops[index];
    const grid = AppState._components.cropGrid;
    const labeler = AppState._components.cropLabeler;
    const fv = AppState._components.frameViewer;

    if (grid) grid.select(index);
    if (labeler) labeler.showCrop(crop, AppState.sessionId);
    if (fv && crop.frame_idx !== fv.getCurrentFrame()) {
        fv.loadFrame(crop.frame_idx, AppState.sessionId);
    }
}

/**
 * Navigate to the prev/next sampled frame.
 * @param {number} direction - -1 for previous, +1 for next.
 */
function _navigateFrame(direction) {
    // Build a unique sorted list of frame indices from crops
    const frameSet = new Set(AppState.crops.map((c) => c.frame_idx));
    const frames = Array.from(frameSet).sort((a, b) => a - b);
    if (frames.length === 0) return;

    const fv = AppState._components.frameViewer;
    if (!fv) return;

    const currentFrame = fv.getCurrentFrame();
    let currentIdx = frames.indexOf(currentFrame);

    if (currentIdx === -1) {
        // Find closest frame
        currentIdx = frames.findIndex((f) => f >= currentFrame);
        if (currentIdx === -1) currentIdx = frames.length - 1;
    }

    const nextIdx = Math.max(0, Math.min(frames.length - 1, currentIdx + direction));
    const nextFrame = frames[nextIdx];

    AppState.currentFrameIdx = nextFrame;
    fv.loadFrame(nextFrame, AppState.sessionId);
}

/** Handle the Train Classifier button. */
async function _onTrain() {
    const progress = AppState._components.progressOverlay;

    try {
        progress.show('Starting classifier training...', -1);

        const job = await API.post('/detect/train', {
            session_id: AppState.sessionId,
        });

        pollJob(
            job.job_id,
            (p) => {
                progress.show(p.step || 'Training...', p.percent || -1);
            },
            async (p) => {
                progress.hide();
                if (p.status === 'completed') {
                    showToast('Classifier trained successfully', 'success');
                    await _refreshCrops();
                } else {
                    showToast(`Training failed: ${p.error}`, 'error');
                }
            }
        );
    } catch (err) {
        progress.hide();
    }
}

/**
 * Handle a recall strategy selection.
 * @param {string} strategy - 'multi_prompt' or 'feature_search'.
 */
async function _onRecall(strategy) {
    const progress = AppState._components.progressOverlay;

    let extraPrompts = [];
    if (strategy === 'multi_prompt') {
        const closeModal = Modal.show(
            'Multi-Prompt Recall',
            '<p style="color:var(--text-secondary);font-size:0.85rem;margin-bottom:12px;">' +
            'Enter additional detection prompts (one per line):</p>' +
            '<textarea id="recall-prompts" rows="4" style="width:100%;padding:8px;' +
            'background:var(--bg-body);border:1px solid var(--border-default);' +
            'color:var(--text-primary);border-radius:var(--radius-sm);' +
            'font-family:var(--font-stack);font-size:0.85rem;"' +
            ' placeholder="e.g. human\nwalking person\nstanding figure"></textarea>' +
            '<div style="margin-top:12px;display:flex;gap:8px;justify-content:flex-end;">' +
            '<button id="recall-cancel-btn" class="btn btn-ghost">Cancel</button>' +
            '<button id="recall-go-btn" class="btn btn-primary">Run</button></div>'
        );

        return new Promise((resolve) => {
            // Wait for the modal DOM to be ready
            requestAnimationFrame(() => {
                const goBtn = document.getElementById('recall-go-btn');
                const cancelBtn = document.getElementById('recall-cancel-btn');
                const textarea = document.getElementById('recall-prompts');

                if (cancelBtn) {
                    cancelBtn.addEventListener('click', () => {
                        closeModal();
                        resolve();
                    });
                }

                if (goBtn) {
                    goBtn.addEventListener('click', async () => {
                        extraPrompts = (textarea.value || '')
                            .split('\n')
                            .map((s) => s.trim())
                            .filter(Boolean);
                        closeModal();
                        await _runRecallJob(strategy, extraPrompts);
                        resolve();
                    });
                }
            });
        });
    } else {
        await _runRecallJob(strategy, []);
    }
}

/**
 * Execute a recall strategy background job.
 * @param {string} strategy
 * @param {Array<string>} extraPrompts
 */
async function _runRecallJob(strategy, extraPrompts) {
    const progress = AppState._components.progressOverlay;

    try {
        progress.show(`Running ${strategy.replace(/_/g, ' ')}...`, -1);

        const job = await API.post('/detect/recall_strategy', {
            session_id: AppState.sessionId,
            strategy,
            prompts: extraPrompts,
        });

        pollJob(
            job.job_id,
            (p) => {
                progress.show(p.step || 'Recall...', p.percent || -1);
            },
            async (p) => {
                progress.hide();
                if (p.status === 'completed') {
                    showToast(`Recall (${strategy}) complete`, 'success');
                    await _refreshCrops();
                } else {
                    showToast(`Recall failed: ${p.error}`, 'error');
                }
            }
        );
    } catch (err) {
        progress.hide();
    }
}

// ---------------------------------------------------------------------------
// Phase Renderer: ReID
// ---------------------------------------------------------------------------

/**
 * Render the Re-Identification comparison interface.
 * Delegates to renderReIDPhase (defined in reid_ui.js) if available,
 * otherwise renders a basic fallback.
 * @param {HTMLElement} app
 */
function renderReID(app) {
    if (!AppState.sessionId) {
        showToast('No active session. Redirecting to setup.', 'warning');
        navigate('setup');
        return;
    }

    // If reid_ui.js provides an extended renderer, use it
    if (typeof renderReIDPhase === 'function') {
        renderReIDPhase(app);
        return;
    }

    // Fallback: basic ReID interface
    _renderReIDFallback(app);
}

/** Basic ReID interface when reid_ui.js is not loaded. */
function _renderReIDFallback(app) {
    const panel = document.createElement('div');
    panel.className = 'reid-panel';
    panel.style.minWidth = '0';
    panel.style.display = 'flex';
    panel.style.flexDirection = 'column';
    panel.style.gap = '16px';
    panel.style.maxWidth = '1200px';
    panel.style.margin = '0 auto';

    // Header with start button
    const header = document.createElement('div');
    header.style.cssText = 'display:flex;align-items:center;gap:16px;';
    header.innerHTML = '<h2 style="font-size:1.1rem;">Re-Identification</h2>';

    const startBtn = document.createElement('button');
    startBtn.className = 'btn btn-primary';
    startBtn.textContent = 'Start Clustering';
    startBtn.addEventListener('click', () => _startReIDClustering(panel));
    header.appendChild(startBtn);

    const advanceBtn = document.createElement('button');
    advanceBtn.className = 'btn btn-secondary';
    advanceBtn.textContent = 'Skip to Seeding';
    advanceBtn.addEventListener('click', () => navigate('seeding'));
    header.appendChild(advanceBtn);

    panel.appendChild(header);

    // Comparison area (populated after clustering)
    const compArea = document.createElement('div');
    compArea.id = 'reid-comparison-area';
    panel.appendChild(compArea);

    app.appendChild(panel);
}

/** Start the ReID clustering job and then show pairs. */
async function _startReIDClustering(container) {
    try {
        showToast('Starting ReID clustering...', 'info');

        const job = await API.post('/reid/start', {
            session_id: AppState.sessionId,
        });

        pollJob(
            job.job_id,
            (p) => {
                // Could show inline progress here
            },
            async (p) => {
                if (p.status === 'completed') {
                    showToast('Clustering complete', 'success');
                    await _loadReIDPairs();
                } else {
                    showToast(`Clustering failed: ${p.error}`, 'error');
                }
            }
        );
    } catch (err) {
        // Already toasted
    }
}

/** Load ReID pairs and render the comparison UI. */
async function _loadReIDPairs() {
    try {
        const data = await API.get('/reid/clusters', {
            session_id: AppState.sessionId,
        });

        AppState._reidData = data;
        AppState._reidPairIndex = 0;

        const area = document.getElementById('reid-comparison-area');
        if (!area) return;
        area.innerHTML = '';

        const info = document.createElement('div');
        info.style.cssText = 'font-size:0.8rem;color:var(--text-secondary);margin-bottom:12px;';
        info.textContent =
            `${data.n_identities} identities estimated | ` +
            `${data.total_pairs} pairs total | ` +
            `${data.unresolved_pairs.length} unresolved`;
        area.appendChild(info);

        if (data.unresolved_pairs.length > 0) {
            _renderReIDPair(area, data.unresolved_pairs, 0);
        } else {
            const done = document.createElement('p');
            done.style.color = 'var(--color-accepted)';
            done.textContent = 'All pairs resolved. Ready for seeding.';
            area.appendChild(done);

            const seedBtn = document.createElement('button');
            seedBtn.className = 'btn btn-primary';
            seedBtn.textContent = 'Proceed to Seeding';
            seedBtn.addEventListener('click', () => navigate('seeding'));
            area.appendChild(seedBtn);
        }
    } catch (err) {
        // Already toasted
    }
}

/**
 * Render a single ReID comparison pair.
 * @param {HTMLElement} area
 * @param {Array} pairs
 * @param {number} index
 */
function _renderReIDPair(area, pairs, index) {
    if (index >= pairs.length) {
        area.innerHTML =
            '<p style="color:var(--color-accepted)">All pairs resolved!</p>';
        const seedBtn = document.createElement('button');
        seedBtn.className = 'btn btn-primary';
        seedBtn.textContent = 'Proceed to Seeding';
        seedBtn.addEventListener('click', () => navigate('seeding'));
        area.appendChild(seedBtn);
        return;
    }

    const pair = pairs[index];
    AppState._reidPairIndex = index;

    // Remove existing pair UI
    const existingPairUI = area.querySelector('.reid-pair-ui');
    if (existingPairUI) existingPairUI.remove();

    const pairUI = document.createElement('div');
    pairUI.className = 'reid-pair-ui';
    pairUI.style.cssText =
        'display:flex;flex-direction:column;gap:16px;' +
        'background:var(--bg-surface);border:1px solid var(--border-default);' +
        'border-radius:var(--radius-md);padding:20px;';

    // Counter
    const counter = document.createElement('div');
    counter.style.cssText = 'font-size:0.75rem;color:var(--text-muted);';
    counter.textContent = `Pair ${index + 1} of ${pairs.length} | ` +
        `Similarity: ${pair.similarity != null ? pair.similarity.toFixed(3) : 'N/A'} | ` +
        `Pool: ${pair.pool || 'unknown'}`;
    pairUI.appendChild(counter);

    // Crop images side by side
    const crops = document.createElement('div');
    crops.className = 'reid-crops';

    const imgA = document.createElement('img');
    imgA.src = `/interview/api/detect/crop/${pair.crop_id_a}/image?session_id=${AppState.sessionId}`;
    imgA.alt = 'Crop A';
    imgA.style.cssText = 'max-height:240px;border-radius:var(--radius-sm);border:2px solid var(--border-default);';

    const vsLabel = document.createElement('span');
    vsLabel.style.cssText = 'font-size:1.2rem;font-weight:700;color:var(--text-muted);';
    vsLabel.textContent = 'vs';

    const imgB = document.createElement('img');
    imgB.src = `/interview/api/detect/crop/${pair.crop_id_b}/image?session_id=${AppState.sessionId}`;
    imgB.alt = 'Crop B';
    imgB.style.cssText = 'max-height:240px;border-radius:var(--radius-sm);border:2px solid var(--border-default);';

    crops.appendChild(imgA);
    crops.appendChild(vsLabel);
    crops.appendChild(imgB);
    pairUI.appendChild(crops);

    // Verdict buttons
    const verdict = document.createElement('div');
    verdict.className = 'reid-verdict';

    const btnSame = document.createElement('button');
    btnSame.className = 'btn btn-accept';
    btnSame.innerHTML = 'Same <kbd style="margin-left:4px;font-size:0.65rem;">Y</kbd>';
    btnSame.addEventListener('click', () => _resolveReIDPair(pair.pair_id, 'same', area, pairs, index));

    const btnDiff = document.createElement('button');
    btnDiff.className = 'btn btn-reject';
    btnDiff.innerHTML = 'Different <kbd style="margin-left:4px;font-size:0.65rem;">N</kbd>';
    btnDiff.addEventListener('click', () => _resolveReIDPair(pair.pair_id, 'different', area, pairs, index));

    const btnUnsure = document.createElement('button');
    btnUnsure.className = 'btn btn-unsure';
    btnUnsure.innerHTML = 'Unsure <kbd style="margin-left:4px;font-size:0.65rem;">U</kbd>';
    btnUnsure.addEventListener('click', () => _resolveReIDPair(pair.pair_id, 'unsure', area, pairs, index));

    verdict.appendChild(btnSame);
    verdict.appendChild(btnDiff);
    verdict.appendChild(btnUnsure);
    pairUI.appendChild(verdict);

    // Keyboard hints
    const hints = document.createElement('div');
    hints.className = 'keyboard-hints';
    hints.innerHTML =
        '<span><kbd>Y</kbd> Same</span>' +
        '<span><kbd>N</kbd> Different</span>' +
        '<span><kbd>U</kbd> Unsure</span>';
    pairUI.appendChild(hints);

    area.appendChild(pairUI);
}

/**
 * Resolve a ReID pair and advance to the next.
 * @param {string} pairId
 * @param {string} resolution - 'same', 'different', 'unsure'.
 * @param {HTMLElement} area
 * @param {Array} pairs
 * @param {number} index
 */
async function _resolveReIDPair(pairId, resolution, area, pairs, index) {
    try {
        await API.post('/reid/resolve', {
            session_id: AppState.sessionId,
            resolutions: { [pairId]: resolution },
        });
        _renderReIDPair(area, pairs, index + 1);
    } catch (err) {
        // Already toasted
    }
}

/**
 * Resolve the current ReID pair (called by keyboard handler).
 * @param {string} resolution - 'same', 'different', 'unsure'.
 */
function resolveCurrentPair(resolution) {
    if (AppState.phase !== 'reid') return;
    if (!AppState._reidData) return;

    const pairs = AppState._reidData.unresolved_pairs;
    const index = AppState._reidPairIndex || 0;
    if (index >= pairs.length) return;

    const area = document.getElementById('reid-comparison-area');
    if (!area) return;

    _resolveReIDPair(pairs[index].pair_id, resolution, area, pairs, index);
}

// ---------------------------------------------------------------------------
// Phase Renderer: Seeding
// ---------------------------------------------------------------------------

/**
 * Render the seeding configuration and upload interface.
 * Delegates to renderSeedingPhase (defined in seeding_ui.js) if available.
 * @param {HTMLElement} app
 */
function renderSeeding(app) {
    if (!AppState.sessionId) {
        showToast('No active session. Redirecting to setup.', 'warning');
        navigate('setup');
        return;
    }

    // If seeding_ui.js provides an extended renderer, use it
    if (typeof renderSeedingPhase === 'function') {
        renderSeedingPhase(app);
        return;
    }

    // Fallback: basic seeding interface
    _renderSeedingFallback(app);
}

/** Basic seeding interface when seeding_ui.js is not loaded. */
function _renderSeedingFallback(app) {
    const panel = document.createElement('div');
    panel.className = 'seeding-panel';

    // Config form
    const configEl = document.createElement('div');
    configEl.className = 'seeding-config';
    configEl.innerHTML = `
        <div class="form-group">
            <label for="seed-frame-interval">Frame Interval</label>
            <input type="number" id="seed-frame-interval" value="5" min="1" max="100">
        </div>
        <div class="form-group">
            <label for="seed-confidence">Confidence Threshold</label>
            <input type="number" id="seed-confidence" value="0.8" min="0" max="1" step="0.05">
        </div>
    `;
    panel.appendChild(configEl);

    // Action buttons
    const actions = document.createElement('div');
    actions.className = 'seeding-actions';

    const saveConfigBtn = document.createElement('button');
    saveConfigBtn.className = 'btn btn-secondary';
    saveConfigBtn.textContent = 'Save Config';
    saveConfigBtn.addEventListener('click', async () => {
        const interval = parseInt(document.getElementById('seed-frame-interval').value, 10);
        const threshold = parseFloat(document.getElementById('seed-confidence').value);
        try {
            await API.put('/seeds/config', {
                session_id: AppState.sessionId,
                frame_interval: interval,
                confidence_threshold: threshold,
            });
            showToast('Config saved', 'success');
        } catch (err) {
            // Already toasted
        }
    });
    actions.appendChild(saveConfigBtn);

    const generateBtn = document.createElement('button');
    generateBtn.className = 'btn btn-primary';
    generateBtn.textContent = 'Generate Seeds';
    generateBtn.addEventListener('click', () => _generateSeeds(panel));
    actions.appendChild(generateBtn);

    const uploadBtn = document.createElement('button');
    uploadBtn.className = 'btn btn-primary';
    uploadBtn.textContent = 'Upload to Label Studio';
    uploadBtn.disabled = true;
    uploadBtn.id = 'seed-upload-btn';
    uploadBtn.addEventListener('click', () => _uploadSeeds());
    actions.appendChild(uploadBtn);

    panel.appendChild(actions);

    // Seed preview area
    const preview = document.createElement('div');
    preview.id = 'seed-preview';
    preview.style.marginTop = '20px';
    panel.appendChild(preview);

    app.appendChild(panel);

    // Load current config
    _loadSeedConfig();
}

/** Load existing seed config from backend. */
async function _loadSeedConfig() {
    try {
        const config = await API.get('/seeds/config', {
            session_id: AppState.sessionId,
        });
        const intervalInput = document.getElementById('seed-frame-interval');
        const confidenceInput = document.getElementById('seed-confidence');
        if (intervalInput) intervalInput.value = config.frame_interval;
        if (confidenceInput) confidenceInput.value = config.confidence_threshold;
    } catch (err) {
        // Use defaults
    }
}

/** Run seed generation and show preview. */
async function _generateSeeds(panel) {
    try {
        showToast('Generating seeds...', 'info');

        const job = await API.post('/seeds/generate', {
            session_id: AppState.sessionId,
        });

        pollJob(
            job.job_id,
            (p) => {
                // Inline progress
            },
            async (p) => {
                if (p.status === 'completed') {
                    showToast('Seeds generated', 'success');
                    await _loadSeedPreview();
                } else {
                    showToast(`Seed generation failed: ${p.error}`, 'error');
                }
            }
        );
    } catch (err) {
        // Already toasted
    }
}

/** Load and display seed summary. */
async function _loadSeedPreview() {
    try {
        const data = await API.get('/seeds/list', { session_id: AppState.sessionId });
        const preview = document.getElementById('seed-preview');
        if (!preview) return;

        preview.innerHTML = '';

        // Summary
        const summary = document.createElement('div');
        summary.style.cssText =
            'font-size:0.85rem;color:var(--text-secondary);margin-bottom:16px;';
        summary.textContent = `Total seeds: ${data.total_seeds}`;
        preview.appendChild(summary);

        // Identity table
        if (data.identities && Object.keys(data.identities).length > 0) {
            const wrapper = document.createElement('div');
            wrapper.className = 'seed-table-wrapper';

            const table = document.createElement('table');
            table.className = 'seed-table';
            table.innerHTML = `
                <thead>
                    <tr>
                        <th>Identity</th>
                        <th>Seed Count</th>
                        <th>Frames</th>
                    </tr>
                </thead>
                <tbody></tbody>
            `;

            const tbody = table.querySelector('tbody');
            for (const [identity, info] of Object.entries(data.identities)) {
                const tr = document.createElement('tr');
                const frames = info.frames.slice(0, 10).join(', ');
                const more = info.frames.length > 10 ? ` (+${info.frames.length - 10} more)` : '';
                tr.innerHTML = `
                    <td>${identity}</td>
                    <td>${info.count}</td>
                    <td style="font-size:0.75rem;color:var(--text-muted)">${frames}${more}</td>
                `;
                tbody.appendChild(tr);
            }

            wrapper.appendChild(table);
            preview.appendChild(wrapper);
        }

        // Enable upload button
        const uploadBtn = document.getElementById('seed-upload-btn');
        if (uploadBtn) uploadBtn.disabled = data.total_seeds === 0;
    } catch (err) {
        // Already toasted
    }
}

/** Upload seeds to Label Studio. */
async function _uploadSeeds() {
    const confirmed = await Modal.confirm(
        'Upload Seeds',
        'Upload seed annotations to Label Studio? This will create/modify annotations.'
    );
    if (!confirmed) return;

    try {
        showToast('Uploading seeds...', 'info');

        const job = await API.post('/seeds/upload', {
            session_id: AppState.sessionId,
        });

        pollJob(
            job.job_id,
            (p) => { /* progress */ },
            (p) => {
                if (p.status === 'completed') {
                    showToast('Seeds uploaded successfully!', 'success');
                } else {
                    showToast(`Upload failed: ${p.error}`, 'error');
                }
            }
        );
    } catch (err) {
        // Already toasted
    }
}

// ---------------------------------------------------------------------------
// Keyboard Shortcuts
// ---------------------------------------------------------------------------

document.addEventListener('keydown', (e) => {
    // Skip if user is typing in an input or textarea
    const tag = (e.target.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'textarea' || tag === 'select') return;

    const phase = AppState.phase;

    // Detection / Classification shortcuts
    if (phase === 'detection' || phase === 'classification') {
        if (e.key === 'Enter') {
            e.preventDefault();
            _acceptCurrentCrop();
        } else if (e.key === 'Backspace') {
            e.preventDefault();
            _rejectCurrentCrop();
        } else if (e.key === 'ArrowRight') {
            e.preventDefault();
            _nextCrop();
        } else if (e.key === 'ArrowLeft') {
            e.preventDefault();
            _prevCrop();
        } else if (e.key === 'd' || e.key === 'D') {
            // Toggle draw mode
            e.preventDefault();
            AppState.drawMode = !AppState.drawMode;
            const fv = AppState._components.frameViewer;
            if (fv) {
                if (AppState.drawMode) fv.enableDrawMode();
                else fv.disableDrawMode();
            }
            _renderToolbar();
        }
    }

    // ReID shortcuts
    if (phase === 'reid') {
        if (e.key === 'y' || e.key === 'Y') {
            e.preventDefault();
            resolveCurrentPair('same');
        } else if (e.key === 'n' || e.key === 'N') {
            e.preventDefault();
            resolveCurrentPair('different');
        } else if (e.key === 'u' || e.key === 'U') {
            e.preventDefault();
            resolveCurrentPair('unsure');
        }
    }
});

/** Accept the current crop via keyboard. */
function _acceptCurrentCrop() {
    const crop = AppState.crops[AppState.currentCropIndex];
    if (crop && crop.label === 'pending') {
        _labelCrop(crop, 'accepted');
    }
}

/** Reject the current crop via keyboard. */
function _rejectCurrentCrop() {
    const crop = AppState.crops[AppState.currentCropIndex];
    if (crop && crop.label === 'pending') {
        _labelCrop(crop, 'rejected');
    }
}

/** Move to the next crop. */
function _nextCrop() {
    const next = Math.min(AppState.currentCropIndex + 1, AppState.crops.length - 1);
    _selectCropByIndex(next);
}

/** Move to the previous crop. */
function _prevCrop() {
    const prev = Math.max(AppState.currentCropIndex - 1, 0);
    _selectCropByIndex(prev);
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
    // Set up hash-based routing
    window.addEventListener('hashchange', _onHashChange);

    // Parse initial hash
    const hash = window.location.hash.replace(/^#\/?/, '') || 'setup';
    AppState.phase = hash;
    renderPhase(hash);
    updatePhaseIndicator(hash);
});
