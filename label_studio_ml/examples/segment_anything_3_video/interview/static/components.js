/* ==========================================================================
   SAM3 Interview UI - Reusable Components
   Vanilla JS UI components for the interview workflow SPA.
   No frameworks, no imports -- everything attaches to the global scope.
   ========================================================================== */

'use strict';

// ---------------------------------------------------------------------------
// SplitPanel
// ---------------------------------------------------------------------------

class SplitPanel {
    /**
     * Creates a two-column split layout inside the given container.
     * @param {HTMLElement} container - Parent element to render into.
     * @param {number} leftRatio - Fraction for the left panel width (0-1).
     */
    constructor(container, leftRatio = 0.6) {
        this.container = container;
        this.leftRatio = leftRatio;
        this._render();
    }

    _render() {
        this.el = document.createElement('div');
        this.el.className = 'split-panel';

        this.leftEl = document.createElement('div');
        this.leftEl.className = 'panel-left';
        this.leftEl.style.flex = `0 0 ${this.leftRatio * 100}%`;

        this.rightEl = document.createElement('div');
        this.rightEl.className = 'panel-right';

        this.el.appendChild(this.leftEl);
        this.el.appendChild(this.rightEl);
        this.container.appendChild(this.el);
    }

    /** @returns {HTMLElement} The left panel element. */
    getLeft() {
        return this.leftEl;
    }

    /** @returns {HTMLElement} The right panel element. */
    getRight() {
        return this.rightEl;
    }

    destroy() {
        if (this.el && this.el.parentNode) {
            this.el.parentNode.removeChild(this.el);
        }
    }
}

// ---------------------------------------------------------------------------
// FrameViewer
// ---------------------------------------------------------------------------

class FrameViewer {
    /**
     * Frame image viewer with optional canvas overlay for drawing boxes.
     * @param {HTMLElement} container - Element to render the viewer into.
     * @param {Object} options
     * @param {number} [options.width] - Video pixel width (for coordinate mapping).
     * @param {number} [options.height] - Video pixel height (for coordinate mapping).
     */
    constructor(container, options = {}) {
        this.container = container;
        this.videoWidth = options.width || 1920;
        this.videoHeight = options.height || 1080;
        this._drawMode = false;
        this._drawing = false;
        this._startX = 0;
        this._startY = 0;
        this._boxCallbacks = [];
        this._currentFrameIdx = -1;

        this._render();
        this._bindEvents();
    }

    _render() {
        this.el = document.createElement('div');
        this.el.className = 'frame-viewer';

        // Main frame image
        this.img = document.createElement('img');
        this.img.className = 'frame-image';
        this.img.alt = 'Video frame';
        this.img.draggable = false;
        this.el.appendChild(this.img);

        // Canvas overlay for drawing boxes
        this.canvas = document.createElement('canvas');
        this.canvas.className = 'frame-canvas-overlay';
        this.canvas.style.position = 'absolute';
        this.canvas.style.top = '0';
        this.canvas.style.left = '0';
        this.canvas.style.width = '100%';
        this.canvas.style.height = '100%';
        this.canvas.style.pointerEvents = 'none';
        this.canvas.style.cursor = 'default';
        this.el.appendChild(this.canvas);
        this.ctx = this.canvas.getContext('2d');

        // Frame indicator badge
        this.badge = document.createElement('div');
        this.badge.className = 'card-badge';
        this.badge.style.cssText =
            'position:absolute;bottom:8px;left:8px;font-size:0.75rem;' +
            'background:rgba(0,0,0,0.7);color:#fff;padding:2px 8px;' +
            'border-radius:4px;pointer-events:none;';
        this.badge.textContent = 'Frame --';
        this.el.appendChild(this.badge);

        this.container.appendChild(this.el);
    }

    _bindEvents() {
        // We use the canvas for mouse events in draw mode.
        this.canvas.addEventListener('mousedown', (e) => this._onMouseDown(e));
        this.canvas.addEventListener('mousemove', (e) => this._onMouseMove(e));
        this.canvas.addEventListener('mouseup', (e) => this._onMouseUp(e));
        this.canvas.addEventListener('mouseleave', (e) => this._onMouseUp(e));

        // Resize observer to keep canvas size in sync
        this._resizeObserver = new ResizeObserver(() => this._syncCanvasSize());
        this._resizeObserver.observe(this.el);
    }

    _syncCanvasSize() {
        const rect = this.el.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
    }

    /**
     * Convert mouse event coordinates to pixel coordinates on the video frame.
     * Accounts for the image being object-fit:contain inside the viewer.
     */
    _eventToPixelCoords(e) {
        const rect = this.el.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        // Compute the rendered image rectangle (object-fit: contain)
        const containerW = rect.width;
        const containerH = rect.height;
        const imgAspect = this.videoWidth / this.videoHeight;
        const containerAspect = containerW / containerH;

        let renderW, renderH, offsetX, offsetY;
        if (containerAspect > imgAspect) {
            // Container is wider -- image is height-limited
            renderH = containerH;
            renderW = containerH * imgAspect;
            offsetX = (containerW - renderW) / 2;
            offsetY = 0;
        } else {
            // Container is taller -- image is width-limited
            renderW = containerW;
            renderH = containerW / imgAspect;
            offsetX = 0;
            offsetY = (containerH - renderH) / 2;
        }

        const px = ((mouseX - offsetX) / renderW) * this.videoWidth;
        const py = ((mouseY - offsetY) / renderH) * this.videoHeight;

        return {
            px: Math.max(0, Math.min(this.videoWidth, px)),
            py: Math.max(0, Math.min(this.videoHeight, py)),
            mouseX,
            mouseY,
            renderW,
            renderH,
            offsetX,
            offsetY,
        };
    }

    _onMouseDown(e) {
        if (!this._drawMode) return;
        e.preventDefault();
        const coords = this._eventToPixelCoords(e);
        this._drawing = true;
        this._startX = coords.px;
        this._startY = coords.py;
        this._startMouseX = coords.mouseX;
        this._startMouseY = coords.mouseY;
    }

    _onMouseMove(e) {
        if (!this._drawMode || !this._drawing) return;
        e.preventDefault();
        const coords = this._eventToPixelCoords(e);

        // Draw the rubber-band rectangle on the canvas
        this._syncCanvasSize();
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.strokeStyle = '#e94560';
        this.ctx.lineWidth = 2;
        this.ctx.setLineDash([6, 3]);
        this.ctx.strokeRect(
            this._startMouseX,
            this._startMouseY,
            coords.mouseX - this._startMouseX,
            coords.mouseY - this._startMouseY
        );
        this.ctx.setLineDash([]);
    }

    _onMouseUp(e) {
        if (!this._drawMode || !this._drawing) return;
        this._drawing = false;
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        const coords = this._eventToPixelCoords(e);
        const x1 = Math.min(this._startX, coords.px);
        const y1 = Math.min(this._startY, coords.py);
        const x2 = Math.max(this._startX, coords.px);
        const y2 = Math.max(this._startY, coords.py);

        // Minimum box size: 8px in video coords
        if (x2 - x1 < 8 || y2 - y1 < 8) return;

        const box = { x1, y1, x2, y2 };
        for (const cb of this._boxCallbacks) {
            cb(box);
        }
    }

    /**
     * Load and display an annotated frame from the backend.
     * @param {number} frameIdx - 0-based frame index.
     * @param {string} sessionId - Current session ID.
     * @param {boolean} [annotated=true] - Whether to load the annotated version.
     */
    loadFrame(frameIdx, sessionId, annotated = true) {
        this._currentFrameIdx = frameIdx;
        const endpoint = annotated ? 'annotated' : '';
        const path = endpoint
            ? `/interview/api/detect/frame/${frameIdx}/annotated?session_id=${sessionId}`
            : `/interview/api/detect/frame/${frameIdx}?session_id=${sessionId}`;
        this.img.src = path;
        this.badge.textContent = `Frame ${frameIdx}`;
    }

    /** Reload the currently displayed frame. */
    reload(sessionId) {
        if (this._currentFrameIdx >= 0 && sessionId) {
            this.loadFrame(this._currentFrameIdx, sessionId);
        }
    }

    /** @returns {number} The currently displayed frame index. */
    getCurrentFrame() {
        return this._currentFrameIdx;
    }

    /** Enable draw mode -- canvas becomes interactive for drawing boxes. */
    enableDrawMode() {
        this._drawMode = true;
        this.canvas.style.pointerEvents = 'auto';
        this.canvas.style.cursor = 'crosshair';
    }

    /** Disable draw mode -- canvas stops capturing mouse events. */
    disableDrawMode() {
        this._drawMode = false;
        this._drawing = false;
        this.canvas.style.pointerEvents = 'none';
        this.canvas.style.cursor = 'default';
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }

    /** @returns {boolean} Whether draw mode is currently active. */
    isDrawMode() {
        return this._drawMode;
    }

    /**
     * Register a callback for when a box is drawn.
     * @param {Function} callback - Receives {x1, y1, x2, y2} in pixel coords.
     */
    onBoxDrawn(callback) {
        this._boxCallbacks.push(callback);
    }

    /**
     * Update the video dimensions used for coordinate mapping.
     * @param {number} width
     * @param {number} height
     */
    setVideoDimensions(width, height) {
        this.videoWidth = width;
        this.videoHeight = height;
    }

    destroy() {
        if (this._resizeObserver) {
            this._resizeObserver.disconnect();
        }
        if (this.el && this.el.parentNode) {
            this.el.parentNode.removeChild(this.el);
        }
    }
}

// ---------------------------------------------------------------------------
// CropLabeler
// ---------------------------------------------------------------------------

class CropLabeler {
    /**
     * Right-panel component: displays a zoomed crop image with
     * accept / reject buttons and metadata.
     * @param {HTMLElement} container
     */
    constructor(container) {
        this.container = container;
        this._acceptCallbacks = [];
        this._rejectCallbacks = [];
        this._currentCrop = null;
        this._render();
    }

    _render() {
        this.el = document.createElement('div');
        this.el.className = 'crop-preview';

        // Crop image
        this.img = document.createElement('img');
        this.img.alt = 'Crop preview';
        this.img.draggable = false;
        this.el.appendChild(this.img);

        // Metadata block
        this.metaEl = document.createElement('div');
        this.metaEl.className = 'crop-meta';
        this.metaEl.style.cssText =
            'font-size:0.75rem;color:var(--text-secondary);text-align:center;';
        this.el.appendChild(this.metaEl);

        // Action buttons
        this.actionsEl = document.createElement('div');
        this.actionsEl.className = 'crop-actions';

        this.rejectBtn = document.createElement('button');
        this.rejectBtn.className = 'btn btn-reject';
        this.rejectBtn.textContent = 'Reject';
        this.rejectBtn.addEventListener('click', () => this._fireReject());

        this.acceptBtn = document.createElement('button');
        this.acceptBtn.className = 'btn btn-accept';
        this.acceptBtn.textContent = 'Accept';
        this.acceptBtn.addEventListener('click', () => this._fireAccept());

        this.actionsEl.appendChild(this.rejectBtn);
        this.actionsEl.appendChild(this.acceptBtn);
        this.el.appendChild(this.actionsEl);

        // Keyboard hints
        this.hintsEl = document.createElement('div');
        this.hintsEl.className = 'keyboard-hints';
        this.hintsEl.innerHTML =
            '<span><kbd>Enter</kbd> Accept</span>' +
            '<span><kbd>Backspace</kbd> Reject</span>' +
            '<span><kbd>&larr;</kbd><kbd>&rarr;</kbd> Navigate</span>';
        this.el.appendChild(this.hintsEl);

        this.container.appendChild(this.el);
    }

    /**
     * Display a crop image and its metadata.
     * @param {Object} crop - Crop data object from the API.
     * @param {string} sessionId - Current session ID.
     */
    showCrop(crop, sessionId) {
        this._currentCrop = crop;
        this.img.src = `/interview/api/detect/crop/${crop.crop_id}/image?session_id=${sessionId}`;

        const labelBadge = `<span style="color:var(--color-${crop.label})">${crop.label}</span>`;
        this.metaEl.innerHTML =
            `Frame ${crop.frame_idx} | Score ${crop.score.toFixed(2)} | ` +
            `Source: ${crop.source} | ${labelBadge}` +
            (crop.cluster_id != null ? ` | Cluster ${crop.cluster_id}` : '') +
            (crop.uncertainty != null ? ` | Unc ${crop.uncertainty.toFixed(2)}` : '');

        // Visually disable buttons if already labeled
        const isLabeled = crop.label === 'accepted' || crop.label === 'rejected';
        this.acceptBtn.disabled = crop.label === 'accepted';
        this.rejectBtn.disabled = crop.label === 'rejected';
    }

    /** Clear the crop preview. */
    clear() {
        this._currentCrop = null;
        this.img.src = '';
        this.metaEl.textContent = 'No crop selected';
        this.acceptBtn.disabled = true;
        this.rejectBtn.disabled = true;
    }

    /** @returns {Object|null} The currently displayed crop. */
    getCurrentCrop() {
        return this._currentCrop;
    }

    /**
     * Register a callback for when the Accept button is clicked.
     * @param {Function} callback - Receives the current crop object.
     */
    onAccept(callback) {
        this._acceptCallbacks.push(callback);
    }

    /**
     * Register a callback for when the Reject button is clicked.
     * @param {Function} callback - Receives the current crop object.
     */
    onReject(callback) {
        this._rejectCallbacks.push(callback);
    }

    _fireAccept() {
        if (!this._currentCrop) return;
        for (const cb of this._acceptCallbacks) {
            cb(this._currentCrop);
        }
    }

    _fireReject() {
        if (!this._currentCrop) return;
        for (const cb of this._rejectCallbacks) {
            cb(this._currentCrop);
        }
    }

    destroy() {
        if (this.el && this.el.parentNode) {
            this.el.parentNode.removeChild(this.el);
        }
    }
}

// ---------------------------------------------------------------------------
// CropGrid
// ---------------------------------------------------------------------------

class CropGrid {
    /**
     * Scrollable grid of crop thumbnails with color-coded borders.
     * @param {HTMLElement} container
     */
    constructor(container) {
        this.container = container;
        this._selectCallbacks = [];
        this._selectedIndex = -1;
        this._crops = [];
        this._sessionId = null;
        this._render();
    }

    _render() {
        this.el = document.createElement('div');
        this.el.className = 'crop-grid';
        this.container.appendChild(this.el);
    }

    /**
     * Render a list of crop thumbnails.
     * @param {Array} crops - Array of crop data objects.
     * @param {string} sessionId
     */
    render(crops, sessionId) {
        this._crops = crops;
        this._sessionId = sessionId;
        this.el.innerHTML = '';

        crops.forEach((crop, index) => {
            const card = document.createElement('div');
            card.className = 'crop-card';
            card.classList.add(crop.label);

            if (crop.source === 'human_drawn') card.classList.add('human');
            if (crop.source === 'feature_search') card.classList.add('feature');
            if (index === this._selectedIndex) card.classList.add('selected');

            const img = document.createElement('img');
            img.src = `/interview/api/detect/crop/${crop.crop_id}/image?session_id=${sessionId}`;
            img.alt = `Crop ${crop.crop_id}`;
            img.loading = 'lazy';
            card.appendChild(img);

            // Badge showing uncertainty or cluster
            const badge = document.createElement('span');
            badge.className = 'card-badge';
            if (crop.uncertainty != null && crop.uncertainty > 0) {
                badge.textContent = crop.uncertainty.toFixed(1);
            } else if (crop.cluster_id != null) {
                badge.textContent = `C${crop.cluster_id}`;
            }
            card.appendChild(badge);

            card.addEventListener('click', () => {
                this.select(index);
                for (const cb of this._selectCallbacks) {
                    cb(crop, index);
                }
            });

            this.el.appendChild(card);
        });
    }

    /**
     * Visually select a crop card by index.
     * @param {number} index
     */
    select(index) {
        if (index < 0 || index >= this._crops.length) return;

        // Remove previous selection
        const prev = this.el.querySelector('.crop-card.selected');
        if (prev) prev.classList.remove('selected');

        // Apply new selection
        this._selectedIndex = index;
        const cards = this.el.querySelectorAll('.crop-card');
        if (cards[index]) {
            cards[index].classList.add('selected');
            cards[index].scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    }

    /** @returns {number} Currently selected index. */
    getSelectedIndex() {
        return this._selectedIndex;
    }

    /** @returns {Object|null} Currently selected crop data. */
    getSelectedCrop() {
        if (this._selectedIndex < 0 || this._selectedIndex >= this._crops.length) {
            return null;
        }
        return this._crops[this._selectedIndex];
    }

    /** @returns {Array} All crops in the grid. */
    getCrops() {
        return this._crops;
    }

    /**
     * Update a single crop card's label status in place.
     * @param {number} index
     * @param {string} label - 'accepted', 'rejected', or 'pending'.
     */
    updateCardLabel(index, label) {
        if (index < 0 || index >= this._crops.length) return;
        this._crops[index].label = label;
        const cards = this.el.querySelectorAll('.crop-card');
        if (cards[index]) {
            cards[index].classList.remove('accepted', 'rejected', 'pending');
            cards[index].classList.add(label);
        }
    }

    /**
     * Register a callback for when a crop card is clicked.
     * @param {Function} callback - Receives (crop, index).
     */
    onCropSelect(callback) {
        this._selectCallbacks.push(callback);
    }

    destroy() {
        if (this.el && this.el.parentNode) {
            this.el.parentNode.removeChild(this.el);
        }
    }
}

// ---------------------------------------------------------------------------
// ProgressOverlay
// ---------------------------------------------------------------------------

class ProgressOverlay {
    /**
     * Full-area overlay with animated progress bar, step text, and percentage.
     * @param {HTMLElement} container - Element to overlay.
     */
    constructor(container) {
        this.container = container;
        this._visible = false;
        this._render();
    }

    _render() {
        this.el = document.createElement('div');
        this.el.className = 'progress-overlay';
        this.el.style.cssText =
            'position:absolute;inset:0;display:flex;flex-direction:column;' +
            'align-items:center;justify-content:center;gap:16px;' +
            'background:rgba(26,26,46,0.92);z-index:100;pointer-events:none;' +
            'opacity:0;transition:opacity 250ms ease;';

        this.stepEl = document.createElement('div');
        this.stepEl.style.cssText =
            'font-size:0.9rem;color:var(--text-primary);font-weight:600;';
        this.stepEl.textContent = 'Processing...';
        this.el.appendChild(this.stepEl);

        // Progress bar container
        const barWrapper = document.createElement('div');
        barWrapper.className = 'progress-bar-wrapper';
        barWrapper.style.cssText = 'width:320px;padding:0;background:transparent;border:none;';

        const track = document.createElement('div');
        track.className = 'progress-bar-track';

        this.fill = document.createElement('div');
        this.fill.className = 'progress-bar-fill';
        this.fill.style.width = '0%';

        track.appendChild(this.fill);
        barWrapper.appendChild(track);
        this.el.appendChild(barWrapper);

        // Percentage text
        this.pctEl = document.createElement('div');
        this.pctEl.style.cssText = 'font-size:0.8rem;color:var(--text-muted);';
        this.pctEl.textContent = '';
        this.el.appendChild(this.pctEl);

        // Ensure container is positioned for overlay
        const containerPos = getComputedStyle(this.container).position;
        if (containerPos === 'static') {
            this.container.style.position = 'relative';
        }
        this.container.appendChild(this.el);
    }

    /**
     * Show the overlay with a step description and percentage.
     * @param {string} step - Description of the current operation.
     * @param {number} percent - Progress percentage (0-100). Use -1 for indeterminate.
     */
    show(step, percent) {
        this._visible = true;
        this.el.style.opacity = '1';
        this.el.style.pointerEvents = 'auto';
        this.stepEl.textContent = step || 'Processing...';

        if (percent < 0) {
            // Indeterminate
            this.fill.classList.add('indeterminate');
            this.fill.style.width = '';
            this.pctEl.textContent = '';
        } else {
            this.fill.classList.remove('indeterminate');
            this.fill.style.width = `${Math.min(100, Math.max(0, percent))}%`;
            this.pctEl.textContent = `${Math.round(percent)}%`;
        }
    }

    /** Hide the overlay. */
    hide() {
        this._visible = false;
        this.el.style.opacity = '0';
        this.el.style.pointerEvents = 'none';
    }

    /** @returns {boolean} Whether the overlay is currently visible. */
    isVisible() {
        return this._visible;
    }

    destroy() {
        if (this.el && this.el.parentNode) {
            this.el.parentNode.removeChild(this.el);
        }
    }
}

// ---------------------------------------------------------------------------
// Toolbar
// ---------------------------------------------------------------------------

class Toolbar {
    /**
     * Top toolbar for the detection/classification phase.
     * @param {HTMLElement} container
     */
    constructor(container) {
        this.container = container;
        this._render();
    }

    _render() {
        this.el = document.createElement('div');
        this.el.className = 'toolbar';
        this.container.appendChild(this.el);
    }

    /**
     * Render the toolbar with the given options.
     * @param {Object} options
     * @param {boolean} options.drawMode - Whether draw mode is active.
     * @param {Function} options.onDrawToggle - Callback when draw toggle is clicked.
     * @param {Function} options.onTrain - Callback for the Train button.
     * @param {Function} options.onRecall - Callback when a recall strategy is chosen. Receives strategy name.
     * @param {Array<string>} [options.recallStrategies] - Available recall strategy names.
     * @param {Function} [options.onPrevFrame] - Navigate to previous sampled frame.
     * @param {Function} [options.onNextFrame] - Navigate to next sampled frame.
     * @param {Function} [options.onAdvancePhase] - Advance to next workflow phase.
     * @param {Object} [options.stats] - Session stats for display.
     * @param {string} [options.sortBy] - Current sort mode.
     * @param {Function} [options.onSortChange] - Callback when sort changes.
     * @param {string} [options.filterLabel] - Current filter label.
     * @param {Function} [options.onFilterChange] - Callback when filter changes.
     */
    render(options = {}) {
        this.el.innerHTML = '';

        // Frame navigation
        const frameNav = document.createElement('div');
        frameNav.style.cssText = 'display:flex;align-items:center;gap:6px;';

        const prevBtn = document.createElement('button');
        prevBtn.className = 'btn btn-ghost btn-small';
        prevBtn.textContent = 'Prev Frame';
        prevBtn.addEventListener('click', () => {
            if (options.onPrevFrame) options.onPrevFrame();
        });
        frameNav.appendChild(prevBtn);

        const nextBtn = document.createElement('button');
        nextBtn.className = 'btn btn-ghost btn-small';
        nextBtn.textContent = 'Next Frame';
        nextBtn.addEventListener('click', () => {
            if (options.onNextFrame) options.onNextFrame();
        });
        frameNav.appendChild(nextBtn);

        this.el.appendChild(frameNav);

        // Separator
        this.el.appendChild(this._separator());

        // Draw mode toggle
        const drawBtn = document.createElement('button');
        drawBtn.className = 'btn btn-secondary btn-small';
        if (options.drawMode) drawBtn.classList.add('active');
        drawBtn.textContent = options.drawMode ? 'Draw Mode ON' : 'Draw Mode';
        drawBtn.addEventListener('click', () => {
            if (options.onDrawToggle) options.onDrawToggle();
        });
        this.el.appendChild(drawBtn);

        // Separator
        this.el.appendChild(this._separator());

        // Train button
        const trainBtn = document.createElement('button');
        trainBtn.className = 'btn btn-primary btn-small';
        trainBtn.textContent = 'Train Classifier';
        trainBtn.addEventListener('click', () => {
            if (options.onTrain) options.onTrain();
        });
        this.el.appendChild(trainBtn);

        // Recall strategies dropdown
        const strategies = options.recallStrategies || ['multi_prompt', 'feature_search'];
        const dropdown = document.createElement('div');
        dropdown.className = 'dropdown';
        dropdown.style.position = 'relative';

        const dropBtn = document.createElement('button');
        dropBtn.className = 'btn btn-secondary btn-small';
        dropBtn.textContent = 'Recall Strategies';
        dropdown.appendChild(dropBtn);

        const menu = document.createElement('ul');
        menu.className = 'dropdown-menu';

        strategies.forEach((strategy) => {
            const li = document.createElement('li');
            li.textContent = strategy.replace(/_/g, ' ');
            li.addEventListener('click', () => {
                menu.classList.remove('open');
                if (options.onRecall) options.onRecall(strategy);
            });
            menu.appendChild(li);
        });
        dropdown.appendChild(menu);

        dropBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            menu.classList.toggle('open');
        });
        // Close dropdown on outside click
        document.addEventListener('click', () => {
            menu.classList.remove('open');
        }, { once: false });

        this.el.appendChild(dropdown);

        // Separator
        this.el.appendChild(this._separator());

        // Sort select
        const sortGroup = document.createElement('div');
        sortGroup.style.cssText = 'display:flex;align-items:center;gap:4px;';
        const sortLabel = document.createElement('span');
        sortLabel.style.cssText = 'font-size:0.75rem;color:var(--text-muted);';
        sortLabel.textContent = 'Sort:';
        sortGroup.appendChild(sortLabel);

        const sortSelect = document.createElement('select');
        sortSelect.style.cssText =
            'padding:3px 6px;font-size:0.75rem;background:var(--bg-body);' +
            'color:var(--text-primary);border:1px solid var(--border-default);' +
            'border-radius:var(--radius-sm);';
        ['uncertainty', 'cluster', 'frame'].forEach((s) => {
            const opt = document.createElement('option');
            opt.value = s;
            opt.textContent = s.charAt(0).toUpperCase() + s.slice(1);
            if (s === (options.sortBy || 'uncertainty')) opt.selected = true;
            sortSelect.appendChild(opt);
        });
        sortSelect.addEventListener('change', () => {
            if (options.onSortChange) options.onSortChange(sortSelect.value);
        });
        sortGroup.appendChild(sortSelect);
        this.el.appendChild(sortGroup);

        // Filter select
        const filterGroup = document.createElement('div');
        filterGroup.style.cssText = 'display:flex;align-items:center;gap:4px;';
        const filterLabel = document.createElement('span');
        filterLabel.style.cssText = 'font-size:0.75rem;color:var(--text-muted);';
        filterLabel.textContent = 'Filter:';
        filterGroup.appendChild(filterLabel);

        const filterSelect = document.createElement('select');
        filterSelect.style.cssText =
            'padding:3px 6px;font-size:0.75rem;background:var(--bg-body);' +
            'color:var(--text-primary);border:1px solid var(--border-default);' +
            'border-radius:var(--radius-sm);';
        ['all', 'pending', 'accepted', 'rejected'].forEach((f) => {
            const opt = document.createElement('option');
            opt.value = f;
            opt.textContent = f.charAt(0).toUpperCase() + f.slice(1);
            if (f === (options.filterLabel || 'all')) opt.selected = true;
            filterSelect.appendChild(opt);
        });
        filterSelect.addEventListener('change', () => {
            if (options.onFilterChange) options.onFilterChange(filterSelect.value);
        });
        filterGroup.appendChild(filterSelect);
        this.el.appendChild(filterGroup);

        // Separator
        this.el.appendChild(this._separator());

        // Stats display
        if (options.stats) {
            const statsEl = document.createElement('span');
            statsEl.style.cssText =
                'font-size:0.7rem;color:var(--text-muted);margin-left:auto;';
            const s = options.stats;
            statsEl.textContent =
                `${s.accepted || 0} accepted | ${s.rejected || 0} rejected | ` +
                `${s.pending || 0} pending | ${s.total_crops || 0} total`;
            this.el.appendChild(statsEl);
        }

        // Advance phase button
        if (options.onAdvancePhase) {
            const advBtn = document.createElement('button');
            advBtn.className = 'btn btn-primary btn-small';
            advBtn.style.marginLeft = '8px';
            advBtn.textContent = 'Next Phase';
            advBtn.addEventListener('click', () => options.onAdvancePhase());
            this.el.appendChild(advBtn);
        }
    }

    _separator() {
        const sep = document.createElement('div');
        sep.style.cssText =
            'width:1px;height:20px;background:var(--border-default);flex-shrink:0;';
        return sep;
    }

    destroy() {
        if (this.el && this.el.parentNode) {
            this.el.parentNode.removeChild(this.el);
        }
    }
}

// ---------------------------------------------------------------------------
// Modal
// ---------------------------------------------------------------------------

class Modal {
    /**
     * Show a confirmation dialog. Returns a Promise that resolves
     * to true (confirmed) or false (cancelled).
     * @param {string} title
     * @param {string} message
     * @returns {Promise<boolean>}
     */
    static confirm(title, message) {
        return new Promise((resolve) => {
            const container = document.getElementById('modal-container');
            container.innerHTML = '';
            container.setAttribute('aria-hidden', 'false');

            const box = document.createElement('div');
            box.className = 'modal-box';

            const h3 = document.createElement('h3');
            h3.textContent = title;
            box.appendChild(h3);

            const p = document.createElement('p');
            p.textContent = message;
            box.appendChild(p);

            const actions = document.createElement('div');
            actions.className = 'modal-actions';

            const cancelBtn = document.createElement('button');
            cancelBtn.className = 'btn btn-ghost';
            cancelBtn.textContent = 'Cancel';
            cancelBtn.addEventListener('click', () => {
                Modal._close(container);
                resolve(false);
            });

            const confirmBtn = document.createElement('button');
            confirmBtn.className = 'btn btn-primary';
            confirmBtn.textContent = 'Confirm';
            confirmBtn.addEventListener('click', () => {
                Modal._close(container);
                resolve(true);
            });

            actions.appendChild(cancelBtn);
            actions.appendChild(confirmBtn);
            box.appendChild(actions);
            container.appendChild(box);

            // Close on backdrop click
            container.addEventListener('click', function _backdrop(e) {
                if (e.target === container) {
                    container.removeEventListener('click', _backdrop);
                    Modal._close(container);
                    resolve(false);
                }
            });

            // Close on Escape
            const _esc = (e) => {
                if (e.key === 'Escape') {
                    document.removeEventListener('keydown', _esc);
                    Modal._close(container);
                    resolve(false);
                }
            };
            document.addEventListener('keydown', _esc);

            // Focus the confirm button
            requestAnimationFrame(() => confirmBtn.focus());
        });
    }

    /**
     * Show a generic modal with arbitrary HTML content.
     * Returns a close function.
     * @param {string} title
     * @param {string|HTMLElement} content - HTML string or DOM element.
     * @returns {Function} A function to call to close the modal.
     */
    static show(title, content) {
        const container = document.getElementById('modal-container');
        container.innerHTML = '';
        container.setAttribute('aria-hidden', 'false');

        const box = document.createElement('div');
        box.className = 'modal-box';

        const h3 = document.createElement('h3');
        h3.textContent = title;
        box.appendChild(h3);

        const body = document.createElement('div');
        body.style.marginBottom = '16px';
        if (typeof content === 'string') {
            body.innerHTML = content;
        } else if (content instanceof HTMLElement) {
            body.appendChild(content);
        }
        box.appendChild(body);

        const actions = document.createElement('div');
        actions.className = 'modal-actions';

        const closeBtn = document.createElement('button');
        closeBtn.className = 'btn btn-ghost';
        closeBtn.textContent = 'Close';
        closeBtn.addEventListener('click', () => closeFn());
        actions.appendChild(closeBtn);
        box.appendChild(actions);

        container.appendChild(box);

        const closeFn = () => Modal._close(container);

        container.addEventListener('click', function _backdrop(e) {
            if (e.target === container) {
                container.removeEventListener('click', _backdrop);
                closeFn();
            }
        });

        const _esc = (e) => {
            if (e.key === 'Escape') {
                document.removeEventListener('keydown', _esc);
                closeFn();
            }
        };
        document.addEventListener('keydown', _esc);

        return closeFn;
    }

    /** @private Close the modal container. */
    static _close(container) {
        container.setAttribute('aria-hidden', 'true');
        // Clear content after transition
        setTimeout(() => {
            container.innerHTML = '';
        }, 300);
    }
}
