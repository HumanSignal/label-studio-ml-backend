/**
 * ClusterMap — SVG force-directed cluster visualization.
 *
 * Pure vanilla JS, no external dependencies.
 * Animates merges/splits in real-time as pairs are resolved.
 */

/* global API, AppState */

var ClusterMap = (function () {
    'use strict';

    var _container = null;
    var _sessionId = null;
    var _svg = null;
    var _width = 0;
    var _height = 0;
    var _nodes = [];      // { id, cluster, x, y, vx, vy, track, frame }
    var _prevClusters = {};
    var _animFrame = null;
    var _running = false;

    // Cluster colors (10 distinct colors)
    var COLORS = [
        '#58a6ff', '#3fb950', '#f85149', '#d29922', '#a371f7',
        '#79c0ff', '#56d364', '#ff7b72', '#e3b341', '#bc8cff',
    ];

    function init(containerId, sessionId) {
        _container = document.getElementById(containerId);
        _sessionId = sessionId;

        var rect = _container.getBoundingClientRect();
        _width = rect.width || 400;
        _height = rect.height || 400;

        _svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        _svg.setAttribute('viewBox', '0 0 ' + _width + ' ' + _height);
        _svg.setAttribute('width', '100%');
        _svg.setAttribute('height', '100%');
        _container.appendChild(_svg);

        refresh();
        _startSimulation();
    }

    function refresh() {
        if (!_sessionId) return;
        API.get('/clusters', { session_id: _sessionId }).then(function (data) {
            _updateNodes(data);
            _render();
        });
    }

    function _updateNodes(data) {
        var clusters = data.clusters || {};
        var serverNodes = data.nodes || [];

        // Build lookup of existing node positions
        var existing = {};
        for (var i = 0; i < _nodes.length; i++) {
            existing[_nodes[i].id] = _nodes[i];
        }

        var newNodes = [];
        for (var j = 0; j < serverNodes.length; j++) {
            var sn = serverNodes[j];
            var prev = existing[sn.id];
            if (prev) {
                // Keep position, update cluster
                prev.cluster = sn.cluster;
                prev.track = sn.track;
                prev.frame = sn.frame;
                newNodes.push(prev);
            } else {
                // New node — random position near cluster center
                var cx = _width / 2 + (Math.random() - 0.5) * _width * 0.6;
                var cy = _height / 2 + (Math.random() - 0.5) * _height * 0.6;
                newNodes.push({
                    id: sn.id,
                    cluster: sn.cluster,
                    track: sn.track,
                    frame: sn.frame,
                    x: cx,
                    y: cy,
                    vx: 0,
                    vy: 0,
                });
            }
        }

        _nodes = newNodes;
        _prevClusters = clusters;
    }

    function _render() {
        if (!_svg) return;
        _svg.innerHTML = '';

        // Draw links between same-cluster nodes
        var linkGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        for (var i = 0; i < _nodes.length; i++) {
            for (var j = i + 1; j < _nodes.length; j++) {
                if (_nodes[i].cluster === _nodes[j].cluster) {
                    var line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                    line.setAttribute('x1', _nodes[i].x);
                    line.setAttribute('y1', _nodes[i].y);
                    line.setAttribute('x2', _nodes[j].x);
                    line.setAttribute('y2', _nodes[j].y);
                    line.setAttribute('class', 'cluster-link');
                    linkGroup.appendChild(line);
                }
            }
        }
        _svg.appendChild(linkGroup);

        // Draw nodes
        var nodeGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        for (var k = 0; k < _nodes.length; k++) {
            var n = _nodes[k];
            var circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            circle.setAttribute('cx', n.x);
            circle.setAttribute('cy', n.y);
            circle.setAttribute('r', 6);
            circle.setAttribute('fill', COLORS[n.cluster % COLORS.length]);
            circle.setAttribute('class', 'cluster-node');

            // Tooltip
            var title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
            title.textContent = n.id + ' (cluster ' + n.cluster + ')';
            circle.appendChild(title);

            nodeGroup.appendChild(circle);
        }
        _svg.appendChild(nodeGroup);

        // Draw cluster labels
        var clusterCenters = {};
        for (var m = 0; m < _nodes.length; m++) {
            var cl = _nodes[m].cluster;
            if (!clusterCenters[cl]) clusterCenters[cl] = { sx: 0, sy: 0, count: 0 };
            clusterCenters[cl].sx += _nodes[m].x;
            clusterCenters[cl].sy += _nodes[m].y;
            clusterCenters[cl].count++;
        }
        var labelGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        for (var clId in clusterCenters) {
            var cc = clusterCenters[clId];
            var text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            text.setAttribute('x', cc.sx / cc.count);
            text.setAttribute('y', cc.sy / cc.count - 12);
            text.setAttribute('class', 'cluster-label');
            text.textContent = 'ID ' + clId;
            labelGroup.appendChild(text);
        }
        _svg.appendChild(labelGroup);
    }

    // ------------------------------------------------------------------
    // Simple force simulation
    // ------------------------------------------------------------------

    function _startSimulation() {
        if (_running) return;
        _running = true;
        _tick();
    }

    function _tick() {
        if (!_running) return;

        var damping = 0.85;
        var repulsion = 200;
        var clusterAttraction = 0.02;
        var centerGravity = 0.001;

        // Compute cluster centers
        var centers = {};
        var counts = {};
        for (var i = 0; i < _nodes.length; i++) {
            var cl = _nodes[i].cluster;
            if (!centers[cl]) { centers[cl] = { x: 0, y: 0 }; counts[cl] = 0; }
            centers[cl].x += _nodes[i].x;
            centers[cl].y += _nodes[i].y;
            counts[cl]++;
        }
        for (var c in centers) {
            centers[c].x /= counts[c];
            centers[c].y /= counts[c];
        }

        // Apply forces
        for (var a = 0; a < _nodes.length; a++) {
            var node = _nodes[a];
            var fx = 0, fy = 0;

            // Repulsion from all other nodes
            for (var b = 0; b < _nodes.length; b++) {
                if (a === b) continue;
                var dx = node.x - _nodes[b].x;
                var dy = node.y - _nodes[b].y;
                var dist = Math.sqrt(dx * dx + dy * dy) || 1;
                var force = repulsion / (dist * dist);
                fx += (dx / dist) * force;
                fy += (dy / dist) * force;
            }

            // Attraction to cluster center
            var cc = centers[node.cluster];
            if (cc) {
                fx += (cc.x - node.x) * clusterAttraction;
                fy += (cc.y - node.y) * clusterAttraction;
            }

            // Gravity toward canvas center
            fx += (_width / 2 - node.x) * centerGravity;
            fy += (_height / 2 - node.y) * centerGravity;

            node.vx = (node.vx + fx) * damping;
            node.vy = (node.vy + fy) * damping;

            node.x += node.vx;
            node.y += node.vy;

            // Boundary clamping
            node.x = Math.max(10, Math.min(_width - 10, node.x));
            node.y = Math.max(10, Math.min(_height - 10, node.y));
        }

        _render();
        _animFrame = requestAnimationFrame(_tick);
    }

    function stop() {
        _running = false;
        if (_animFrame) cancelAnimationFrame(_animFrame);
    }

    return {
        init: init,
        refresh: refresh,
        stop: stop,
    };
})();
