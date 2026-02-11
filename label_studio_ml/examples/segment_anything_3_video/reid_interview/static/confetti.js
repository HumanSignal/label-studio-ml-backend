/**
 * Confetti — Lightweight canvas-based particle celebration.
 *
 * Confetti.burst(x, y, count) — spawn particles at (x, y)
 * Auto-creates/removes canvas as needed.
 */

var Confetti = (function () {
    'use strict';

    var _canvas = null;
    var _ctx = null;
    var _particles = [];
    var _running = false;
    var _animFrame = null;

    var COLORS = [
        '#58a6ff', '#3fb950', '#f85149', '#d29922', '#a371f7',
        '#79c0ff', '#56d364', '#ff7b72', '#e3b341', '#bc8cff',
        '#ffffff',
    ];

    function _ensureCanvas() {
        if (_canvas) return;
        _canvas = document.createElement('canvas');
        _canvas.id = 'confetti-canvas';
        _canvas.width = window.innerWidth;
        _canvas.height = window.innerHeight;
        document.body.appendChild(_canvas);
        _ctx = _canvas.getContext('2d');

        window.addEventListener('resize', function () {
            if (_canvas) {
                _canvas.width = window.innerWidth;
                _canvas.height = window.innerHeight;
            }
        });
    }

    function burst(x, y, count) {
        _ensureCanvas();
        count = count || 20;

        for (var i = 0; i < count; i++) {
            var angle = Math.random() * Math.PI * 2;
            var speed = 2 + Math.random() * 6;
            var size = 3 + Math.random() * 5;

            _particles.push({
                x: x,
                y: y,
                vx: Math.cos(angle) * speed,
                vy: Math.sin(angle) * speed - 2,
                size: size,
                color: COLORS[Math.floor(Math.random() * COLORS.length)],
                life: 1.0,
                decay: 0.015 + Math.random() * 0.02,
                rotation: Math.random() * Math.PI * 2,
                rotSpeed: (Math.random() - 0.5) * 0.3,
                shape: Math.random() > 0.5 ? 'rect' : 'circle',
            });
        }

        if (!_running) {
            _running = true;
            _animate();
        }
    }

    function _animate() {
        if (!_ctx || _particles.length === 0) {
            _running = false;
            if (_canvas) {
                _ctx.clearRect(0, 0, _canvas.width, _canvas.height);
            }
            return;
        }

        _ctx.clearRect(0, 0, _canvas.width, _canvas.height);

        var alive = [];
        for (var i = 0; i < _particles.length; i++) {
            var p = _particles[i];

            p.x += p.vx;
            p.y += p.vy;
            p.vy += 0.12; // gravity
            p.vx *= 0.99; // air resistance
            p.life -= p.decay;
            p.rotation += p.rotSpeed;

            if (p.life <= 0) continue;

            _ctx.save();
            _ctx.globalAlpha = p.life;
            _ctx.translate(p.x, p.y);
            _ctx.rotate(p.rotation);
            _ctx.fillStyle = p.color;

            if (p.shape === 'rect') {
                _ctx.fillRect(-p.size / 2, -p.size / 2, p.size, p.size * 0.6);
            } else {
                _ctx.beginPath();
                _ctx.arc(0, 0, p.size / 2, 0, Math.PI * 2);
                _ctx.fill();
            }

            _ctx.restore();
            alive.push(p);
        }

        _particles = alive;
        _animFrame = requestAnimationFrame(_animate);
    }

    return {
        burst: burst,
    };
})();
