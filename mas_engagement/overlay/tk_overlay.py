"""Tkinter overlays used by the Delivery agent.

Card design follows ``docs/demo_intervensi_3tingkat.html``:

* Cream card (#fffdf8), rounded corners (14 px on Windows via
  ``-transparentcolor``; clean rectangle fallback elsewhere).
* Header row: tier-coloured icon, the app name "EduAgent".
* Body text in modern sans-serif (Plus Jakarta Sans if installed, Segoe UI
  otherwise) at body weight 500.
* Bottom-edge 3 px countdown bar in the tier accent colour, animated from
  100 % width down to 0 % across the auto-dismiss timeout.
* Smooth slide-in from the right; slide-out on dismiss.

Two production entry points:

* :func:`show_toast` — Tier-1 / Tier-2 auto-dismissing card.
* :func:`show_tier3_widget` — Tier-3 persistent card with two buttons
  (Lanjutkan / Istirahat sebentar) and a wellness-break follow-up. No
  countdown — Tier-3 is intentionally persistent.

:func:`show_persistent_widget` is retained for legacy callers/tests.
"""
import threading
import time
from typing import Any, List, Optional

try:
    import tkinter as tk
    from tkinter import font as tkfont
    _TK_AVAILABLE = True
except ImportError:
    _TK_AVAILABLE = False


# ── Palette (mirrors demo_intervensi_3tingkat.html) ───────────────────────────

_TIER_ACCENT = {"1": "#5b9bd5", "2": "#e0a64a", "3": "#36a48c"}

_CARD_BG       = "#fffdf8"   # cream card surface
_CARD_BORDER   = "#e8e2d6"   # faint outline so the card defines on bright BGs
_INK           = "#1d2430"   # card primary ink
_INK_SOFT      = "#58616f"   # header app-name + ghost button label
_BTN_PRIMARY_INK = "#ffffff"
_BTN_GHOST_BORDER = "#d6d3c8"

# Magenta key used as the transparent colour (Windows). Pick a value that
# certainly does not appear inside the card. If the platform doesn't support
# -transparentcolor we fall back to a plain rectangle (no rounded corners).
_TRANSPARENT_KEY = "#ff00fe"


# ── Bahasa labels for the message-bank `intent` codes ─────────────────────────
# Kept for backward compatibility if other modules import this dictionary

INTENT_LABELS = {
    "link_to_material":          "Mengaitkan ke materi",
    "spark_curiosity":           "Memancing penasaran",
    "orient_to_screen":          "Menunjukkan posisi sekarang",
    "normalize_then_redirect":   "Normalisasi & arahkan kembali",
    "emphasize_importance":      "Tekankan pentingnya",
    "smallest_step":             "Langkah terkecil",
    "persistent_check_in":       "Cek-in lanjut",
}


def intent_to_purpose(intent: Optional[str]) -> Optional[str]:
    """Map a message-bank intent code to a Bahasa Indonesia label."""
    if not intent:
        return None
    return INTENT_LABELS.get(intent, intent.replace("_", " "))


# ── Layout constants ──────────────────────────────────────────────────────────

_CARD_WIDTH     = 340
_CARD_RADIUS    = 14
_ICON_SIZE      = 14
_ICON_RADIUS    = 4
_BAR_HEIGHT     = 3
_PAD_X          = 18
_PAD_TOP        = 14
_PAD_BOTTOM     = 14
_GAP_HEADER     = 11      # between header row and body
_GAP_BUTTONS    = 12      # between body and action buttons (Tier 3)
_MARGIN_RIGHT   = 24
_MARGIN_BOTTOM  = 96      # leaves room for Windows taskbar
_ALPHA          = 0.98
_TICK_MS        = 25      # ≈ 40 fps for animations
_SLIDE_FRAMES   = 14      # ≈ 350 ms slide duration


# ── Font selection (Plus Jakarta Sans with Segoe UI fallback) ─────────────────

def _pick_family() -> str:
    """Plus Jakarta Sans if installed on the system; else Segoe UI."""
    families = tkfont.families()
    if "Plus Jakarta Sans" in families:
        return "Plus Jakarta Sans"
    return "Segoe UI"


def _font(family: str, size: int, weight: str = "normal") -> tuple:
    if weight == "bold":
        return (family, size, "bold")
    return (family, size)


# ── Canvas helper: rounded rectangle via smooth polygon ───────────────────────

def _rounded_rect(canvas: "tk.Canvas", x1, y1, x2, y2, r, **kw):
    points = [
        x1 + r, y1, x2 - r, y1, x2, y1,
        x2,     y1 + r, x2, y2 - r, x2, y2,
        x2 - r, y2, x1 + r, y2, x1, y2,
        x1,     y2 - r, x1, y1 + r, x1, y1,
    ]
    return canvas.create_polygon(points, smooth=True, **kw)


def _try_transparent_window(win: "tk.Toplevel") -> bool:
    """Make the magenta key transparent. Returns True if accepted."""
    try:
        win.attributes("-transparentcolor", _TRANSPARENT_KEY)
        return True
    except tk.TclError:
        return False


def _measure_text_height(root, text: str, font: tuple, wraplength: int) -> int:
    """Return the rendered height in px of `text` wrapped at `wraplength`."""
    probe = tk.Label(root, text=text, font=font, wraplength=wraplength,
                     justify="left", anchor="w")
    probe.update_idletasks()
    h = probe.winfo_reqheight()
    probe.destroy()
    return max(h, 1)


# ── Public: Tier-1 / Tier-2 toast ─────────────────────────────────────────────

def show_toast(
    text: str,
    timeout: int = 8,
    *,
    tier: str = "1",
    header: str = "EduAgent",
    purpose: Optional[str] = None, # Kept so caller doesn't break, but ignored in UI
) -> str:
    """Cream-card auto-dismissing notification with a tier-coloured countdown.

    Returns ``"clicked"`` if the user clicks the card before timeout,
    ``"ignored"`` otherwise.
    """
    if not _TK_AVAILABLE:
        raise ImportError("tkinter is not available on this platform")

    outcome: List[str] = ["ignored"]
    done = threading.Event()
    accent = _TIER_ACCENT.get(str(tier), _TIER_ACCENT["1"])

    def _build() -> None:
        root = tk.Tk()
        root.withdraw()

        family = _pick_family()
        font_header  = _font(family, 9,  "bold")
        font_body    = _font(family, 11)

        win = tk.Toplevel(root)
        win.overrideredirect(True)
        win.attributes("-topmost", True)
        win.attributes("-alpha", _ALPHA)

        rounded_ok = _try_transparent_window(win)
        canvas_bg = _TRANSPARENT_KEY if rounded_ok else _CARD_BG
        win.configure(bg=canvas_bg)

        # ── Measure body so the card height fits the wrapped text ────────────
        wrap_w = _CARD_WIDTH - 2 * _PAD_X
        body_h = _measure_text_height(root, text, font_body, wrap_w)

        header_h  = _ICON_SIZE + 6   # icon + a bit of breathing room
        card_h = (
            _PAD_TOP + header_h
            + _GAP_HEADER + body_h
            + _PAD_BOTTOM
            + _BAR_HEIGHT
        )

        canvas = tk.Canvas(
            win, width=_CARD_WIDTH, height=card_h,
            bg=canvas_bg, highlightthickness=0, bd=0,
        )
        canvas.pack(fill="both", expand=True)

        # ── Card surface (rounded fill) + 1 px outline ───────────────────────
        _rounded_rect(
            canvas, 0, 0, _CARD_WIDTH, card_h - _BAR_HEIGHT,
            _CARD_RADIUS, fill=_CARD_BG, outline=_CARD_BORDER, width=1,
        )

        # ── Header: icon ┃ "EduAgent" ─────────────────────────────────────────
        icon_y1 = _PAD_TOP
        icon_y2 = icon_y1 + _ICON_SIZE
        _rounded_rect(
            canvas, _PAD_X, icon_y1, _PAD_X + _ICON_SIZE, icon_y2,
            _ICON_RADIUS, fill=accent, outline="",
        )
        canvas.create_text(
            _PAD_X + _ICON_SIZE + 8, (icon_y1 + icon_y2) // 2,
            text=header, anchor="w", font=font_header, fill=_INK_SOFT,
        )

        # ── Body ──────────────────────────────────────────────────────────────
        body_y = _PAD_TOP + header_h + _GAP_HEADER
        body_lbl = tk.Label(
            canvas, text=text, font=font_body,
            bg=_CARD_BG, fg=_INK,
            wraplength=wrap_w, justify="left", anchor="w",
        )
        canvas.create_window(_PAD_X, body_y, anchor="nw", window=body_lbl)

        # ── Countdown bar (drawn directly on the canvas) ──────────────────────
        bar_y1 = card_h - _BAR_HEIGHT
        bar_y2 = card_h
        bar_full_w = _CARD_WIDTH
        bar_id = canvas.create_rectangle(
            0, bar_y1, bar_full_w, bar_y2, fill=accent, outline="",
        )

        # ── Slide-in / slide-out plumbing ─────────────────────────────────────
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        target_x = sw - _CARD_WIDTH - _MARGIN_RIGHT
        target_y = sh - card_h - _MARGIN_BOTTOM
        start_x  = sw + 20

        win.geometry(f"{_CARD_WIDTH}x{card_h}+{start_x}+{target_y}")

        def _set_geom(x: int) -> bool:
            try:
                win.geometry(f"{_CARD_WIDTH}x{card_h}+{x}+{target_y}")
                return True
            except tk.TclError:
                return False

        def _destroy() -> None:
            try:
                win.destroy()
                root.destroy()
            except tk.TclError:
                pass

        def _slide_out_then_destroy() -> None:
            def _step(i: int) -> None:
                if done.is_set():
                    _destroy()
                    return
                frac = i / _SLIDE_FRAMES
                x = int(target_x + (start_x - target_x) * frac)
                if not _set_geom(x):
                    return
                if i < _SLIDE_FRAMES:
                    root.after(_TICK_MS, lambda: _step(i + 1))
                else:
                    done.set()
                    _destroy()
            _step(0)

        def _on_click(_e=None) -> None:
            if done.is_set():
                return
            outcome[0] = "clicked"
            _slide_out_then_destroy()

        for widget in (canvas, body_lbl):
            widget.bind("<Button-1>", _on_click)

        # ── Countdown animation: starts only after slide-in finishes ─────────
        deadline_holder: List[float] = [0.0]

        def _tick_countdown() -> None:
            if done.is_set():
                return
            rem = max(0.0, deadline_holder[0] - time.monotonic())
            frac = rem / timeout if timeout > 0 else 0.0
            new_w = max(0, int(bar_full_w * frac))
            try:
                canvas.coords(bar_id, 0, bar_y1, new_w, bar_y2)
            except tk.TclError:
                return
            if rem <= 0.0:
                outcome[0] = "ignored"
                _slide_out_then_destroy()
            else:
                root.after(_TICK_MS, _tick_countdown)

        def _slide_in() -> None:
            def _step(i: int) -> None:
                if done.is_set():
                    return
                frac = i / _SLIDE_FRAMES
                eased = 1 - (1 - frac) ** 3  # ease-out cubic
                x = int(start_x + (target_x - start_x) * eased)
                if not _set_geom(x):
                    return
                if i < _SLIDE_FRAMES:
                    root.after(_TICK_MS, lambda: _step(i + 1))
                else:
                    deadline_holder[0] = time.monotonic() + float(timeout)
                    _tick_countdown()
            _step(0)

        root.after(0, _slide_in)
        root.mainloop()

    t = threading.Thread(target=_build, daemon=True)
    t.start()
    done.wait()
    return outcome[0]


# ── Public: Tier-3 persistent widget (no countdown bar) ───────────────────────

def show_tier3_widget(
    prompt: str,
    continue_label: str,
    break_label: str,
    break_followup: str,
    break_nudge_sec: int = 5,
) -> str:
    """Persistent Tier-3 widget — cream card with two buttons, no auto-dismiss.

    On ``continue_label`` close immediately and return ``"continue"``.
    On ``break_label`` swap the body for ``break_followup``, remove the
    buttons, hold for ``break_nudge_sec`` seconds, then close, return
    ``"break"``.
    """
    if not _TK_AVAILABLE:
        raise ImportError("tkinter is not available on this platform")

    outcome: List[str] = [""]
    done = threading.Event()
    accent = _TIER_ACCENT["3"]

    def _build() -> None:
        root = tk.Tk()
        root.withdraw()

        family = _pick_family()
        font_header = _font(family, 9,  "bold")
        font_body   = _font(family, 11)
        font_btn    = _font(family, 10, "bold")

        win = tk.Toplevel(root)
        win.overrideredirect(True)
        win.attributes("-topmost", True)
        win.attributes("-alpha", _ALPHA)

        rounded_ok = _try_transparent_window(win)
        canvas_bg = _TRANSPARENT_KEY if rounded_ok else _CARD_BG
        win.configure(bg=canvas_bg)

        wrap_w = _CARD_WIDTH - 2 * _PAD_X
        header_h = _ICON_SIZE + 6
        body_h = _measure_text_height(root, prompt, font_body, wrap_w)
        btn_row_h = 36

        card_h = (
            _PAD_TOP + header_h
            + _GAP_HEADER + body_h
            + _GAP_BUTTONS + btn_row_h
            + _PAD_BOTTOM
        )

        canvas = tk.Canvas(
            win, width=_CARD_WIDTH, height=card_h,
            bg=canvas_bg, highlightthickness=0, bd=0,
        )
        canvas.pack(fill="both", expand=True)

        _rounded_rect(
            canvas, 0, 0, _CARD_WIDTH, card_h,
            _CARD_RADIUS, fill=_CARD_BG, outline=_CARD_BORDER, width=1,
        )

        # Header — icon ┃ "EduAgent" 
        icon_y1 = _PAD_TOP
        icon_y2 = icon_y1 + _ICON_SIZE
        _rounded_rect(
            canvas, _PAD_X, icon_y1, _PAD_X + _ICON_SIZE, icon_y2,
            _ICON_RADIUS, fill=accent, outline="",
        )
        canvas.create_text(
            _PAD_X + _ICON_SIZE + 8, (icon_y1 + icon_y2) // 2,
            text="EduAgent", anchor="w", font=font_header, fill=_INK_SOFT,
        )

        # Body — prompt
        body_y = _PAD_TOP + header_h + _GAP_HEADER
        body_lbl = tk.Label(
            canvas, text=prompt, font=font_body,
            bg=_CARD_BG, fg=_INK,
            wraplength=wrap_w, justify="left", anchor="w",
        )
        canvas.create_window(_PAD_X, body_y, anchor="nw", window=body_lbl)

        # Button row — Lanjutkan (primary teal) + Istirahat sebentar (ghost)
        btn_row = tk.Frame(canvas, bg=_CARD_BG)
        canvas.create_window(
            _PAD_X, body_y + body_h + _GAP_BUTTONS,
            anchor="nw", window=btn_row, width=wrap_w,
        )

        def _destroy() -> None:
            try:
                win.destroy()
                root.destroy()
            except tk.TclError:
                pass

        def _on_continue() -> None:
            outcome[0] = "continue"
            done.set()
            _destroy()

        def _on_break() -> None:
            outcome[0] = "break"
            body_lbl.configure(text=break_followup)
            try:
                btn_row.destroy()
            except tk.TclError:
                pass
            # Hold the wellness nudge for break_nudge_sec then close.
            root.after(int(break_nudge_sec * 1000), lambda: (done.set(), _destroy()))

        primary = tk.Button(
            btn_row, text=continue_label, command=_on_continue,
            bg=accent, fg=_BTN_PRIMARY_INK,
            activebackground=accent, activeforeground=_BTN_PRIMARY_INK,
            font=font_btn, relief="flat", bd=0,
            padx=14, pady=6, cursor="hand2",
        )
        primary.pack(side="left", padx=(0, 8))

        ghost = tk.Button(
            btn_row, text=break_label, command=_on_break,
            bg=_CARD_BG, fg=_INK_SOFT,
            activebackground=_CARD_BG, activeforeground=_INK,
            font=font_btn, relief="solid", bd=1,
            highlightbackground=_BTN_GHOST_BORDER,
            padx=12, pady=5, cursor="hand2",
        )
        ghost.pack(side="left")

        # Slide-in
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        target_x = sw - _CARD_WIDTH - _MARGIN_RIGHT
        target_y = sh - card_h - _MARGIN_BOTTOM
        start_x  = sw + 20
        win.geometry(f"{_CARD_WIDTH}x{card_h}+{start_x}+{target_y}")

        def _slide_in() -> None:
            def _step(i: int) -> None:
                if done.is_set():
                    return
                frac = i / _SLIDE_FRAMES
                eased = 1 - (1 - frac) ** 3
                x = int(start_x + (target_x - start_x) * eased)
                try:
                    win.geometry(f"{_CARD_WIDTH}x{card_h}+{x}+{target_y}")
                except tk.TclError:
                    return
                if i < _SLIDE_FRAMES:
                    root.after(_TICK_MS, lambda: _step(i + 1))
            _step(0)

        root.after(0, _slide_in)
        root.mainloop()

    t = threading.Thread(target=_build, daemon=True)
    t.start()
    done.wait()
    return outcome[0]


# ── Legacy: kept for older callers / tests. Not used by Delivery anymore. ─────

def show_persistent_widget(text: str, options: List[str]) -> str:
    """Legacy persistent widget — light-themed, returns the chosen option."""
    if not _TK_AVAILABLE:
        raise ImportError("tkinter is not available on this platform")

    outcome: List[str] = [""]
    done = threading.Event()

    def _build() -> None:
        root = tk.Tk()
        root.withdraw()
        family = _pick_family()
        font_body = _font(family, 11)
        font_btn  = _font(family, 10, "bold")

        win = tk.Toplevel(root)
        win.overrideredirect(True)
        win.attributes("-topmost", True)
        win.attributes("-alpha", _ALPHA)
        win.configure(bg=_CARD_BG)

        body_lbl = tk.Label(
            win, text=text, font=font_body,
            bg=_CARD_BG, fg=_INK,
            wraplength=_CARD_WIDTH - 2 * _PAD_X, justify="left", anchor="w",
            padx=_PAD_X,
        )
        body_lbl.pack(fill="x", pady=(_PAD_TOP, _PAD_BOTTOM))

        btn_row = tk.Frame(win, bg=_CARD_BG, padx=_PAD_X)
        btn_row.pack(fill="x", pady=(0, _PAD_BOTTOM))

        def _make_handler(label: str):
            def _h() -> None:
                outcome[0] = label
                done.set()
                try:
                    win.destroy()
                    root.destroy()
                except tk.TclError:
                    pass
            return _h

        for opt in options:
            btn = tk.Button(
                btn_row, text=opt, command=_make_handler(opt),
                bg=_CARD_BG, fg=_INK_SOFT,
                activebackground=_CARD_BG, activeforeground=_INK,
                font=font_btn, relief="solid", bd=1,
                padx=12, pady=5, cursor="hand2",
            )
            btn.pack(side="left", padx=(0, 8))

        root.update_idletasks()
        w = max(win.winfo_reqwidth(), _CARD_WIDTH)
        h = win.winfo_reqheight()
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        win.geometry(f"{w}x{h}+{sw - w - _MARGIN_RIGHT}+{sh - h - _MARGIN_BOTTOM}")
        root.mainloop()

    t = threading.Thread(target=_build, daemon=True)
    t.start()
    done.wait()
    return outcome[0]