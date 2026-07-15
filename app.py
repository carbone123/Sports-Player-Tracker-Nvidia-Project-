from flask import Flask, request, render_template_string, url_for
from ultralytics import YOLO
import cv2, subprocess, os, uuid, collections, threading
import numpy as np

app = Flask(__name__)
BASE = os.path.expanduser("~/soccer_webapp")
UPLOAD_DIR = os.path.join(BASE, "uploads")
OUTPUT_DIR = os.path.join(BASE, "static", "outputs")
FRAMES_DIR = os.path.join(BASE, "static", "frames")
MODEL_PATH = os.path.expanduser("~/yolov8football/yolov8m-football.pt")
TRACKER_PATH = os.path.expanduser("~/custom_botsort.yaml")

for d in (UPLOAD_DIR, OUTPUT_DIR, FRAMES_DIR):
    os.makedirs(d, exist_ok=True)

model = YOLO(MODEL_PATH)

SESSIONS = {}
PROGRESS = {}
RENDER_RESULTS = {}

PLAYER_COLOR = (129, 222, 74)
GK_COLOR = (250, 165, 96)
BALL_COLOR = (21, 204, 250)

FAVICON = '<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 viewBox=%270 0 100 100%27%3E%3Ctext y=%27.9em%27 font-size=%2790%27%3E%E2%9A%BD%3C/text%3E%3C/svg%3E">'

class BoxSmoother:
    def __init__(self, alpha=0.35, hold_frames=6):
        self.alpha = alpha
        self.hold = hold_frames
        self.state = {}

    def update(self, key, box, frame_idx):
        s = self.state.get(key)
        if s is None:
            smoothed = list(box)
        else:
            smoothed = [self.alpha * b + (1 - self.alpha) * sb for b, sb in zip(box, s["box"])]
        self.state[key] = {"box": smoothed, "last_seen": frame_idx}
        return smoothed

    def get_held(self, key, frame_idx):
        s = self.state.get(key)
        if s is None:
            return None
        if 0 < frame_idx - s["last_seen"] <= self.hold:
            return s["box"]
        return None


def draw_corner_box(frame, x1, y1, x2, y2, color, thickness=2, corner_len=16, glow=True):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    if glow:
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), color, 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    pts = [
        ((x1, y1), (x1 + corner_len, y1), (x1, y1 + corner_len)),
        ((x2, y1), (x2 - corner_len, y1), (x2, y1 + corner_len)),
        ((x1, y2), (x1 + corner_len, y2), (x1, y2 - corner_len)),
        ((x2, y2), (x2 - corner_len, y2), (x2, y2 - corner_len)),
    ]
    for corner, h_end, v_end in pts:
        cv2.line(frame, corner, h_end, color, thickness, cv2.LINE_AA)
        cv2.line(frame, corner, v_end, color, thickness, cv2.LINE_AA)


def draw_label_pill(frame, x, y, text, color, text_color=(15, 15, 15)):
    x, y = int(x), int(y)
    font = cv2.FONT_HERSHEY_DUPLEX
    scale, thick = 0.52, 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    pad_x, pad_y = 9, 6
    box_w, box_h = tw + pad_x * 2, th + pad_y * 2
    x1, y1 = x, max(0, y - box_h - 4)
    x2, y2 = x1 + box_w, y1 + box_h
    r = box_h // 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.circle(overlay, (x1 + r, y1 + r), r, color, -1, cv2.LINE_AA)
    cv2.circle(overlay, (x2 - r, y1 + r), r, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)
    cv2.putText(frame, text, (x1 + pad_x, y2 - pad_y), font, scale, text_color, thick, cv2.LINE_AA)


def draw_tapering_trail(frame, trail, color, max_points=40):
    pts = trail[-max_points:]
    n = len(pts)
    for i in range(1, n):
        thickness = max(1, round(1 + 4 * (i / max(1, n - 1))))
        cv2.line(frame, pts[i - 1], pts[i], color, thickness, cv2.LINE_AA)


def sample_frames_for_groups(per_frame, tid_to_rep, max_samples=10):
    group_frames = collections.defaultdict(list)
    for fidx, dets in per_frame.items():
        for d in dets:
            if d["id"] in tid_to_rep:
                group_frames[tid_to_rep[d["id"]]].append((fidx, d["box"]))
    sample_map = collections.defaultdict(list)
    for rep, entries in group_frames.items():
        entries.sort(key=lambda x: x[0])
        if len(entries) <= max_samples:
            chosen = entries
        else:
            step = len(entries) / max_samples
            chosen = [entries[int(i * step)] for i in range(max_samples)]
        for fidx, box in chosen:
            sample_map[fidx].append((rep, box))
    return sample_map


def compute_team_colors(video_path, sample_map):
    accum = collections.defaultdict(list)
    cap = cv2.VideoCapture(video_path)
    fidx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fidx in sample_map:
            for rep, box in sample_map[fidx]:
                x1, y1, x2, y2 = [int(v) for v in box]
                x1, y1 = max(0, x1), max(0, y1)
                bh = y2 - y1
                ty1, ty2 = y1 + int(bh * 0.15), y1 + int(bh * 0.55)
                crop = frame[ty1:ty2, x1:x2]
                if crop.size == 0:
                    continue
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
                non_green = cv2.bitwise_not(green_mask)
                pixels = crop[non_green > 0]
                if pixels.shape[0] < 10:
                    pixels = crop.reshape(-1, 3)
                accum[rep].append(pixels.mean(axis=0))
        fidx += 1
    cap.release()
    return {rep: np.mean(np.array(v), axis=0) for rep, v in accum.items() if v}


def kmeans_2(colors_dict, iters=12):
    reps = list(colors_dict.keys())
    if len(reps) < 2:
        centers = np.array([colors_dict[r] for r in reps] * 2)[:2] if reps else np.array([[120, 120, 120], [120, 120, 120]])
        return {r: 0 for r in reps}, centers
    X = np.array([colors_dict[r] for r in reps], dtype=np.float32)
    c0 = X[0]
    dists = np.linalg.norm(X - c0, axis=1)
    c1 = X[np.argmax(dists)]
    centers = np.stack([c0, c1])
    labels = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        d0 = np.linalg.norm(X - centers[0], axis=1)
        d1 = np.linalg.norm(X - centers[1], axis=1)
        labels = (d1 < d0).astype(int)
        new_centers = []
        for k in range(2):
            pts = X[labels == k]
            new_centers.append(pts.mean(axis=0) if len(pts) else centers[k])
        centers = np.stack(new_centers)
    return {r: int(labels[i]) for i, r in enumerate(reps)}, centers


def bgr_to_css(color):
    b, g, r = [max(0, min(255, int(v))) for v in color]
    return f"rgb({r},{g},{b})"


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
* { box-sizing: border-box; }
body { margin: 0; min-height: 100vh; overflow-x: hidden; background: #08090b; color: #eee; font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; display: flex; align-items: center; justify-content: center; padding: 24px; position: relative; }
.blob { position: fixed; border-radius: 50%; filter: blur(90px); opacity: 0.35; z-index: 0; animation: float 14s ease-in-out infinite; }
.blob1 { width: 420px; height: 420px; background: #22c55e; top: -160px; left: -140px; }
.blob2 { width: 380px; height: 380px; background: #3b82f6; bottom: -160px; right: -120px; animation-delay: -6s; }
@keyframes float { 0%, 100% { transform: translate(0, 0) scale(1); } 50% { transform: translate(30px, -20px) scale(1.08); } }
.card { position: relative; z-index: 1; background: rgba(20, 22, 26, 0.72); backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px); border: 1px solid rgba(255,255,255,0.08); border-radius: 24px; max-width: 700px; width: 100%; padding: 44px; box-shadow: 0 24px 70px rgba(0,0,0,0.55), inset 0 1px 0 rgba(255,255,255,0.04); animation: rise 0.5s cubic-bezier(0.16, 1, 0.3, 1); }
@keyframes rise { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: translateY(0); } }
.steps { display: flex; align-items: center; gap: 8px; margin-bottom: 20px; }
.step-dot { width: 7px; height: 7px; border-radius: 50%; background: #2e3138; transition: all 0.3s ease; }
.step-dot.active { background: #4ade80; width: 20px; border-radius: 4px; }
.step-label { font-size: 12px; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; color: #6b7280; margin-left: 4px; }
h1 { font-size: 28px; font-weight: 800; margin: 0 0 8px; letter-spacing: -0.6px; background: linear-gradient(135deg, #fff, #b8bcc4); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
.subtitle { color: #8a8f98; margin: 0 0 26px; font-size: 15px; line-height: 1.6; }
.toggle-group { display: flex; gap: 6px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 14px; padding: 5px; margin-bottom: 22px; }
.toggle-option { flex: 1; position: relative; }
.toggle-option input { position: absolute; opacity: 0; width: 100%; height: 100%; cursor: pointer; margin: 0; inset: 0; }
.toggle-label { display: flex; flex-direction: column; align-items: center; text-align: center; gap: 2px; padding: 13px 8px; border-radius: 10px; font-size: 14px; font-weight: 600; color: #8a8f98; transition: all 0.2s ease; }
.toggle-option input:checked + .toggle-label { background: linear-gradient(135deg, #4ade80, #16a34a); color: #052e12; box-shadow: 0 4px 16px rgba(74, 222, 128, 0.35); }
.toggle-sub { font-size: 11px; font-weight: 500; opacity: 0.8; }
.dropzone { border: 1.5px dashed rgba(255,255,255,0.14); background: rgba(255,255,255,0.02); border-radius: 18px; padding: 46px 20px; text-align: center; cursor: pointer; transition: all 0.2s ease; display: flex; flex-direction: column; align-items: center; gap: 10px; }
.dropzone:hover { border-color: rgba(74, 222, 128, 0.4); background: rgba(74, 222, 128, 0.03); }
.dropzone.drag { border-color: #4ade80; background: rgba(74, 222, 128, 0.06); transform: scale(1.01); }
.dz-icon { width: 40px; height: 40px; opacity: 0.5; }
.dz-text { color: #d4d7dc; font-size: 15px; font-weight: 600; margin: 0; }
.dz-sub { color: #5f6570; font-size: 13px; margin: 0; }
input[type=file] { display: none; }
button, .select-wrap { width: 100%; border: none; padding: 16px; border-radius: 13px; font-weight: 700; font-size: 15px; cursor: pointer; margin-top: 18px; }
button { background: linear-gradient(135deg, #4ade80, #16a34a); color: #052e12; transition: transform 0.15s ease, box-shadow 0.15s ease; box-shadow: 0 6px 20px rgba(74, 222, 128, 0.25); display: flex; align-items: center; justify-content: center; gap: 8px; }
button:hover:not(:disabled) { transform: translateY(-2px); box-shadow: 0 10px 28px rgba(74, 222, 128, 0.35); }
button:active:not(:disabled) { transform: translateY(0); }
button:disabled { background: #24272e; color: #666; cursor: default; box-shadow: none; }
.select-wrap { position: relative; padding: 0; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); }
select { width: 100%; appearance: none; -webkit-appearance: none; background: transparent; border: none; color: #eee; font-weight: 600; font-size: 15px; padding: 15px 40px 15px 16px; cursor: pointer; }
.select-wrap::after { content: ''; position: absolute; right: 16px; top: 50%; width: 8px; height: 8px; border-right: 2px solid #8a8f98; border-bottom: 2px solid #8a8f98; transform: translateY(-70%) rotate(45deg); pointer-events: none; }
select option { background: #14161a; }
.spinner { width: 15px; height: 15px; border: 2.5px solid rgba(5,46,18,0.25); border-top-color: #052e12; border-radius: 50%; animation: spin 0.7s linear infinite; flex-shrink: 0; }
@keyframes spin { to { transform: rotate(360deg); } }
.result-badge { display: inline-flex; align-items: center; gap: 8px; color: #4ade80; font-weight: 700; font-size: 13px; background: rgba(74, 222, 128, 0.1); padding: 7px 14px; border-radius: 20px; margin-bottom: 18px; letter-spacing: 0.3px; text-transform: uppercase; animation: pulse-once 0.6s ease; }
@keyframes pulse-once { 0% { transform: scale(0.9); opacity: 0; } 60% { transform: scale(1.04); } 100% { transform: scale(1); opacity: 1; } }
video, img { width: 100%; border-radius: 16px; display: block; background: #000; border: 1px solid rgba(255,255,255,0.06); }
.legend { display: flex; gap: 18px; margin-top: 16px; font-size: 13px; color: #9199a6; flex-wrap: wrap; }
.legend span { display: inline-flex; align-items: center; gap: 7px; }
.dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
.actions { display: flex; gap: 10px; margin-top: 20px; }
.actions a, .actions button { margin-top: 0; }
a.link-btn { flex: 1; text-align: center; text-decoration: none; padding: 14px; border-radius: 13px; font-weight: 700; font-size: 14px; background: rgba(255,255,255,0.04); color: #d4d7dc; border: 1px solid rgba(255,255,255,0.08); transition: all 0.2s ease; display: flex; align-items: center; justify-content: center; gap: 8px; }
a.link-btn:hover { background: rgba(255,255,255,0.08); border-color: rgba(255,255,255,0.16); }
.count-pill { display: inline-block; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08); color: #b8bcc4; font-size: 12px; font-weight: 600; padding: 4px 10px; border-radius: 20px; margin-bottom: 12px; }
.team-options { display: flex; flex-direction: column; gap: 10px; margin-bottom: 6px; }
.team-option { position: relative; }
.team-option input { position: absolute; opacity: 0; width: 100%; height: 100%; cursor: pointer; margin: 0; inset: 0; }
.team-card { display: flex; align-items: center; gap: 12px; padding: 14px 16px; border-radius: 13px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); transition: all 0.2s ease; }
.team-option input:checked + .team-card { border-color: #4ade80; background: rgba(74, 222, 128, 0.08); }
.swatch { width: 22px; height: 22px; border-radius: 50%; border: 2px solid rgba(255,255,255,0.2); flex-shrink: 0; }
.team-card .tlabel { font-weight: 600; font-size: 14px; }
.team-card .tsub { font-size: 12px; color: #8a8f98; margin-left: auto; }
.progress-track { width: 100%; height: 10px; background: rgba(255,255,255,0.06); border-radius: 6px; overflow: hidden; margin-top: 26px; }
.progress-fill { height: 100%; width: 0%; background: linear-gradient(90deg, #4ade80, #16a34a); border-radius: 6px; transition: width 0.3s ease; }
.progress-meta { display: flex; justify-content: space-between; margin-top: 10px; font-size: 13px; }
.progress-phase { color: #b8bcc4; font-weight: 500; }
.progress-pct { color: #4ade80; font-weight: 700; }
.progress-detail { color: #5f6570; font-size: 12px; margin-top: 6px; }
"""

ICON_UPLOAD = '<svg class="dz-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 16V4M12 4L7 9M12 4l5 5"/><path d="M4 16v3a2 2 0 002 2h12a2 2 0 002-2v-3"/></svg>'
ICON_PLAY = '<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>'
ICON_BACK = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M19 12H5M12 19l-7-7 7-7"/></svg>'
ICON_DOWNLOAD = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M12 3v12M7 10l5 5 5-5"/><path d="M5 21h14"/></svg>'
BLOBS = '<div class="blob blob1"></div><div class="blob blob2"></div>'

UPLOAD_PAGE = """
<!DOCTYPE html><html><head>""" + FAVICON + """<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Soccer Vision</title><style>""" + CSS + """</style></head><body>
""" + BLOBS + """
<div class="card">
  <div class="steps"><div class="step-dot active"></div><div class="step-dot"></div><span class="step-label">Upload</span></div>
  <h1>Soccer Ball &amp; Player Tracker</h1>
  <p class="subtitle">Upload match footage &mdash; players and the ball get detected and tracked automatically, on-device.</p>
  <form method="POST" action="/upload" enctype="multipart/form-data" id="uploadForm">
    <div class="toggle-group">
      <div class="toggle-option">
        <input type="radio" name="mode" id="modePlayer" value="player" checked>
        <label class="toggle-label" for="modePlayer">Track One Player<span class="toggle-sub">pick a player to follow</span></label>
      </div>
      <div class="toggle-option">
        <input type="radio" name="mode" id="modeTeam" value="team">
        <label class="toggle-label" for="modeTeam">Show Team<span class="toggle-sub">choose which side</span></label>
      </div>
    </div>
    <label class="dropzone" id="dropzone">
      """ + ICON_UPLOAD + """
      <p class="dz-text" id="dzText">Drag a video here, or click to browse</p>
      <p class="dz-sub">MP4, MOV, or AVI</p>
      <input type="file" name="video" id="fileInput" accept="video/*" required>
    </label>
    <button type="submit" id="submitBtn">""" + ICON_PLAY + """ Run Detection &amp; Tracking</button>
  </form>
</div>
<script>
  const dz = document.getElementById('dropzone');
  const input = document.getElementById('fileInput');
  const dzText = document.getElementById('dzText');
  const form = document.getElementById('uploadForm');
  const btn = document.getElementById('submitBtn');
  dz.addEventListener('click', () => input.click());
  dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('drag'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('drag'));
  dz.addEventListener('drop', e => { e.preventDefault(); dz.classList.remove('drag'); input.files = e.dataTransfer.files; dzText.textContent = input.files[0].name; });
  input.addEventListener('change', () => { if (input.files[0]) dzText.textContent = input.files[0].name; });
  form.addEventListener('submit', () => { btn.disabled = true; btn.innerHTML = '<span class="spinner"></span>Uploading...'; });
</script>
</body></html>
"""

def progress_page(title, heading, subtitle, status_url):
    return """
<!DOCTYPE html><html><head>""" + FAVICON + """<meta name="viewport" content="width=device-width, initial-scale=1">
<title>""" + title + """</title><style>""" + CSS + """</style></head><body>
""" + BLOBS + """
<div class="card">
  <div class="steps"><div class="step-dot active"></div><div class="step-dot"></div><span class="step-label">Processing</span></div>
  <h1>""" + heading + """</h1>
  <p class="subtitle">""" + subtitle + """</p>
  <div class="progress-track"><div class="progress-fill" id="bar"></div></div>
  <div class="progress-meta">
    <span class="progress-phase" id="phase">Starting...</span>
    <span class="progress-pct" id="pct">0%</span>
  </div>
  <div class="progress-detail" id="detail"></div>
</div>
<script>
  async function poll() {
    try {
      const res = await fetch('""" + status_url + """');
      const data = await res.json();
      document.getElementById('bar').style.width = data.percent + '%';
      document.getElementById('pct').textContent = data.percent + '%';
      document.getElementById('phase').textContent = data.phase;
      if (data.frame && data.total_frames) {
        document.getElementById('detail').textContent = 'Frame ' + data.frame + ' of ' + data.total_frames;
      }
      if (data.status === 'done') { window.location.href = data.next; return; }
      if (data.status === 'error') { document.getElementById('phase').textContent = 'Error: ' + (data.error_msg || 'something went wrong'); return; }
    } catch (e) {}
    setTimeout(poll, 500);
  }
  poll();
</script>
</body></html>
"""

SELECT_PAGE = """
<!DOCTYPE html><html><head>""" + FAVICON + """<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pick a Player</title><style>""" + CSS + """</style></head><body>
""" + BLOBS + """
<div class="card">
  <div class="steps"><div class="step-dot active"></div><div class="step-dot active"></div><span class="step-label">Select Player</span></div>
  <h1>Pick a player to follow</h1>
  <span class="count-pill">{{ options|length }} player(s) detected</span>
  <p class="subtitle">Choose the player you want to track through the whole clip.</p>
  <img src="{{ url_for('static', filename='frames/' + frame_img) }}">
  <form method="POST" action="/render/{{ uid }}" id="renderForm">
    <div class="select-wrap">
      <select name="track_id">
        {% for tid, label in options %}<option value="{{ tid }}">{{ label }}</option>{% endfor %}
      </select>
    </div>
    <button type="submit" id="renderBtn">""" + ICON_PLAY + """ Track This Player</button>
  </form>
  <a class="link-btn" href="/" style="margin-top:10px;">""" + ICON_BACK + """ Start over</a>
</div>
<script>
  document.getElementById('renderForm').addEventListener('submit', () => {
    const b = document.getElementById('renderBtn'); b.disabled = true;
    b.innerHTML = '<span class="spinner"></span>Starting render...';
  });
</script>
</body></html>
"""

TEAM_SELECT_PAGE = """
<!DOCTYPE html><html><head>""" + FAVICON + """<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pick a Team</title><style>""" + CSS + """</style></head><body>
""" + BLOBS + """
<div class="card">
  <div class="steps"><div class="step-dot active"></div><div class="step-dot active"></div><span class="step-label">Select Team</span></div>
  <h1>Which side do you want to see?</h1>
  <span class="count-pill">{{ team_a_count }} vs {{ team_b_count }} players &middot; grouped by kit color</span>
  <p class="subtitle">Detected automatically from jersey color &mdash; double check against the preview below before choosing.</p>
  <img src="{{ url_for('static', filename='frames/' + frame_img) }}">
  <form method="POST" action="/render_team/{{ uid }}" id="teamForm">
    <div class="team-options">
      <label class="team-option">
        <input type="radio" name="team" value="both" checked>
        <div class="team-card"><span class="swatch" style="background:linear-gradient(90deg, {{ color_a }} 50%, {{ color_b }} 50%)"></span><span class="tlabel">Both Teams</span><span class="tsub">everyone + ball</span></div>
      </label>
      <label class="team-option">
        <input type="radio" name="team" value="a">
        <div class="team-card"><span class="swatch" style="background:{{ color_a }}"></span><span class="tlabel">Team A</span><span class="tsub">{{ team_a_count }} players</span></div>
      </label>
      <label class="team-option">
        <input type="radio" name="team" value="b">
        <div class="team-card"><span class="swatch" style="background:{{ color_b }}"></span><span class="tlabel">Team B</span><span class="tsub">{{ team_b_count }} players</span></div>
      </label>
    </div>
    <button type="submit" id="teamBtn">""" + ICON_PLAY + """ Render Selection</button>
  </form>
  <a class="link-btn" href="/" style="margin-top:10px;">""" + ICON_BACK + """ Start over</a>
</div>
<script>
  document.getElementById('teamForm').addEventListener('submit', () => {
    const b = document.getElementById('teamBtn'); b.disabled = true;
    b.innerHTML = '<span class="spinner"></span>Starting render...';
  });
</script>
</body></html>
"""

RESULT_PAGE = """
<!DOCTYPE html><html><head>""" + FAVICON + """<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Result</title><style>""" + CSS + """</style></head><body>
""" + BLOBS + """
<div class="card">
  <div class="result-badge">✓ {{ headline }}</div>
  <video controls autoplay muted loop>
    <source src="{{ url_for('static', filename='outputs/' + result) }}" type="video/mp4">
  </video>
  {% if show_legend %}
  <div class="legend">
    <span><span class="dot" style="background:#81de4a"></span>Player</span>
    <span><span class="dot" style="background:#60a5fa"></span>Goalkeeper</span>
    <span><span class="dot" style="background:#facc15"></span>Ball</span>
  </div>
  {% endif %}
  <div class="actions">
    <a class="link-btn" href="/">""" + ICON_BACK + """ Upload another</a>
    <a class="link-btn" href="{{ url_for('static', filename='outputs/' + result) }}" download>""" + ICON_DOWNLOAD + """ Download</a>
  </div>
</div>
</body></html>
"""


def group_player_tracks(id_frames, max_gap=45, max_dist=140):
    parent = {tid: tid for tid in id_frames}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    def center(box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    for _ in range(4):
        changed = False
        group_span = {}
        for tid, frames in id_frames.items():
            root = find(tid)
            first_frame, first_box = frames[0]
            last_frame, last_box = frames[-1]
            if root not in group_span:
                group_span[root] = [first_frame, first_box, last_frame, last_box]
            else:
                if first_frame < group_span[root][0]:
                    group_span[root][0], group_span[root][1] = first_frame, first_box
                if last_frame > group_span[root][2]:
                    group_span[root][2], group_span[root][3] = last_frame, last_box
        roots = list(group_span.keys())
        for r1 in roots:
            if find(r1) != r1:
                continue
            last_frame, last_box = group_span[r1][2], group_span[r1][3]
            last_center = center(last_box)
            best_candidate, best_gap = None, None
            for r2 in roots:
                if r2 == r1 or find(r2) == find(r1):
                    continue
                first_frame, first_box = group_span[r2][0], group_span[r2][1]
                gap = first_frame - last_frame
                if 0 < gap <= max_gap:
                    dist = ((center(first_box)[0] - last_center[0]) ** 2 +
                            (center(first_box)[1] - last_center[1]) ** 2) ** 0.5
                    if dist <= max_dist and (best_gap is None or gap < best_gap):
                        best_candidate, best_gap = r2, gap
            if best_candidate is not None:
                union(r1, best_candidate)
                changed = True
        if not changed:
            break
    return {tid: find(tid) for tid in id_frames}


def build_player_groups(per_frame, id_class):
    KEEP_CLASSES = {"player", "goalkeeper"}
    id_frames = collections.defaultdict(list)
    for fidx, dets in per_frame.items():
        for d in dets:
            if id_class.get(d["id"]) in KEEP_CLASSES:
                id_frames[d["id"]].append((fidx, d["box"]))
    for tid in id_frames:
        id_frames[tid].sort(key=lambda x: x[0])

    tid_to_group = group_player_tracks(id_frames)

    group_total_frames = collections.Counter()
    group_first_frame = {}
    for tid, frames in id_frames.items():
        g = tid_to_group[tid]
        group_total_frames[g] += len(frames)
        first = frames[0][0]
        if g not in group_first_frame or first < group_first_frame[g]:
            group_first_frame[g] = first

    MIN_FRAMES = 10
    valid_groups = {g for g, c in group_total_frames.items() if c >= MIN_FRAMES}
    id_frames = {tid: f for tid, f in id_frames.items() if tid_to_group[tid] in valid_groups}
    tid_to_group = {tid: g for tid, g in tid_to_group.items() if g in valid_groups}

    group_members = collections.defaultdict(set)
    for tid, g in tid_to_group.items():
        group_members[g].add(tid)

    ordered_groups = sorted(valid_groups, key=lambda g: group_first_frame[g])

    rep_to_members = {}
    labels = []
    player_count = 0
    for g in ordered_groups:
        members = group_members[g]
        rep = min(members)
        rep_to_members[rep] = members
        cls = id_class.get(rep, "player")
        if cls == "goalkeeper":
            label = "Goalkeeper"
        else:
            player_count += 1
            label = f"Player {player_count}"
        labels.append((rep, label, members))

    return rep_to_members, labels


def draw_and_encode_progress(video_path, per_frame, draw_fn, out_avi, out_mp4, job_id):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    writer = cv2.VideoWriter(out_avi, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        dets = per_frame.get(frame_idx, [])
        draw_fn(frame, dets, frame_idx)
        writer.write(frame)
        frame_idx += 1
        pct = int(min(frame_idx / total_frames, 1.0) * 85)
        PROGRESS[job_id].update(percent=pct, phase="Drawing tracked video...", frame=frame_idx, total_frames=total_frames)

    cap.release()
    writer.release()

    PROGRESS[job_id].update(percent=92, phase="Encoding final video...")
    subprocess.run(["ffmpeg", "-y", "-i", out_avi, "-c:v", "libx264", "-pix_fmt", "yuv420p", out_mp4], check=True)


def process_video(uid, in_path, mode):
    try:
        cap_probe = cv2.VideoCapture(in_path)
        total_frames = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        cap_probe.release()

        PROGRESS[uid] = {"percent": 5, "phase": "Detecting players & ball...", "status": "processing", "next": None, "frame": 0, "total_frames": total_frames}

        per_frame = {}
        id_class = {}
        frame_idx = 0
        first_frame_saved = False
        frame_img_name = f"{uid}_frame.jpg"

        stream = model.track(
            source=in_path, conf=0.4, imgsz=1280,
            tracker=TRACKER_PATH, persist=True, stream=True, verbose=False
        )

        r = None
        for r in stream:
            boxes = r.boxes
            frame_dets = []
            if boxes is not None and boxes.id is not None:
                for box, tid, cls in zip(boxes.xyxy.cpu().numpy(), boxes.id.cpu().numpy(), boxes.cls.cpu().numpy()):
                    tid = int(tid)
                    cls = int(cls)
                    frame_dets.append({"id": tid, "cls": cls, "box": box.tolist()})
                    id_class[tid] = model.names.get(cls, str(cls))
            per_frame[frame_idx] = frame_dets

            if not first_frame_saved and frame_idx == 20:
                cv2.imwrite(os.path.join(FRAMES_DIR, frame_img_name), r.plot())
                first_frame_saved = True

            frame_idx += 1
            pct = 5 + int(min(frame_idx / total_frames, 1.0) * 80)
            PROGRESS[uid].update(percent=pct, frame=frame_idx, total_frames=total_frames)

        if not first_frame_saved and r is not None:
            cv2.imwrite(os.path.join(FRAMES_DIR, frame_img_name), r.plot())

        PROGRESS[uid].update(percent=88, phase="Grouping player tracks...")
        rep_to_members, labels = build_player_groups(per_frame, id_class)

        if mode == "team":
            PROGRESS[uid].update(percent=92, phase="Analyzing jersey colors...")
            tid_to_rep = {tid: rep for rep, label, members in labels for tid in members}
            sample_map = sample_frames_for_groups(per_frame, tid_to_rep)
            group_colors = compute_team_colors(in_path, sample_map)
            group_team, centers = kmeans_2(group_colors)

            member_to_label = {}
            member_to_team = {}
            for rep, label, members in labels:
                team = group_team.get(rep, 0)
                for m in members:
                    member_to_label[m] = label
                    member_to_team[m] = team

            team_a_count = sum(1 for rep, _, _ in labels if group_team.get(rep, 0) == 0)
            team_b_count = sum(1 for rep, _, _ in labels if group_team.get(rep, 0) == 1)
            color_a = bgr_to_css(centers[0])
            color_b = bgr_to_css(centers[1])

            SESSIONS[uid] = {
                "video_path": in_path, "per_frame": per_frame, "id_class": id_class,
                "member_to_label": member_to_label, "member_to_team": member_to_team,
                "frame_img": frame_img_name, "team_a_count": team_a_count, "team_b_count": team_b_count,
                "color_a": color_a, "color_b": color_b,
            }
            PROGRESS[uid] = {"percent": 100, "phase": "Done", "status": "done", "next": f"/team_select/{uid}"}
        else:
            options = [(rep, label) for rep, label, _ in labels]
            SESSIONS[uid] = {
                "video_path": in_path, "per_frame": per_frame, "rep_to_members": rep_to_members,
                "frame_img": frame_img_name, "options": options,
            }
            PROGRESS[uid] = {"percent": 100, "phase": "Done", "status": "done", "next": f"/select/{uid}"}
    except Exception as e:
        PROGRESS[uid] = {"percent": 0, "phase": "Error", "status": "error", "next": None, "error_msg": str(e)}


def render_player_bg(job_id, video_path, per_frame, member_ids):
    try:
        smoother = BoxSmoother(alpha=0.35, hold_frames=6)
        trail = []
        KEY = "target"

        def draw_player(frame, dets, frame_idx):
            seen = False
            for d in dets:
                if d["id"] in member_ids:
                    box = smoother.update(KEY, d["box"], frame_idx)
                    x1, y1, x2, y2 = box
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    trail.append((cx, cy))
                    draw_corner_box(frame, x1, y1, x2, y2, PLAYER_COLOR, thickness=3, corner_len=20)
                    seen = True
            if not seen:
                held = smoother.get_held(KEY, frame_idx)
                if held is not None:
                    x1, y1, x2, y2 = held
                    draw_corner_box(frame, x1, y1, x2, y2, PLAYER_COLOR, thickness=3, corner_len=20)
            draw_tapering_trail(frame, trail, PLAYER_COLOR)

        raw_avi = os.path.join(OUTPUT_DIR, f"{job_id}_raw.avi")
        out_name = f"{job_id}_result.mp4"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        draw_and_encode_progress(video_path, per_frame, draw_player, raw_avi, out_path, job_id)

        RENDER_RESULTS[job_id] = {"result": out_name, "headline": "Player tracked", "show_legend": False}
        PROGRESS[job_id] = {"percent": 100, "phase": "Done", "status": "done", "next": f"/result/{job_id}"}
    except Exception as e:
        PROGRESS[job_id] = {"percent": 0, "phase": "Error", "status": "error", "next": None, "error_msg": str(e)}


def render_team_bg(job_id, video_path, per_frame, id_class, member_to_label, member_to_team, choice):
    try:
        wanted_team = None if choice == "both" else (0 if choice == "a" else 1)
        smoother = BoxSmoother(alpha=0.35, hold_frames=6)
        all_labels = set(member_to_label.values())
        label_to_team = {}
        for tid, label in member_to_label.items():
            label_to_team[label] = member_to_team[tid]
        if wanted_team is not None:
            active_labels = {lbl for lbl in all_labels if label_to_team.get(lbl) == wanted_team}
        else:
            active_labels = all_labels

        def draw_team(frame, dets, frame_idx):
            seen_labels = set()
            for d in dets:
                x1, y1, x2, y2 = d["box"]
                cls = id_class.get(d["id"], "")
                if d["id"] in member_to_label:
                    label = member_to_label[d["id"]]
                    if label not in active_labels:
                        continue
                    box = smoother.update(label, d["box"], frame_idx)
                    x1, y1, x2, y2 = box
                    color = GK_COLOR if cls == "goalkeeper" else PLAYER_COLOR
                    draw_corner_box(frame, x1, y1, x2, y2, color)
                    draw_label_pill(frame, x1, y1, label, color)
                    seen_labels.add(label)
                elif cls == "ball":
                    draw_corner_box(frame, x1, y1, x2, y2, BALL_COLOR, corner_len=8, glow=False)

            for label in active_labels - seen_labels:
                held = smoother.get_held(label, frame_idx)
                if held is not None:
                    x1, y1, x2, y2 = held
                    color = GK_COLOR if label == "Goalkeeper" else PLAYER_COLOR
                    draw_corner_box(frame, x1, y1, x2, y2, color)
                    draw_label_pill(frame, x1, y1, label, color)

        raw_avi = os.path.join(OUTPUT_DIR, f"{job_id}_raw.avi")
        out_name = f"{job_id}_result.mp4"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        draw_and_encode_progress(video_path, per_frame, draw_team, raw_avi, out_path, job_id)

        headline = "Full team tracked" if choice == "both" else f"Team {choice.upper()} tracked"
        RENDER_RESULTS[job_id] = {"result": out_name, "headline": headline, "show_legend": True}
        PROGRESS[job_id] = {"percent": 100, "phase": "Done", "status": "done", "next": f"/result/{job_id}"}
    except Exception as e:
        PROGRESS[job_id] = {"percent": 0, "phase": "Error", "status": "error", "next": None, "error_msg": str(e)}


@app.route("/")
def index():
    return render_template_string(UPLOAD_PAGE)


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files["video"]
    mode = request.form.get("mode", "player")
    uid = uuid.uuid4().hex[:8]
    in_path = os.path.join(UPLOAD_DIR, f"{uid}_{f.filename}")
    f.save(in_path)

    PROGRESS[uid] = {"percent": 0, "phase": "Starting...", "status": "processing", "next": None}
    thread = threading.Thread(target=process_video, args=(uid, in_path, mode), daemon=True)
    thread.start()

    page = progress_page(
        "Processing", "Running detection &amp; tracking",
        "This runs fully on-device &mdash; hang tight while we go frame by frame.",
        f"/status/{uid}"
    )
    return page


@app.route("/status/<uid>")
def status(uid):
    return PROGRESS.get(uid, {"percent": 0, "phase": "Unknown", "status": "error", "next": None, "error_msg": "Not found"})


@app.route("/select/<uid>")
def select_page(uid):
    s = SESSIONS[uid]
    return render_template_string(SELECT_PAGE, uid=uid, frame_img=s["frame_img"], options=s["options"])


@app.route("/team_select/<uid>")
def team_select_page(uid):
    s = SESSIONS[uid]
    return render_template_string(
        TEAM_SELECT_PAGE, uid=uid, frame_img=s["frame_img"],
        team_a_count=s["team_a_count"], team_b_count=s["team_b_count"],
        color_a=s["color_a"], color_b=s["color_b"]
    )


@app.route("/render/<uid>", methods=["POST"])
def render(uid):
    session = SESSIONS[uid]
    rep_id = int(request.form["track_id"])
    video_path = session["video_path"]
    per_frame = session["per_frame"]
    member_ids = session["rep_to_members"].get(rep_id, {rep_id})

    job_id = uuid.uuid4().hex[:8]
    PROGRESS[job_id] = {"percent": 0, "phase": "Starting render...", "status": "processing", "next": None}
    thread = threading.Thread(target=render_player_bg, args=(job_id, video_path, per_frame, member_ids), daemon=True)
    thread.start()

    return progress_page(
        "Rendering", "Rendering your clip",
        "Drawing the tracked player onto the video, frame by frame.",
        f"/status/{job_id}"
    )


@app.route("/render_team/<uid>", methods=["POST"])
def render_team(uid):
    session = SESSIONS[uid]
    choice = request.form.get("team", "both")
    video_path = session["video_path"]
    per_frame = session["per_frame"]
    id_class = session["id_class"]
    member_to_label = session["member_to_label"]
    member_to_team = session["member_to_team"]

    job_id = uuid.uuid4().hex[:8]
    PROGRESS[job_id] = {"percent": 0, "phase": "Starting render...", "status": "processing", "next": None}
    thread = threading.Thread(
        target=render_team_bg,
        args=(job_id, video_path, per_frame, id_class, member_to_label, member_to_team, choice),
        daemon=True
    )
    thread.start()

    return progress_page(
        "Rendering", "Rendering your clip",
        "Drawing the tracked team onto the video, frame by frame.",
        f"/status/{job_id}"
    )


@app.route("/result/<job_id>")
def result_page(job_id):
    r = RENDER_RESULTS[job_id]
    return render_template_string(RESULT_PAGE, result=r["result"], headline=r["headline"], show_legend=r["show_legend"])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)