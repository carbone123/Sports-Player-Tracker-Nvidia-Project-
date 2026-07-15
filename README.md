### Soccer Ball & Player Tracker

On-device computer vision app that detects, tracks, and visualizes soccer players and the ball, running fully on an NVIDIA Jetson Orin Nano.

What it does

Upload a clip of soccer footage through the web interface, and the system will:


Detect every player, the goalkeeper, and the ball in each frame
Track each detected player across the whole clip, keeping a consistent identity even through brief occlusion
Automatically split players into two teams based on jersey color (no manual labeling)
Let you choose to follow one specific player through the clip, or view the full team (or just one side) with color-coded tracking boxes
Show live progress bars for both the detection/tracking stage and the final video rendering stage


Everything — detection, tracking, and rendering — runs locally on the Jetson. No footage is sent anywhere.

How it works

Detection: A YOLOv8 model, fine-tuned specifically on football imagery, scans each frame and identifies four classes: player, goalkeeper, ball, and referee.

Tracking: BoT-SORT (with appearance-based re-identification, "ReID") takes those per-frame detections and links them into consistent tracks over time, using both motion prediction and visual appearance to keep an ID stable even when a player is briefly blocked from view.

Fragment merging: Even with a strong tracker, brief tracking hiccups can occasionally split one player into two IDs. A custom post-processing step looks for track fragments that start soon after another ends, in roughly the same position, and merges them back into a single player identity.

Team clustering: For each detected player, the system samples their jersey color across several frames (masking out the grass so pitch color doesn't bias the reading) and runs a simple 2-means clustering to automatically split the players into two teams.

Box smoothing: Tracked box positions are exponentially smoothed frame to frame so they glide rather than jitter, and briefly "hold" at the last known position for a few frames if detection momentarily drops out, rather than flickering on and off.

Interface: A Flask web app handles video upload, runs the pipeline in a background thread with live progress reporting, and lets the user choose what to visualize before rendering the final annotated video.

Known limitations


Ball detection is unreliable at typical broadcast distance. The ball is often only a few pixels wide in wide-shot footage, which is near the practical limit of what a lightweight detector can resolve. It works much better in close-up or replay footage.
Jersey-number reading (OCR) was attempted and set aside (might be included later). At broadcast resolution, digits on a jersey are frequently too small or blurry to read reliably — this is a genuine hardware/footage-resolution limitation, not a bug.
Team assignment is color-based, not true team recognition. It can be confused by unusual goalkeeper or referee kit colors, since it clusters purely on visual jersey color.
Tracking assumes one continuous camera shot. It does not re-identify players across a hard camera cut (e.g., cutting from a wide shot to a replay) — a new track begins after every cut.


Reproducibility

To run this project on another Jetson Orin Nano (or similar Jetson device):


Base requirements

Jetson Orin Nano running JetPack 6.0 (L4T R36.3.0)
Python 3.10
jetson-inference installed (for camera/detection tooling used during development)



Install the Jetson-specific PyTorch and torchvision builds
Generic pip install torch does not have Jetson GPU support — you need NVIDIA's board-specific wheel:


bash   pip3 install --no-cache https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.4.0a0+07cecf4168.nv24.05.14710581-cp310-cp310-linux_aarch64.whl

Then install the matching torchvision build with --no-deps so pip doesn't overwrite the correct torch version:

bash   pip3 install --no-deps torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl

Verify GPU support is working:

bash   python3 -c "import torch; print(torch.cuda.is_available())"

This should print True.


Install Ultralytics and remaining dependencies


bash   pip3 install ultralytics --no-deps
   pip3 install pyyaml requests tqdm matplotlib pandas seaborn psutil py-cpuinfo nvidia-ml-py polars ultralytics-thop
   pip3 install flask opencv-python
   sudo apt install ffmpeg -y


Get the detection model
Download the football-fine-tuned YOLOv8 weights (yolov8m-football.pt) and place them at ~/yolov8football/yolov8m-football.pt.
Set up the tracker config
Create ~/custom_botsort.yaml:


yaml   tracker_type: botsort
   track_high_thresh: 0.5
   track_low_thresh: 0.1
   new_track_thresh: 0.6
   track_buffer: 60
   match_thresh: 0.8
   fuse_score: True
   gmc_method: none
   proximity_thresh: 0.5
   appearance_thresh: 0.25
   with_reid: True
   model: auto


Run the app
Place app.py in a ~/soccer_webapp/ folder, then:


bash   cd ~/soccer_webapp
   python3 app.py

Open the forwarded port (default 8000) in a browser to use the interface.

Project structure

soccer_webapp/
├── app.py                  # Flask app: routes, tracking pipeline, rendering
├── uploads/                # User-uploaded source clips
└── static/
    ├── outputs/             # Rendered result videos
    └── frames/              # Saved preview frames for the selection screens

yolov8football/
└── yolov8m-football.pt     # Detection model weights

custom_botsort.yaml          # Tracker configuration
