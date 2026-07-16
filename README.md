# ⚽ Soccer Ball & Player Tracker

This is my project for iD Tech's NVIDIA camp. It's a computer vision app that runs on a Jetson Orin Nano and can detect and track soccer players, the goalkeeper, and the ball in video footage. You upload a clip, it runs everything on-device, and you can either follow one specific player through the video or watch the whole team with color-coded boxes.

![Platform](https://img.shields.io/badge/platform-Jetson%20Orin%20Nano-76B900)
![Python](https://img.shields.io/badge/python-3.10-blue)
![Status](https://img.shields.io/badge/status-camp%20project-yellow)

## Contents

- [Why I made this](#why-i-made-this)
- [What it actually does](#what-it-actually-does)
- [How it works (roughly)](#how-it-works-roughly)
- [Problems I actually ran into](#problems-i-actually-ran-into)
- [Known limitations](#known-limitations)
- [How to run this yourself](#how-to-run-this-yourself)
- [Video](#video)

## Why I made this

I wanted to see if I could build something like what Veo (the sports camera company) does, but on much cheaper hardware. Veo tracks players and the ball automatically during a match, but it needs expensive dedicated cameras and a subscription. I wanted to see how close I could get using a Jetson Orin Nano and free/open detection models.

## What it actually does

- Upload a video of a soccer match
- It detects every player, the goalkeeper, and the ball, frame by frame
- It tracks each player so they keep the same identity as they move around, even if they get blocked from view for a second
- You can either pick one player to follow the whole clip (it draws a box and trail on just them), or view the whole team, or filter to just one side
- It figures out which team is which automatically, based on jersey color, so I didn't have to label anything by hand
- There's a live progress bar while it processes, since detection + tracking takes a bit of time on the Jetson

## How it works (roughly)

I'm not going to pretend I understand every internal detail of these models, but here's the gist of how the pipeline fits together:

1. **Detection** — I used a YOLOv8[^1] model that someone had already fine-tuned specifically to recognize soccer players, goalkeepers, referees, and the ball. It looks at each frame and draws a box around anything it recognizes.
2. **Tracking** — Detection alone doesn't know that "the player in this frame" is the same person as "the player in the last frame." For that I used a tracking algorithm called BoT-SORT[^2], which predicts where things should move next and matches new detections to existing tracks. I turned on its "ReID" (re-identification) feature, which helps it recognize a player again even after they're briefly hidden behind someone else.
3. **Fixing broken tracks** — Even with that, I noticed players would sometimes get a new ID out of nowhere after being blocked for a second. I wrote a bit of extra code that looks for a track that ends and another one that starts right after, in about the same spot, and stitches them back together as the same player.
4. **Team colors** — For splitting players into teams, I sample each player's shirt color from a few frames (ignoring green pixels so the grass doesn't mess up the reading), then group all the colors into two clusters using a simple k-means[^3] approach. Whichever cluster a player's color falls into decides their team.
5. **The web app** — Everything is wrapped in a simple web page (built with Flask) where you upload a video, watch it process, and pick what you want to see.

| Piece | What I used |
|---|---|
| Detection | YOLOv8 (fine-tuned on football footage) |
| Tracking | BoT-SORT + ReID |
| Team clustering | Hand-rolled 2-means on jersey color |
| Backend | Flask + Python threads (for live progress bars) |
| Video I/O | OpenCV + FFmpeg |
| Hardware | NVIDIA Jetson Orin Nano, JetPack 6.0 |

## Problems I actually ran into

This is probably the part I'm most proud of, honestly, since almost nothing worked on the first try:

- **The ball was basically invisible.** At normal camera distance the ball is just a few pixels, so my first attempt at detecting it (using basic shape/color detection) barely worked. I ended up switching to a proper trained model that could pick it up more reliably, though it's still not perfect on wide shots.
- **PyTorch flat out didn't work on the Jetson at first.** The normal way of installing it (`pip install torch`) doesn't actually give you GPU support on a Jetson — I had to track down NVIDIA's own specific version of PyTorch built for my exact JetPack version, which took a while to figure out.
- **Players kept randomly changing ID.** Any time a player got blocked by another player for even a second, the tracker would think it was a brand new person. That's what led me to add the ReID tracking and the track-stitching fix mentioned above.

## Known limitations

Being honest about what doesn't fully work:

- The ball still isn't reliably detected in wide shots — it works a lot better in close-up replay footage.
- I tried getting jersey numbers to work using OCR (reading the digits off the actual jersey) but the numbers are usually too small/blurry on regular match footage to read consistently, so I dropped that feature.
- Team color detection is just based on color clustering, not actual team recognition, so it can get confused if a goalkeeper or ref's kit color is close to a team's color.
- If the camera cuts to a different angle (like a replay), the tracker treats everyone as new people again — it only really works within one continuous shot.

## How to run this yourself

1. You need a Jetson Orin Nano running JetPack 6.0, with Python 3.10.
2. Install NVIDIA's own build of PyTorch (not the regular pip one) — I got mine from NVIDIA's developer download page, matched to my exact JetPack version.
```bash
   pip3 install --no-cache <link to the matching NVIDIA PyTorch wheel for your JetPack version>
```
3. Install Ultralytics plus its smaller dependencies, OpenCV, and Flask:
```bash
   pip3 install ultralytics --no-deps
   pip3 install flask opencv-python
```
4. Download the fine-tuned football YOLOv8 model weights and put them somewhere the app can find them.
5. Set up a BoT-SORT tracker config file with `with_reid: True`.
6. Run it:
```bash
   python3 app.py
```

> **Heads up:** the PyTorch/Jetson version-matching step is the one most likely to trip you up — that took me the longest to figure out too. Make sure the wheel you download matches your exact JetPack/L4T version, not just "JetPack 6."

## Video

Here's a walkthrough / demo of the project in action:

**[ Add your video link here — YouTube, Google Drive, or wherever you upload it ]**

---

[^1]: [YOLOv8](https://github.com/ultralytics/ultralytics) is an object detection model made by Ultralytics — it's what actually finds the players/ball in each frame.
[^2]: [BoT-SORT](https://github.com/NirAharon/BoT-SORT) is a multi-object tracking algorithm that combines motion prediction with appearance matching (ReID) to keep track of the same object across frames.
[^3]: k-means is a basic clustering algorithm — it groups similar data points (in this case, jersey colors) into a set number of clusters (here, 2, for two teams).
