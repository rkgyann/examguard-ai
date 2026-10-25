# ExamGuard AI

### P-12: Intelligent Examination Hall Monitoring and Anomaly Detection Agent

ExamGuard AI is a college mini project that analyzes examination-hall videos and highlights observable activities that may need an invigilator's attention. It combines pretrained computer-vision models, student tracking, a model-based agent, rule-based reasoning, and a Streamlit dashboard.

**The system flags activities for human review. It does not prove cheating or identify a student's intent.**

## Project objective

Develop a basic AI agent that detects and tracks people in examination videos, remembers recent observations, applies activity rules over time, and generates explainable alerts with saved evidence.

## Features

- Upload and analyze examination videos.
- Detect people and track them across consecutive frames.
- Monitor movement outside an estimated seat area.
- Estimate repeated side turns and sustained sideways posture.
- Detect visible mobile phones and associate them approximately with nearby people.
- Flag reaching toward a neighbouring person.
- Detect gathering and sustained unusual movement using basic rules.
- Maintain `NORMAL`, `OBSERVE`, `SUSPICIOUS`, and `ALERT` states.
- Calculate a rule-based risk score from activity type, duration, repetition, and detection confidence.
- Display explainable alerts, highlighted locations, and supporting evidence.
- Handle camera cuts by starting a new scene and resetting seat assignments.
- Estimate visible row and column positions, with optional manual corrections.
- Play an alert beep and announce selected positions when running locally on macOS.
- Save input/output videos, event records, evidence images, and individual activity clips.

## How it works

```text
Upload examination video
          ↓
Detect people and estimate body landmarks
          ↓
Track people and establish scene-local seat anchors
          ↓
Remember recent observations
          ↓
Apply duration and repetition rules
          ↓
Calculate state and risk score
          ↓
Display alerts and save evidence, records, and clips
```

A short detection gap pauses activity evidence instead of immediately erasing it. Longer gaps end the episode. Missing detections do not automatically count as leaving a seat.

## Technologies

| Technology | Purpose |
|---|---|
| Python | Application and agent logic |
| Streamlit | Dashboard and video upload |
| OpenCV | Video reading, drawing, and evidence images |
| Ultralytics YOLO11 Pose | Person detection and body landmarks |
| Ultralytics YOLO11 | Mobile-phone detection |
| ByteTrack | Temporary person tracking IDs |
| NumPy | Numerical operations and beep generation |
| FFmpeg | Browser-compatible video and clip export |
| macOS `afplay` and `say` | Local speaker playback and position announcements |

The application uses pretrained `yolo11n-pose.pt` and `yolo11n.pt` models. Training a new model is not required for this prototype.

## Project files

```text
examguard-ai/
├── app.py
├── requirements.txt
├── packages.txt
├── README.md
├── .gitignore
└── results/                 # Generated locally; exclude from Git
```

Downloaded model files may also be present in the project folder.

## Run locally

Install Python and open the project folder in VS Code. Python 3.12 is a suitable deployment choice; the local implementation has also been exercised with Python 3.14 on macOS.

### macOS

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

### Windows

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the Local URL printed in the terminal, usually `http://localhost:8501`. Keep the terminal running. Press **Ctrl+C** to stop the application.

Internet access is needed for package installation and for model downloads when the weights are not already available.

## Python dependencies

`requirements.txt`:

```text
streamlit
ultralytics
opencv-python
numpy
lap
imageio-ffmpeg
```

## Using the dashboard

1. Select **Short demo clips** or **Longer CCTV footage**.
2. Upload an MP4, MOV, or AVI examination video.
3. If running on a Mac, use **Test beep** to check the selected audio output and volume.
4. Choose the amount of clip context before and after an activity.
5. Click **Start analysis**.
6. Review the alert history and select an event to inspect its evidence and clip.
7. Check the displayed scene, seat, row, column, and position source.
8. On a local Mac, use **Announce this position**, or enable automatic announcement when selecting clips.

Use short demonstration videos first. CPU processing can be slower than normal video playback.

## Seat positions and camera cuts

Automatic position labels follow this convention:

- **Row 1:** the visible row nearest the camera.
- **Column 1:** the leftmost visible seat in that row.

These are estimates from visible seat anchors, not verified university seat numbers. Perspective, hidden students, cropped views, and staff can affect the estimates.

Review the saved seat-map image. To enter known positions, use the dashboard's manual correction box:

```text
1:S04=2,3
2:S08=1,4
```

This assigns scene 1 seat S04 to row 2 column 3, and scene 2 seat S08 to row 1 column 4. Rerun the video after entering corrections.

Tracking IDs and seat labels are temporary and restart at camera cuts. The application does not perform face recognition or establish identity across different camera views.

## Saved results

Each completed analysis creates a separate run folder beside `app.py`:

```text
results/<run-folder>/
├── input.mp4                  # Original extension is retained
├── annotated_video.mp4
├── events.csv
├── events.json
├── summary.json
├── settings.json
├── last-frame.jpg
├── alert_beep.wav
├── seat_map.csv
├── seat_map.json
├── seat_maps/
│   └── scene-01.jpg
└── events/
    └── event-001/
        ├── evidence.jpg
        ├── original.jpg
        ├── details.json
        ├── activity.mp4
        ├── original.mp4
        ├── announcement.txt
        └── position.aiff       # When macOS speech generation succeeds
```

Event records include video time, scene, seat, activity type, duration, confidence, risk score, explanation, row/column position, and evidence/clip paths.

Each clip includes the supported activity episode plus the selected context, limited by camera cuts and video boundaries. Different participants or rules can produce overlapping clips.

The annotated output uses sampled frames at approximately eight frames per second while preserving the source timeline. Exported annotated videos and event clips are silent; the saved input retains its original audio.

If no alerts occur, the application still saves the input, annotated output, empty event records, settings, and summary. Available records are retained if a later export fails.

## Deploy with GitHub and Streamlit Community Cloud

GitHub stores the source code. Streamlit Community Cloud runs the Python application.

1. Upload `app.py`, `requirements.txt`, `packages.txt`, and this README to a GitHub repository.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and connect GitHub.
3. Create an app and select the repository, `main` branch, and `app.py` entrypoint.
4. Choose Python 3.12 under Advanced settings.
5. Deploy and share the generated `.streamlit.app` URL.
6. Configure app sharing separately from repository privacy if using a private repository.

`packages.txt`:

```text
libgl1
libglib2.0-0
ffmpeg
```

### Cloud-specific limitations

- Turn off the Mac beep and automatic announcements when running on cloud hosting. The current `afplay` and `say` functions require macOS and do not play through a remote visitor's speakers. Browser-based audio is a separate adaptation.
- Generated files are saved on the hosting server, not automatically on the user's computer. Cloud files are not guaranteed to persist. Download required evidence and clips.
- Local Mac beep and speech features do not require a paid API or cloud speech service.
- Start with short clips because hosted CPU and memory resources are limited.

## Validation

The revised monitoring pipeline was tested on two supplied ten-second demonstration videos. Each produced three activity alerts. Annotated and original per-event clips were opened and their durations checked.

The sound/position version includes eighteen tests covering event timing, missing observations, duplicate alerts, tracking reassociation, dashboard loading, seat grouping, manual corrections, beep-file validity, speaker-command invocation, announcement wording, and voice-failure handling.

The tool environment blocked macOS speech generation, so audible speech must be checked on the user's Mac. Invalid empty speech files are rejected, and the application retains announcement text with a direct speech fallback.

These checks are limited demonstrations, not a measured general accuracy score. Test normal footage as well as staged activity and record false alerts and missed events.

## Limitations

- Sideways posture can include ordinary leaning or other innocent behaviour.
- Reaching does not establish paper transfer or misconduct.
- A detected phone does not establish ownership or usage.
- Small or hidden faces, hands, and phones may be missed.
- Camera movement and changing perspective can affect geometry-based rules.
- Automatic row/column estimates require review or a manually supplied seating plan.
- Detection confidence is not the probability that cheating occurred.
- An invigilator makes the final decision after reviewing the evidence.

## Academic concepts demonstrated

- Computer vision and object detection
- Multi-object tracking
- Model-based intelligent agents
- Internal state and observation history
- Rule-based reasoning
- Temporal reasoning
- Risk scoring
- Explainable alerts and human review

## References

- [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/)
- [Ultralytics tracking](https://docs.ultralytics.com/modes/track/)
- [Ultralytics pose estimation](https://docs.ultralytics.com/tasks/pose/)
- [Streamlit documentation](https://docs.streamlit.io/)
- [Deploy on Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [Streamlit cloud file persistence](https://docs.streamlit.io/develop/concepts/configuration/serving-static-files)
