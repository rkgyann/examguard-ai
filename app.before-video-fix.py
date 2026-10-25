import csv
import math
from collections import deque
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import streamlit as st
from ultralytics import YOLO


# Event: (minimum duration in video seconds, base risk points)
RULES = {
    "Left seat": (5, 30),
    "Repeated side turn": (1, 25),
    "Phone detected": (2, 40),
    "Gathering": (4, 35),
    "Unusual movement": (3, 25),
}

STATES = ["NORMAL", "OBSERVE", "SUSPICIOUS", "ALERT"]

# OpenCV uses BGR colours.
COLORS = [
    (40, 180, 40),    # Green
    (0, 220, 240),    # Yellow
    (0, 140, 255),    # Orange
    (30, 30, 230),    # Red
]


def distance(a, b):
    """Calculate the distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def check_event(student, name, active, now, confidence, frequency=None):
    """Remember an activity and decide whether to generate an alert."""

    event = student["events"].setdefault(name, {
        "start": None,
        "times": deque(),
        "saved": False,
        "last_alert": -1000,
    })

    # Remember activity episodes from the last 30 seconds.
    while event["times"] and now - event["times"][0] > 30:
        event["times"].popleft()

    if not active:
        event["start"] = None
        event["saved"] = False
        return 0, None

    if event["start"] is None:
        event["start"] = now
        event["times"].append(now)

    duration = now - event["start"]
    threshold, base = RULES[name]

    count = (
        frequency
        if frequency is not None
        else len(event["times"])
    )

    # Basic rule-based risk score.
    score = round(min(
        100,
        base
        + min(30, 20 * duration / threshold)
        + min(15, 5 * count)
        + 10 * confidence
    ))

    if score >= 65:
        risk = "High"
    elif score >= 45:
        risk = "Medium"
    else:
        risk = "Low"

    if duration >= threshold:
        state = 3
    elif duration >= threshold / 2:
        state = 2
    else:
        state = 1

    # Wait until the activity lasts long enough.
    if state < 3 or event["saved"]:
        return state, None

    # Save only one alert for a continuous activity.
    event["saved"] = True

    # Prevent frequent duplicate alerts.
    if now - event["last_alert"] < 15:
        return state, None

    event["last_alert"] = now

    reason = (
        f"{name} for {duration:.1f}s; "
        f"{count} episode(s) in 30s"
    )

    alert = {
        "Time (seconds)": round(now, 2),
        "Seat": student["seat"],
        "Event": name,
        "Confidence": round(confidence, 2),
        "Risk score": score,
        "Risk": risk,
        "Reason": reason + ". Invigilator review required.",
    }

    return state, alert


def analyze(video_path, output, screen, message):
    # This model detects people and estimates body/face landmarks.
    people_model = YOLO("yolo11n-pose.pt")

    # This model detects objects, including mobile phones.
    phone_model = YOLO("yolo11n.pt")

    memory = {}
    alerts = []

    cap = cv2.VideoCapture(str(video_path))

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)

        if not cap.isOpened() or not 0 < fps < 1000:
            raise ValueError(
                "Cannot read video. Use a constant-frame-rate MP4."
            )

        frame_number = 0

        while True:
            ok, frame = cap.read()

            if not ok:
                break

            # Use video time, not the computer's processing time.
            now = frame_number / fps

            height, width = frame.shape[:2]
            diagonal = math.hypot(width, height)

            # Detect people and maintain their tracking IDs.
            result = people_model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=0.35,
                device="cpu",
                verbose=False
            )[0]

            # Class 67 is "cell phone" in this model.
            phones = phone_model.predict(
                frame,
                classes=[67],
                conf=0.4,
                device="cpu",
                verbose=False
            )[0]

            people = []

            if result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().tolist()
                ids = result.boxes.id.int().cpu().tolist()
                confidences = result.boxes.conf.cpu().tolist()
                landmarks = result.keypoints.data.cpu().tolist()

                for box, identity, confidence, points in zip(
                    boxes, ids, confidences, landmarks
                ):
                    x1, y1, x2, y2 = box

                    center = (
                        (x1 + x2) / 2,
                        (y1 + y2) / 2
                    )

                    # Create memory when a person first appears.
                    if identity not in memory:
                        memory[identity] = {
                            "seat": (
                                f"S{identity}"
                                if now <= 3
                                else f"Unassigned-{identity}"
                            ),
                            "home": center if now <= 3 else None,
                            "radius": max(30, (x2 - x1) * 0.75),
                            "events": {},
                            "positions": deque(),
                            "turns": deque(),
                            "side_latched": False,
                            "neutral": [],
                        }

                    people.append({
                        "id": identity,
                        "box": box,
                        "center": center,
                        "confidence": confidence,
                        "points": points,
                        "phone": 0,
                    })

            # A missing detection must not count as leaving a seat.
            visible = {person["id"] for person in people}

            for identity, student in memory.items():
                if identity not in visible:
                    student["positions"].clear()

                    for event in student["events"].values():
                        event["start"] = None
                        event["saved"] = False

            # Associate detected phones with nearby people.
            for box, confidence in zip(
                phones.boxes.xyxy.cpu().tolist(),
                phones.boxes.conf.cpu().tolist()
            ):
                x1, y1, x2, y2 = box

                point = (
                    (x1 + x2) / 2,
                    (y1 + y2) / 2
                )

                candidates = [
                    person for person in people
                    if (
                        person["box"][0] - 20
                        <= point[0]
                        <= person["box"][2] + 20
                    )
                    and (
                        person["box"][1] - 20
                        <= point[1]
                        <= person["box"][3] + 20
                    )
                ]

                if candidates:
                    nearest = min(
                        candidates,
                        key=lambda person: distance(
                            person["center"], point
                        )
                    )

                    nearest["phone"] = max(
                        nearest["phone"], confidence
                    )

                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (220, 0, 220),
                    2
                )

            pending = []

            for person in people:
                student = memory[person["id"]]
                center = person["center"]

                # First three pose landmarks: nose and two eyes.
                nose, eye1, eye2 = person["points"][:3]

                face_confidence = min(
                    nose[2], eye1[2], eye2[2]
                )

                eye_width = distance(
                    eye1[:2], eye2[:2]
                )

                side = False

                # Basic sideways-head estimate.
                if face_confidence >= 0.5 and eye_width >= 8:
                    offset = (
                        nose[0] - (eye1[0] + eye2[0]) / 2
                    ) / eye_width

                    # Learn the forward-facing position first.
                    if now <= 3:
                        student["neutral"].append(offset)

                    elif len(student["neutral"]) >= 3:
                        baseline = (
                            sum(student["neutral"])
                            / len(student["neutral"])
                        )

                        side = abs(offset - baseline) > 0.35

                        # Count the start of a new side turn.
                        if side and not student["side_latched"]:
                            student["turns"].append(now)

                        student["side_latched"] = side

                while (
                    student["turns"]
                    and now - student["turns"][0] > 30
                ):
                    student["turns"].popleft()

                # Remember recent positions to estimate movement.
                positions = student["positions"]
                positions.append((now, center))

                while (
                    len(positions) > 1
                    and now - positions[0][0] > 2
                ):
                    positions.popleft()

                span = now - positions[0][0]

                path = sum(
                    distance(a[1], b[1])
                    for a, b in zip(
                        positions, list(positions)[1:]
                    )
                )

                speed = (
                    path / diagonal / max(span, 0.001)
                )

                # Count people near this person, including themselves.
                neighbors = sum(
                    distance(center, other["center"])
                    / diagonal < 0.12
                    for other in people
                )

                # The five basic activity rules.
                conditions = {
                    "Left seat": (
                        student["home"] is not None
                        and distance(center, student["home"])
                        > student["radius"]
                    ),
                    "Repeated side turn": (
                        side and len(student["turns"]) >= 3
                    ),
                    "Phone detected": person["phone"] > 0,
                    "Gathering": neighbors >= 3,
                    "Unusual movement": (
                        span >= 1 and speed > 0.035
                    ),
                }

                state = 0

                # Start monitoring after the calibration period.
                if now > 3:
                    for name, active in conditions.items():
                        if name == "Phone detected":
                            confidence = person["phone"]
                        elif name == "Repeated side turn":
                            confidence = face_confidence
                        else:
                            confidence = person["confidence"]

                        frequency = (
                            len(student["turns"])
                            if name == "Repeated side turn"
                            else None
                        )

                        level, alert = check_event(
                            student,
                            name,
                            active,
                            now,
                            confidence,
                            frequency
                        )

                        state = max(state, level)

                        if alert:
                            pending.append(alert)

                # Show the original seat area.
                if student["home"] is not None:
                    home = tuple(
                        int(value)
                        for value in student["home"]
                    )

                    cv2.circle(
                        frame,
                        home,
                        int(student["radius"]),
                        (150, 150, 150),
                        1
                    )

                # Highlight the person and their current state.
                x1, y1, x2, y2 = map(
                    int, person["box"]
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    COLORS[state],
                    2
                )

                label = (
                    f"{student['seat']} | {STATES[state]}"
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    COLORS[state],
                    2
                )

            # Store each alert with an evidence image.
            for alert in pending:
                evidence = (
                    output / f"event-{len(alerts) + 1:03d}.jpg"
                )

                if not cv2.imwrite(str(evidence), frame):
                    raise OSError(
                        "Could not save evidence image"
                    )

                alert["Evidence"] = str(evidence)
                alerts.append(alert)

                with (output / "events.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as file:
                    writer = csv.DictWriter(
                        file,
                        fieldnames=alerts[0].keys()
                    )

                    writer.writeheader()
                    writer.writerows(alerts)

            # Update the dashboard every five frames.
            if frame_number % 5 == 0:
                screen.image(frame, channels="BGR")

                message.write(
                    f"Video: {now:.1f}s | "
                    f"Alerts: {len(alerts)}"
                )

            frame_number += 1

        if frame_number == 0:
            raise ValueError(
                "Video contains no readable frames"
            )

        return alerts

    finally:
        cap.release()


# ---------------- DASHBOARD ----------------

st.set_page_config(
    page_title="ExamGuard AI",
    layout="wide"
)

st.title("ExamGuard AI — P-12 Mini Project")

st.write(
    "Upload a fixed-camera video. Everyone must be seated "
    "and face forward for the first 3 seconds."
)

st.caption(
    "Basic prototype: side turns and phone associations "
    "are estimates. Every alert needs human review."
)

uploaded = st.file_uploader(
    "Choose an examination video",
    type=["mp4", "mov", "avi"]
)

if uploaded and st.button("Start analysis"):
    output = (
        Path("results")
        / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    )

    output.mkdir(parents=True)

    temporary = NamedTemporaryFile(
        delete=False,
        suffix=Path(uploaded.name).suffix
    )

    temporary.write(uploaded.getvalue())
    temporary.close()

    screen = st.empty()
    message = st.empty()

    try:
        with st.spinner(
            "Analyzing video; initial run downloads the models..."
        ):
            st.session_state["alerts"] = analyze(
                temporary.name,
                output,
                screen,
                message
            )

        st.success(
            f"Finished. Results saved in {output.resolve()}"
        )

    except Exception as error:
        st.session_state.pop("alerts", None)
        st.error(str(error))

    finally:
        Path(temporary.name).unlink(missing_ok=True)

if "alerts" in st.session_state:
    alerts = st.session_state["alerts"]

    st.subheader("Alert history and risk levels")

    if alerts:
        st.dataframe(alerts, hide_index=True)

        selected = st.selectbox(
            "Choose evidence",
            range(len(alerts)),
            format_func=lambda i: (
                f"{alerts[i]['Seat']} — "
                f"{alerts[i]['Event']}"
            )
        )

        st.write(alerts[selected]["Reason"])
        st.image(alerts[selected]["Evidence"])

    else:
        st.info(
            "No events crossed the alert thresholds."
        )