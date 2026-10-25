"""ExamGuard AI: basic observable-activity rules, not a cheating classifier."""
import csv
import json
import math
import shutil
import subprocess
import struct
import wave
from collections import deque
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

# Durations count supported observations, not missing frames.
DEMO_RULES = {
    'Left seat area': (2.0, 30), 'Repeated side turn': (.6, 30),
    'Sustained sideways posture': (.65, 25), 'Reaching toward neighbour': (.45, 35),
    'Phone detected': (.8, 40), 'Gathering': (2.0, 35), 'Unusual movement': (2.0, 25),
}
STATES = ['NORMAL', 'OBSERVE', 'SUSPICIOUS', 'ALERT']
COLORS = [(40, 180, 40), (0, 220, 240), (0, 140, 255), (30, 30, 230)]
FIELDS = ['Event ID', 'Time (seconds)', 'Activity start (seconds)', 'Activity end (seconds)',
          'Scene', 'Seat', 'Event', 'Supported seconds', 'Confidence', 'Risk score',
          'Risk', 'Reason', 'Evidence', 'Raw evidence', 'Clip', 'Original clip',
          'Clip start (seconds)', 'Clip end (seconds)', 'Row', 'Column', 'Position source',
          'Announcement', 'Announcement audio']


def make_beep(path):
    """Generate a short, moderate-volume WAV without installing an audio library."""
    rate = 44100
    t = np.arange(int(rate * .22)) / rate
    envelope = np.minimum(1., t / .012) * np.minimum(1., (.22 - t) / .025)
    samples = (0.22 * 32767 * np.sin(2 * np.pi * 880 * t) * envelope).astype('<i2')
    with wave.open(str(path), 'wb') as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(rate)
        file.writeframes(samples.tobytes())


def play_on_mac(path):
    """Plays through the Mac running Streamlit; it is not a remote-browser alert."""
    player = shutil.which('afplay')
    if not player:
        raise RuntimeError('System speaker playback requires macOS afplay.')
    subprocess.run([player, str(path)], check=True, timeout=30, capture_output=True)


def speak_on_mac(text):
    voice = shutil.which('say')
    if not voice:
        raise RuntimeError('Position speech requires the macOS say command.')
    result = subprocess.run([voice, text], capture_output=True, text=True, timeout=30)
    if result.returncode or 'sandbox_extension' in result.stderr:
        raise RuntimeError('macOS blocked the voice service. The position remains available as text.')


def valid_voice_file(path):
    """Some macOS voice failures return success but write an empty AIFF header."""
    frames, sound = 0, False
    try:
        with Path(path).open('rb') as file:
            header = file.read(12)
            if header[:4] != b'FORM':
                return False
            while True:
                chunk = file.read(8)
                if len(chunk) < 8:
                    break
                name, size = struct.unpack('>4sI', chunk)
                data = file.read(size)
                if name == b'COMM' and len(data) >= 8:
                    frames = struct.unpack('>I', data[2:6])[0]
                if name == b'SSND' and len(data) > 8:
                    sound = True
                if size % 2:
                    file.read(1)
        return frames > 0 and sound
    except (OSError, struct.error):
        return False


def read_position_overrides(text):
    """Optional exact seat map: one line such as 1:S04=2,3."""
    result = {}
    used = set()
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            label, position = line.replace(' ', '').split('=')
            scene, seat = label.split(':')
            row, column = map(int, position.split(','))
            if min(int(scene), row, column) < 1 or not seat or (int(scene), row, column) in used:
                raise ValueError()
            used.add((int(scene), row, column))
            result[f'{int(scene)}:{seat}'] = (row, column)
        except ValueError:
            raise ValueError('Use one position per line, for example 1:S04=2,3. Numbers must be positive.')
    return result


def locate_seats(memory, scene, overrides, excluded=()):
    """Estimate VISIBLE rows, nearest camera first, then columns left to right."""
    groups = []
    for seat, student in sorted(memory.items(), key=lambda item: -item[1]['home'][1]):
        if f'{scene}:{seat}' in excluded:
            continue
        x, y = student['home']
        width = student['width']
        candidates = [g for g in groups if abs(y - g['y']) <= max(12, .28 * min(width, g['width']))]
        if candidates:
            group = min(candidates, key=lambda g: abs(y - g['y']))
            group['seats'].append((seat, x, y))
            group['y'] = float(np.mean([p[2] for p in group['seats']]))
        else:
            groups.append({'y': y, 'width': width, 'seats': [(seat, x, y)]})
    positions = {}
    for row, group in enumerate(sorted(groups, key=lambda g: -g['y']), 1):
        for column, (seat, x, y) in enumerate(sorted(group['seats'], key=lambda p: p[1]), 1):
            position = overrides.get(f'{scene}:{seat}')
            positions[seat] = {'Scene': scene, 'Seat': seat,
                               'Row': position[0] if position else row,
                               'Column': position[1] if position else column,
                               'Position source': 'Manual seat map' if position else 'Estimated visible seats',
                               'Anchor x': round(x, 1), 'Anchor y': round(y, 1)}
    return positions


def position_message(alert):
    estimated = 'estimated ' if alert['Position source'] != 'Manual seat map' else ''
    return (f"Scene {alert['Scene']}. Flagged activity at {estimated}row {alert['Row']}, "
            f"column {alert['Column']}. {alert['Event']}. Invigilator review required.")


def save_announcements(output, alerts):
    """Save text on every platform and speech files on macOS, without playing them."""
    voice = shutil.which('say')
    for alert in alerts:
        folder = output / 'events' / alert['Event ID']
        alert['Announcement'] = position_message(alert)
        (folder / 'announcement.txt').write_text(alert['Announcement'], encoding='utf-8')
        alert['Announcement audio'] = ''
        if voice:
            audio = folder / 'position.aiff'
            try:
                result = subprocess.run([voice, '-o', str(audio), alert['Announcement']],
                                        capture_output=True, text=True, timeout=30)
                valid = result.returncode == 0 and valid_voice_file(audio)
            except (OSError, subprocess.SubprocessError):
                valid = False
            if valid:
                alert['Announcement audio'] = str(audio.relative_to(output))
            else:
                audio.unlink(missing_ok=True)
                with (output / 'voice-errors.txt').open('a', encoding='utf-8') as log:
                    log.write(f"{alert['Event ID']}: voice file unavailable; use direct announcement button.\n")
        (folder / 'details.json').write_text(json.dumps(alert, indent=2), encoding='utf-8')
    save_records(output, alerts)


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def check_event(student, name, active, now, confidence, rules, detail='', frequency=None):
    """False/unknown gaps pause support; a gap >0.45 s ends the episode."""
    event = student['events'].setdefault(name, {
        'last_true': None, 'last_observation': None, 'previous': False,
        'support': 0., 'times': deque(), 'saved': False, 'last_alert': -1000,
        'confidence_sum': 0., 'count': 0,
        'record': None, 'start': None,
    })
    last = event['last_observation']
    dt = 0 if last is None else max(0, now - last)
    if event['last_true'] is not None and now - event['last_true'] > .45:
        event.update(last_true=None, support=0., saved=False, confidence_sum=0., count=0,
                     record=None, start=None)
    event['last_observation'] = now
    while event['times'] and now - event['times'][0] > 30:
        event['times'].popleft()
    if active is not True:
        event['previous'] = False
        # Short gaps retain evidence but add no positive time or new alerts.
        return (1 if event['last_true'] is not None else 0), None
    if event['last_true'] is None:
        event['times'].append(now)
        event['start'] = now
    elif event['previous'] and dt <= .45:
        event['support'] += dt
    event['last_true'], event['previous'] = now, True
    event['confidence_sum'] += confidence
    event['count'] += 1
    duration = event['support']
    threshold, base = rules[name]
    count = len(event['times']) if frequency is None else frequency
    confidence = event['confidence_sum'] / event['count']
    score = round(min(100, base + min(30, 20 * duration / threshold)
                      + min(15, 5 * count) + 10 * confidence))
    risk = 'High' if score >= 65 else 'Medium' if score >= 45 else 'Low'
    state = 3 if duration + 1e-6 >= threshold else 2 if duration >= threshold / 2 else 1
    # Continue updating an alerted episode until it actually ends.
    if event['record'] is not None:
        event['record'].update({'Activity end (seconds)': round(now, 3),
                               'Supported seconds': round(duration, 3),
                               'Confidence': round(confidence, 3),
                               'Risk score': score, 'Risk': risk})
    if state < 3 or event['saved']:
        return state, None
    if now - event['last_alert'] < 8:
        return state, None
    event['saved'], event['last_alert'] = True, now
    record = {
        'Time (seconds)': round(now, 2), 'Seat': student['seat'], 'Event': name,
        'Activity start (seconds)': round(event['start'], 3),
        'Activity end (seconds)': round(now, 3),
        'Supported seconds': round(duration, 2), 'Confidence': round(confidence, 3),
        'Risk score': score, 'Risk': risk,
        'Reason': f'{name}. {detail} Review required.',
    }
    event['record'] = record
    return state, record


def ffmpeg_command(arguments):
    """Use installed FFmpeg, or the binary provided by imageio-ffmpeg."""
    executable = shutil.which('ffmpeg')
    if executable is None:
        from imageio_ffmpeg import get_ffmpeg_exe
        executable = get_ffmpeg_exe()
    result = subprocess.run([executable, '-hide_banner', '-loglevel', 'error', '-y',
                             *map(str, arguments)], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError('Video export failed: ' + result.stderr[-1200:])


def encode_video(source, target, start=None, duration=None):
    arguments = [] if start is None else ['-ss', start]
    arguments += ['-i', source]
    if duration is not None:
        arguments += ['-t', duration]
    # H.264 MP4 works in browsers. Exports are silent; the saved input keeps its audio.
    arguments += ['-an', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                  '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
                  '-pix_fmt', 'yuv420p', '-movflags', '+faststart', target]
    ffmpeg_command(arguments)


def save_records(output, alerts):
    with (output / 'events.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(alerts)
    (output / 'events.json').write_text(json.dumps(alerts, indent=2), encoding='utf-8')


def export_clips(source, annotated, output, alerts, cuts, duration, before, after, sample_fps=8):
    """Save the whole supported episode plus context, within the same camera shot."""
    boundaries = [0., *cuts, duration]
    for alert in alerts:
        scene_index = alert['Scene'] - 1
        start = max(boundaries[scene_index], alert['Activity start (seconds)'] - before)
        end = min(boundaries[scene_index + 1], alert['Activity end (seconds)'] + after)
        # Align scene edges to annotated frames so a clip never shows the previous shot.
        start = max(start, math.ceil(boundaries[scene_index] * sample_fps - 1e-6) / sample_fps)
        end = min(end, math.floor(boundaries[scene_index + 1] * sample_fps + 1e-6) / sample_fps)
        folder = output / 'events' / alert['Event ID']
        annotated_clip, original_clip = folder / 'activity.mp4', folder / 'original.mp4'
        encode_video(annotated, annotated_clip, start, end - start)
        encode_video(source, original_clip, start, end - start)
        alert.update({'Clip': str(annotated_clip.relative_to(output)),
                      'Original clip': str(original_clip.relative_to(output)),
                      'Clip start (seconds)': round(start, 3), 'Clip end (seconds)': round(end, 3)})
        # Update each event's separate record with final duration and clip paths.
        (folder / 'details.json').write_text(json.dumps(alert, indent=2), encoding='utf-8')
    save_records(output, alerts)


def make_person(box, identity, confidence, points):
    nose, left, right = points[0], points[5], points[6]
    shoulders_visible = min(left[2], right[2]) >= .4
    anchor = ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2) if shoulders_visible else (
        (box[0] + box[2]) / 2, box[1] + .3 * (box[3] - box[1]))
    width = max(20, distance(left[:2], right[:2])) if shoulders_visible else max(20, .65 * (box[2] - box[0]))
    return dict(box=box, id=identity, confidence=confidence, points=points,
                anchor=anchor, width=width, reliable=shoulders_visible, phone=0.)


def assign_seats(people, memory, aliases, now, scene_start):
    """Reconnect a briefly lost track only when a nearby seat is unoccupied."""
    used = set()
    # Claim existing IDs before considering replacement IDs.
    for person in people:
        seat = aliases.get(person['id'])
        if seat in memory and seat not in used:
            person['seat'] = seat
            used.add(seat)
    for person in people:
        if 'seat' in person:
            continue
        candidates = [name for name, student in memory.items() if name not in used
                      and 0 <= now - student['seen'] <= 1
                      and distance(person['anchor'], student['last']) < .65 * max(person['width'], student['width'])]
        if candidates:
            seat = min(candidates, key=lambda name: distance(person['anchor'], memory[name]['last']))
        elif now - scene_start <= .65:
            seat = f'S{len(memory) + 1:02d}'
            memory[seat] = dict(seat=seat, home=person['anchor'], width=person['width'],
                                last=person['anchor'], seen=now, events={}, positions=deque(),
                                baseline=[], turns=deque(), side_since=None, side_latched=False)
        else:
            person['seat'] = None
            continue
        aliases[person['id']] = seat
        person['seat'] = seat
        used.add(seat)
    for person in people:
        if person['seat'] is not None:
            student = memory[person['seat']]
            student['last'], student['seen'] = person['anchor'], now


def posture(person, student, now, scene_start):
    """Visible nose/shoulder geometry: sideways POSTURE, not gaze/intent."""
    points = person['points']
    nose, eye1, eye2, left, right = points[0], points[1], points[2], points[5], points[6]
    if not person['reliable'] or nose[2] < .5:
        return None, 0.
    lean = (nose[0] - person['anchor'][0]) / person['width']
    if now - scene_start <= .65:
        student['baseline'].append(lean)
    neutral = float(np.median(student['baseline'])) if student['baseline'] else lean
    eye_span = distance(eye1[:2], eye2[:2])
    yaw = 0.
    if min(eye1[2], eye2[2]) >= .4 and eye_span >= 5:
        yaw = abs(nose[0] - (eye1[0] + eye2[0]) / 2) / eye_span
    side = bool(yaw > .4 or abs(lean) > .58 or (abs(lean - neutral) > .28 and abs(lean) > .3))
    return side, min(nose[2], left[2], right[2])


def reaching(person, people):
    """A visibly extended wrist moving horizontally toward another upper body."""
    if not person['reliable']:
        return None, 0.
    points = person['points']
    for shoulder_id, elbow_id, wrist_id in [(5, 7, 9), (6, 8, 10)]:
        shoulder, elbow, wrist = points[shoulder_id], points[elbow_id], points[wrist_id]
        confidence = min(shoulder[2], elbow[2], wrist[2])
        if confidence < .45:
            continue
        if abs(wrist[0] - shoulder[0]) < .9 * person['width']:
            continue
        for other in people:
            if other is person or other['seat'] is None or not other['reliable']:
                continue
            a, b = person['anchor'], other['anchor']
            scale = max(person['width'], other['width'])
            toward = (wrist[0] - a[0]) * (b[0] - a[0]) > 0
            closer = abs(wrist[0] - b[0]) < .72 * abs(a[0] - b[0])
            same_row = abs(a[1] - b[1]) < 1.1 * scale
            if toward and closer and same_row and abs(wrist[1] - b[1]) < 1.7 * scale:
                return True, confidence
    return False, person['confidence']


def analyze(video_path, output, screen, message, settings=None, model_dir=None, on_alert=None):
    settings = settings or {'sensitivity': 'Short demo clips', 'exclude': []}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    make_beep(output / 'alert_beep.wav')
    # Preserve the complete submitted video in each run folder.
    original = output / ('input' + Path(video_path).suffix.lower())
    if Path(video_path).resolve() != original.resolve():
        shutil.copy2(video_path, original)
    rules = dict(DEMO_RULES)
    if settings['sensitivity'] == 'Longer CCTV footage':
        rules = {name: (duration * 2, base) for name, (duration, base) in rules.items()}
    model_dir = Path(model_dir) if model_dir else Path(__file__).parent
    pose = YOLO(str(model_dir / 'yolo11n-pose.pt'))
    phones = YOLO(str(model_dir / 'yolo11n.pt'))
    memory, aliases, alerts = {}, {}, []
    seat_maps = {}
    overrides = settings.get('positions', {})
    cuts, diagnostics = [], {'sampled_frames': 0, 'valid_head_observations': 0, 'phone_detections': 0}
    cap = cv2.VideoCapture(str(video_path))
    video_writer = None
    previous, scene, scene_start, frame_number = None, 1, 0., 0
    (output / 'settings.json').write_text(json.dumps({'rules': rules, 'settings': settings}, indent=2))
    save_records(output, [])
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or not 0 < fps < 1000:
            raise ValueError('Cannot read video. Use a constant-frame-rate MP4.')
        stride = max(1, round(fps / 8))  # Infer about eight times per video second.
        with (output / 'events.csv').open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                now = frame_number / fps
                small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (96, 54))
                # A camera cut changes the whole image; never carry seat IDs across it.
                change = 0 if previous is None else float(cv2.absdiff(small, previous).mean())
                previous = small
                if change > 25:
                    if scene not in seat_maps:
                        seat_maps[scene] = locate_seats(memory, scene, overrides, settings['exclude'])
                    scene += 1
                    scene_start = now
                    cuts.append(round(now, 3))
                    memory, aliases = {}, {}
                    if hasattr(pose.predictor, 'trackers'):
                        for tracker in pose.predictor.trackers:
                            tracker.reset()
                frame_number += 1
                if (frame_number - 1) % stride:
                    continue
                diagnostics['sampled_frames'] += 1
                map_folder = output / 'seat_maps'
                map_folder.mkdir(exist_ok=True)
                map_image = map_folder / f'scene-{scene:02d}.jpg'
                if not map_image.exists():
                    cv2.imwrite(str(map_image), frame)
                result = pose.track(frame, persist=True, tracker='bytetrack.yaml', conf=.3,
                                    iou=.4, imgsz=960, device='cpu', verbose=False)[0]
                detection = phones.predict(frame, classes=[67], conf=.35, iou=.4,
                                           imgsz=960, device='cpu', verbose=False)[0]
                people = []
                if result.boxes.id is not None:
                    for box, identity, confidence, points in zip(result.boxes.xyxy.cpu().tolist(),
                            result.boxes.id.int().cpu().tolist(), result.boxes.conf.cpu().tolist(),
                            result.keypoints.data.cpu().tolist()):
                        people.append(make_person(box, identity, confidence, points))
                assign_seats(people, memory, aliases, now, scene_start)
                if now - scene_start > .65 and scene not in seat_maps:
                    seat_maps[scene] = locate_seats(memory, scene, overrides, settings['exclude'])
                # Exclude staff by the scene-specific seat label displayed on screen.
                for person in people:
                    if f'{scene}:{person["seat"]}' in settings['exclude']:
                        person['seat'] = None
                observed_seats = {p['seat'] for p in people if p['seat'] is not None}
                for seat, student in memory.items():
                    if seat not in observed_seats:
                        student['positions'].clear()
                        for name in list(student['events']):
                            check_event(student, name, None, now, 0., rules)
                for box, confidence in zip(detection.boxes.xyxy.cpu().tolist(), detection.boxes.conf.cpu().tolist()):
                    diagnostics['phone_detections'] += 1
                    point = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                    candidates = [p for p in people if p['seat'] is not None
                                  and p['box'][0] <= point[0] <= p['box'][2] and p['box'][1] <= point[1] <= p['box'][3]]
                    if candidates:
                        p = min(candidates, key=lambda p: distance(p['anchor'], point))
                        p['phone'] = max(p['phone'], confidence)
                pending = []
                snapshot = frame.copy()
                for person in people:
                    seat = person['seat']
                    level = 0
                    label = 'Uncalibrated/staff: not assessed'
                    color = (160, 160, 160)
                    if seat is not None:
                        student = memory[seat]
                        side, side_confidence = posture(person, student, now, scene_start)
                        diagnostics['valid_head_observations'] += int(side is not None)
                        if side is True:
                            if student['side_since'] is None:
                                student['side_since'] = now
                            if now - student['side_since'] >= .25 and not student['side_latched']:
                                student['turns'].append(now)
                                student['side_latched'] = True
                        elif side is False:
                            student['side_since'], student['side_latched'] = None, False
                        while student['turns'] and now - student['turns'][0] > 30:
                            student['turns'].popleft()
                        positions = student['positions']
                        if positions and now - positions[-1][0] > .45:
                            positions.clear()
                        positions.append((now, person['anchor']))
                        while len(positions) > 1 and now - positions[0][0] > 2:
                            positions.popleft()
                        span = now - positions[0][0]
                        # Net motion avoids counting back-and-forth detector jitter as walking.
                        speed = distance(person['anchor'], positions[0][1]) / max(span, .01) / person['width']
                        away = distance(person['anchor'], student['home']) > .9 * student['width']
                        reach, reach_confidence = reaching(person, people)
                        close = [p for p in people if p['seat'] is not None and p['reliable']
                                 and distance(person['anchor'], p['anchor']) < 1.4 * max(person['width'], p['width'])]
                        moved = sum(distance(p['anchor'], memory[p['seat']]['home']) > .6 * memory[p['seat']]['width'] for p in close)
                        observations = {
                            'Left seat area': (bool(away) if person['reliable'] else None, person['confidence'], 'Position moved beyond the calibrated seat area.'),
                            'Sustained sideways posture': (side, side_confidence, 'Nose/shoulder geometry changed; gaze and intent are unknown.'),
                            'Repeated side turn': (side and len(student['turns']) >= 3 if side is not None else None, side_confidence, f'{len(student["turns"])} estimated turns in 30 seconds.'),
                            'Reaching toward neighbour': (reach, reach_confidence, 'Arm extends toward another person; object transfer is not verified.'),
                            'Phone detected': (bool(person['phone']), person['phone'], 'Phone-to-person association is approximate.'),
                            'Gathering': (bool(len(close) >= 3 and moved >= 2), person['confidence'], 'At least three nearby people; two moved from their seat anchors.'),
                            'Unusual movement': (bool(span >= .8 and speed > .45), person['confidence'], 'Sustained movement relative to shoulder width.'),
                        }
                        for name, (active, confidence, detail) in observations.items():
                            state, alert = check_event(student, name, active, now, confidence, rules, detail,
                                                      len(student['turns']) if name == 'Repeated side turn' else None)
                            level = max(level, state)
                            if alert:
                                alert['Scene'] = scene
                                position = seat_maps.get(scene, {}).get(seat)
                                if position is None:
                                    position = locate_seats(memory, scene, overrides, settings['exclude'])[seat]
                                for key in ['Row', 'Column', 'Position source']:
                                    alert[key] = position[key]
                                pending.append(alert)
                        label = f'{scene}:{seat} | {STATES[level]}'
                        position = seat_maps.get(scene, {}).get(seat)
                        if position:
                            estimated = '~' if position['Position source'] != 'Manual seat map' else ''
                            label = f"{scene}:{seat} {estimated}R{position['Row']} C{position['Column']} | {STATES[level]}"
                        color = COLORS[level]
                    x1, y1, x2, y2 = map(int, person['box'])
                    cv2.rectangle(snapshot, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(snapshot, label, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
                for alert in pending:
                    event_id = f'event-{len(alerts)+1:03d}'
                    folder = output / 'events' / event_id
                    folder.mkdir(parents=True, exist_ok=True)
                    evidence = folder / 'evidence.jpg'
                    raw_evidence = folder / 'original.jpg'
                    if not cv2.imwrite(str(evidence), snapshot):
                        raise OSError('Could not save evidence image.')
                    if not cv2.imwrite(str(raw_evidence), frame):
                        raise OSError('Could not save original evidence image.')
                    alert.update({'Event ID': event_id, 'Evidence': str(evidence.relative_to(output)),
                                  'Raw evidence': str(raw_evidence.relative_to(output)),
                                  'Clip': '', 'Original clip': '',
                                  'Announcement': '', 'Announcement audio': '',
                                  'Clip start (seconds)': '', 'Clip end (seconds)': ''})
                    alerts.append(alert)
                    (folder / 'details.json').write_text(json.dumps(alert, indent=2))
                    writer.writerow(alert)
                    file.flush()
                    (output / 'events.json').write_text(json.dumps(alerts, indent=2))
                    if on_alert is not None:
                        # Sound failure must not stop monitoring or lose saved evidence.
                        try:
                            on_alert(alert)
                        except Exception as error:
                            with (output / 'audio-errors.txt').open('a', encoding='utf-8') as audio_log:
                                audio_log.write(f"{alert['Event ID']}: {error}\n")
                if video_writer is None:
                    height, width = snapshot.shape[:2]
                    video_writer = cv2.VideoWriter(str(output / '_annotated-working.mp4'),
                                                   cv2.VideoWriter_fourcc(*'mp4v'), fps / stride, (width, height))
                    if not video_writer.isOpened():
                        raise OSError('Could not create the annotated output video.')
                video_writer.write(snapshot)
                cv2.imwrite(str(output / 'last-frame.jpg'), snapshot)
                screen.image(snapshot, channels='BGR')
                message.write(f'Video: {now:.1f}s | Scene: {scene} | Alerts: {len(alerts)} | Cuts: {len(cuts)}')
        if frame_number == 0:
            raise ValueError('No readable video frames.')
        video_writer.release()
        video_writer = None
        if scene not in seat_maps:
            seat_maps[scene] = locate_seats(memory, scene, overrides, settings['exclude'])
        for alert in alerts:
            position = seat_maps[alert['Scene']][alert['Seat']]
            for key in ['Row', 'Column', 'Position source']:
                alert[key] = position[key]
        all_positions = [position for positions in seat_maps.values() for position in positions.values()]
        (output / 'seat_map.json').write_text(json.dumps(all_positions, indent=2))
        with (output / 'seat_map.csv').open('w', newline='', encoding='utf-8') as map_file:
            fields = ['Scene', 'Seat', 'Row', 'Column', 'Position source', 'Anchor x', 'Anchor y']
            writer = csv.DictWriter(map_file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_positions)
        for scene_number, positions in seat_maps.items():
            image_path = output / 'seat_maps' / f'scene-{scene_number:02d}.jpg'
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            for seat, position in positions.items():
                point = (int(position['Anchor x']), int(position['Anchor y']))
                text = f"{seat}: R{position['Row']} C{position['Column']}"
                cv2.circle(image, point, 4, (0, 0, 255), -1)
                cv2.putText(image, text, point, cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 3)
                cv2.putText(image, text, point, cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
            cv2.imwrite(str(image_path), image)
        message.write('Saving annotated video and the clips for each flagged episode...')
        annotated = output / 'annotated_video.mp4'
        encode_video(output / '_annotated-working.mp4', annotated)
        (output / '_annotated-working.mp4').unlink()
        export_clips(original, annotated, output, alerts, cuts, frame_number / fps,
                     settings.get('clip_before', 2.), settings.get('clip_after', 2.), fps / stride)
        save_announcements(output, alerts)
        summary = dict(scenes=scene, cuts=cuts, **diagnostics,
                       alerts=len(alerts), input_video=original.name, output_video=annotated.name,
                       duration_seconds=frame_number / fps,
                       analysis_fps=fps / stride,
                       note='Scene IDs are local to each camera view. No configured alert is not proof of no misconduct.')
        (output / 'summary.json').write_text(json.dumps(summary, indent=2))
        return alerts, summary
    except Exception as error:
        (output / 'error.txt').write_text(str(error), encoding='utf-8')
        # Keep any already detected records and images even if export fails.
        save_records(output, alerts)
        raise
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()


def main():
    st.set_page_config(page_title='ExamGuard AI', layout='wide')
    st.title('ExamGuard AI — P-12 Mini Project')
    st.caption('Flags observable actions for an invigilator. It does not classify or prove cheating.')
    uploaded = st.file_uploader('Choose examination video', type=['mp4', 'mov', 'avi'])
    mode = st.selectbox('Timing preset', ['Short demo clips', 'Longer CCTV footage'])
    excluded = st.text_input('Exclude staff labels, if needed (example: 2:S04, 3:S02)', '')
    beep_enabled = st.checkbox('Beep for each new alert through this Mac', value=True)
    auto_announce = st.checkbox('Automatically announce the selected clip position', value=False)
    if st.button('Test beep'):
        try:
            beep_path = Path(__file__).parent / 'alert_beep.wav'
            make_beep(beep_path)
            play_on_mac(beep_path)
            st.success('Beep sent to the Mac audio output. Check volume and selected speakers.')
        except Exception as error:
            st.warning(str(error))
    st.caption('Position convention: row 1 is nearest the camera; column 1 is leftmost in that visible row. '
               'Automatic labels are estimates of visible seats, not official hall seat numbers. '
               'Sound plays on the Mac running this app, through its selected audio output.')
    with st.expander('Optional manual row/column corrections'):
        positions_text = st.text_area('One correction per line: scene:seat=row,column', '',
                                     placeholder='1:S04=2,3\n2:S08=1,4')
        st.caption('Review the saved seat map, enter known positions, and rerun. For official numbering, '
                   'supply your hall seating plan; occluded or cropped seats cannot be inferred reliably.')
    try:
        positions = read_position_overrides(positions_text)
    except ValueError as error:
        st.error(str(error))
        st.stop()
    before = st.slider('Clip context before activity (seconds)', 0, 5, 2)
    after = st.slider('Clip context after activity (seconds)', 0, 5, 2)
    st.info('Edited clips are split into scenes. Seat labels restart at each cut. '
            'Short demo mode can flag a sustained sideways posture or reaching without requiring three turns.')
    if uploaded and st.button('Start analysis', type='primary'):
        output = Path(__file__).parent / 'results' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        output.mkdir(parents=True, exist_ok=True)
        beep_path = output / 'alert_beep.wav'
        make_beep(beep_path)
        with NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as temporary:
            temporary.write(uploaded.getvalue())
        screen, message = st.empty(), st.empty()
        live_alert = st.empty()
        st.session_state.pop('run', None)
        try:
            settings = {'sensitivity': mode, 'exclude': [s.strip() for s in excluded.split(',') if s.strip()],
                        'clip_before': before, 'clip_after': after, 'positions': positions,
                        'beep_enabled': beep_enabled}
            def notify(alert):
                live_alert.warning(f"New alert: scene {alert['Scene']}, row {alert['Row']}, "
                                   f"column {alert['Column']} ({alert['Position source']}): {alert['Event']}")
                if beep_enabled:
                    play_on_mac(beep_path)
            with st.spinner('Analyzing visible activities...'):
                alerts, summary = analyze(temporary.name, output, screen, message, settings, on_alert=notify)
            st.session_state['run'] = (alerts, summary, str(output))
        except Exception as error:
            st.error(str(error))
        finally:
            Path(temporary.name).unlink(missing_ok=True)
    if 'run' in st.session_state:
        alerts, summary, path = st.session_state['run']
        output = Path(path)
        st.success(f'Finished. {len(alerts)} alerts saved in {output}')
        if (output / 'audio-errors.txt').exists():
            st.warning('Some speaker alerts failed. Check audio-errors.txt in this result folder.')
        if summary['cuts']:
            st.warning(f"Detected {len(summary['cuts'])} camera cuts. Identities and seat anchors reset at each cut; "
                       'seat-leaving and repetition history cannot continue across views.')
        st.subheader('Alert history and supporting evidence')
        st.write('Complete annotated output video')
        st.video(str(output / 'annotated_video.mp4'))
        if alerts:
            st.dataframe(alerts, hide_index=True)
            selected = st.selectbox('Choose evidence', range(len(alerts)),
                                    format_func=lambda i: f"{alerts[i]['Time (seconds)']}s | Scene {alerts[i]['Scene']} | R{alerts[i]['Row']} C{alerts[i]['Column']} | {alerts[i]['Event']}")
            alert = alerts[selected]
            st.info(f"Scene {alert['Scene']} · Row {alert['Row']} · Column {alert['Column']} · "
                    f"Seat {alert['Seat']} · {alert['Position source']}")
            announcement_key = (str(output), alert['Event ID'])
            speak = st.button('Announce this position through Mac speakers')
            if speak or (auto_announce and st.session_state.get('announced') != announcement_key):
                try:
                    if alert['Announcement audio']:
                        play_on_mac(output / alert['Announcement audio'])
                    else:
                        speak_on_mac(alert['Announcement'])
                    st.session_state['announced'] = announcement_key
                except Exception as error:
                    st.warning(str(error))
                    st.session_state['announced'] = announcement_key
            st.write(alerts[selected]['Reason'])
            st.image(str(output / alerts[selected]['Evidence']))
            st.write('Saved clip for this flagged episode')
            st.video(str(output / alerts[selected]['Clip']))
            st.download_button('Download this clip', (output / alerts[selected]['Clip']).read_bytes(),
                               alerts[selected]['Event ID'] + '.mp4', 'video/mp4')
            with st.expander('Review estimated seat positions for this scene'):
                st.image(str(output / 'seat_maps' / f"scene-{alert['Scene']:02d}.jpg"))
        else:
            st.warning('No configured activity accumulated enough visible evidence. '
                       'This does not mean the video contains no suspicious actions. '
                       'Hidden faces, brief shots and lost tracks can prevent detection.')
        st.download_button('Download event CSV', (output / 'events.csv').read_bytes(), 'events.csv', 'text/csv')
        with st.expander('Analysis coverage and limitations'):
            st.json(summary)
            st.write('Sideways posture can include ordinary leaning. Reaching does not establish paper transfer. '
                     'Staff may need exclusion. Camera movement, perspective and occlusion can cause errors. '
                     'The confidence field is detector/keypoint confidence, not confidence of cheating.')


if __name__ == '__main__':
    main()
