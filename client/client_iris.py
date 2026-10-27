"""
IrisAgent - laptop client

Inward webcam  -> blink / mouth / gaze monitoring (frames sent to the EC2 server)
Outward USB cam-> frozen for Agentic Vision
State machine: MENU, MOBILITY, VISION, MORSE, LOCKED (E-Stop)
"""
import base64
import io
import os
import queue
import threading
import time
from collections import deque

import cv2
import numpy as np
import requests
SERVER_BASE = os.getenv("IRIS_SERVER", "http://13.206.222.255:5000") #IP Changes everytime make sure to replace
INWARD_CAM_INDEX = int(os.getenv("INWARD_CAM", "0"))
OUTWARD_CAM_INDEX = int(os.getenv("OUTWARD_CAM", "1"))
SEND_WIDTH = 640
JPEG_QUALITY = 75

ESTOP_S = 10.0                
QUICK_BLINK_MAX_S = 4         
LONG_BLINK_MIN_S = 4.0        
TRIPLE_BLINK_WINDOW_S = 8     
DOUBLE_BLINK_WINDOW_S = 6     
OPEN_GRACE_S = 0.2            
MOUTH_ENTRY_S = 5.0           
MORSE_CONFIRM_S = 3.0          
MORSE_SWITCH_S = 3.0          

MOBILITY_ARM_S = 2.0          
MOBILITY_STABLE_FRAMES = 3    
SWAP_LEFT_RIGHT = False       
GAZE_TO_COMMAND = {"left": "left", "straight": "forward", "right": "right", "closed": "stop"}
VISION_CONFIRM = "VISION_CONFIRM"
SPEAK_EACH_LETTER = False     

# Requirement 2: Intention Dwell-Time Selection Config
VISION_DWELL_TARGET_S = 5.0

ALPHABET = {
    "A": "00", "B": "0110", "C": "110", "D": "101", "E": "0", "F": "0100", "G": "0010",
    "H": "100", "I": "000", "J": "1100", "K": "1001", "L": "011", "M": "0000", "N": "001",
    "O": "10", "P": "0001", "Q": "1110", "R": "010", "S": "11", "T": "01", "U": "111",
    "V": "1000", "W": "0111", "X": "1011", "Y": "0011", "Z": "1010", "SPACE": "1", "DELETE": "1111",
}
PHRASES = {"Help!": "Physical key 1", "Need Water": "0", "Need Food": "1", "I'm not feeling well!": "Physical key 2"}

MENU, MOBILITY, VISION, MORSE, LOCKED = "MENU", "MOBILITY", "VISION", "MORSE", "LOCKED"

class Net:
    def __init__(self):
        self.session = requests.Session()

    def predict(self, frame, mode):
        _, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        r = self.session.post(SERVER_BASE + "/predict", files={"image": enc.tobytes()},
                              data={"mode": mode}, timeout=3.0)
        r.raise_for_status()
        return r.json()

    def mobility(self, cmd):
        try:
            requests.post(SERVER_BASE + "/mobility", json={"command": cmd}, timeout=3.0)
        except Exception as e:
            print("[mobility] failed:", e)

    def estop(self, mode):
        r = requests.post(SERVER_BASE + "/estop", json={"mode": mode}, timeout=15.0)
        return r.json()

    def vision(self, jpeg, region):
        r = requests.post(SERVER_BASE + "/vision", files={"image": jpeg}, data={"region": region}, timeout=45.0)
        return r.json()

    def speak(self, text):
        r = requests.post(SERVER_BASE + "/speak", json={"text": text}, timeout=15.0)
        return r.json().get("audio_b64")

    def notify(self, message):
        r = self.session.post(SERVER_BASE + "/notify", json={"message": message}, timeout=10.0)
        return r.json().get("audio_b64")

class Audio:
    def __init__(self):
        self.q = queue.Queue()
        self.ok = True
        try:
            import pygame
            pygame.mixer.init()
            self.pg = pygame
        except Exception as e:
            print("[audio] disabled:", e)
            self.ok = False
            return
        threading.Thread(target=self._run, daemon=True).start()

    def play_b64(self, b64):
        if b64 and self.ok:
            self.q.put(base64.b64decode(b64))

    def interrupt(self):
        if not self.ok:
            return
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass
        self.pg.mixer.music.stop()

    def _run(self):
        while True:
            data = self.q.get()
            try:
                self.pg.mixer.music.load(io.BytesIO(data), "mp3")
                self.pg.mixer.music.play()
                while self.pg.mixer.music.get_busy():
                    time.sleep(0.05)
            except Exception as e:
                print("[audio] play error:", e)

class MobilityLink(threading.Thread):
    def __init__(self, net):
        super().__init__(daemon=True)
        self.net, self.q = net, queue.Queue()
        self.start()

    def send(self, cmd):
        self.q.put(cmd)

    def clear(self):
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass

    def run(self):
        while True:
            self.net.mobility(self.q.get())

class OutwardCam:
    def __init__(self, index):
        self.frame = None
        self.lock = threading.Lock()
        self.ok = False
        threading.Thread(target=self._init_cam, args=(index,), daemon=True).start()

    def _init_cam(self, index):
        print(f"[outward cam] Connecting to camera index {index}...")
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        start_time = time.time()
        while not cap.isOpened() and time.time() - start_time < 3.0:
            time.sleep(0.1)

        if cap.isOpened():
            self.ok = True
            print(f"[outward cam] Successfully connected to camera index {index}")
            while self.ok:
                ret, f = cap.read()
                if ret:
                    with self.lock:
                        self.frame = f
                else:
                    time.sleep(0.05)
        else:
            print(f"[outward cam] WARNING: Camera index {index} failed to open. Vision mode will be disabled.")

    def latest(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

class Inference(threading.Thread):
    def __init__(self, net):
        super().__init__(daemon=True)
        self.net = net
        self.lock = threading.Lock()
        self.frame, self.mode = None, "menu" 
        self.result = None
        self.status = "connecting..."
        self.start()

    def set_frame(self, frame, mode):
        with self.lock:
            self.frame, self.mode = frame, mode

    def take_result(self):
        with self.lock:
            r, self.result = self.result, None
            return r

    def run(self):
        while True:
            with self.lock:
                frame, mode = self.frame, self.mode
            if frame is None:
                time.sleep(0.02)
                continue
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (SEND_WIDTH, int(h * SEND_WIDTH / w)))
            try:
                data = self.net.predict(small, mode)
                with self.lock:
                    self.result = (mode, data)
                    self.status = "online"
            except Exception as e:
                self.status = "SERVER OFFLINE (%s)" % type(e).__name__
                time.sleep(0.3)

class BlinkTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.start = None
        self.last_closed = None
        self.estop_fired = False

    def update(self, closed, now):
        events = []
        if closed:
            self.last_closed = now
            if self.start is None:
                self.start, self.estop_fired = now, False
            elif not self.estop_fired and now - self.start >= ESTOP_S:
                self.estop_fired = True
                events.append(("estop", now - self.start))
        elif self.start is not None and now - self.last_closed >= OPEN_GRACE_S:
            if not self.estop_fired:
                events.append(("blink", self.last_closed - self.start))
            self.reset()
        return events

class MorseEngine:
    def __init__(self, speak):
        self.speak = speak
        self.enter()

    def enter(self):
        self.submode = "phrases"
        self.binary, self.decoded, self.word = "", "", ""
        self.m2 = self.m4 = None

    def on_blink(self, duration):
        self.binary += "1" if duration >= LONG_BLINK_MIN_S else "0"

    def trigger_instant_phrase(self, phrase):
        # For quick physical/instant phrases, we append and speak immediately
        self.decoded += phrase + " "
        self.speak(phrase)

    def _confirm(self):
        table = PHRASES if self.submode == "phrases" else ALPHABET
        for name, code in table.items():
            if self.binary == code:
                if self.submode == "phrases":
                    self.decoded += name + " "
                    # Optional: Uncomment below if you want phrases to speak instantly, 
                    # otherwise leave commented to let user review the full text string first.
                    # self.speak(name)
                elif name == "SPACE":
                    self.decoded += " "
                    # Speaking the built word/sentence string upon a Space command
                    if self.decoded.strip():
                        self.speak(self.decoded.strip())
                elif name == "DELETE":
                    self.decoded = self.decoded[:-1]
                    self.word = self.word[:-1]
                else:
                    # Accumulate letters silently into decoded text string
                    self.decoded += name
                    self.word += name
                    # NOTE: Removed SPEAK_EACH_LETTER call here so it won't interrupt with single letters.
                break
        self.binary = ""

    def on_mouth(self, mouth_open, now):
        if not mouth_open:
            self.m2 = self.m4 = None
            return None
        if self.m2 is None:
            self.m2 = now
        if self.m4 is None:
            self.m4 = now

        # Hold mouth for 5.0 seconds to Exit and speak the WHOLE accumulated text string
        if now - self.m4 >= 5.0:
            self.m2 = self.m4 = None
            self.binary = ""
            if self.submode == "phrases":
                self.submode = "az"
                return None
            
            # SPEAK ENTIRE ACCUMULATED TEXT BEFORE EXITING TO MENU
            if self.decoded.strip():
                self.speak(self.decoded.strip())
                
            self.word = ""
            return "exit"

        # Hold mouth for 2.0 seconds to Confirm current letter/symbol code
        elif now - self.m2 >= 2.0:
            self._confirm()
            self.m2 = None
            
        return None

class Controller:
    def __init__(self, net, audio, mobility, outward):
        self.net, self.audio, self.mob, self.outward = net, audio, mobility, outward
        self.lock = threading.RLock()
        self.blink = BlinkTracker()
        self.morse = MorseEngine(self.speak)
        self.state = MENU
        self.quick = deque()
        self.mouth_t = None
        self.last_open_gaze = None
        self.vision_token = 0
        self.vision_view = None        
        self.vision_text = ""
        self.banner = "MENU"
        self.cmd, self.cand, self.cand_n, self.mob_enter = "stop", None, 0, 0.0
        self.total_blinks = 0

        # Requirement 2: Dwell tracking fields
        self.dwell_zone = None
        self.dwell_start_time = 0.0

    @property
    def server_mode(self):
        # Requirement 1 & 2: When in VISION state, server tracks gaze via 'vision' or fallback mode
        return {MENU: "menu", MOBILITY: "mobility", VISION: "vision"}.get(self.state, "detr")

    def _norm(self, g):
        if SWAP_LEFT_RIGHT and g in ("left", "right"):
            return "right" if g == "left" else "left"
        return g

    def _set(self, state, banner=None):
        self.state = state
        self.blink.reset()
        self.quick.clear()
        self.mouth_t = None
        self.banner = banner or state
        if state != VISION:
            self.dwell_zone = None
            self.dwell_start_time = 0.0

    def speak(self, text):
        def job():
            try:
                self.audio.play_b64(self.net.speak(text))
            except Exception as e:
                print("[speak] failed:", e)
        threading.Thread(target=job, daemon=True).start()

    def _send_cmd(self, cmd):
        if cmd != self.cmd:
            self.cmd = cmd
            self.mob.send(cmd)

    def _quick_count(self, now, window):
        while self.quick and now - self.quick[0] > window:
            self.quick.popleft()
        return len(self.quick)

    def _mouth_hold(self, mouth, now, seconds):
        if not mouth:
            self.mouth_t = None
            return False
        if self.mouth_t is None:
            self.mouth_t = now
        elif now - self.mouth_t >= seconds:
            self.mouth_t = None
            return True
        return False

    def step(self, obs, now):
        with self.lock:
            closed = bool(obs.get("eyes_closed"))
            mouth = bool(obs.get("mouth_open"))
            gaze = obs.get("gaze")
            if gaze and gaze != "closed":
                self.last_open_gaze = self._norm(gaze)

            events = self.blink.update(closed, now)
            
            for kind, dur in events:
                if kind == "blink":
                    self.total_blinks += 1

            if self.state != LOCKED and any(k == "estop" for k, _ in events):
                self.trigger_estop()
                return

            if self.state == LOCKED:
                if self._mouth_hold(mouth, now, MOUTH_ENTRY_S):
                    self._set(MENU, "MENU (reset after E-Stop)")
            elif self.state == MENU:
                self._menu(events, mouth, now)
            elif self.state == MOBILITY:
                self._mobility(events, gaze, now)
            elif self.state == MORSE:
                self._morse(events, mouth, now)
            elif self.state == VISION:
                self._vision_dwell_step(gaze, now)

    def handle_key(self, key, now):
        with self.lock:
            if key == ord('1') or key == 1:
                if self.state == MENU:
                    self.morse.enter()
                    self._set(MORSE, "MORSE MODE (Button 1: Press 2 for Help, 0=Water, 1=Food)")
                elif self.state == MORSE:
                    self.morse.trigger_instant_phrase("I'm not feeling well!")
                elif self.state == VISION_CONFIRM:
                    # Requirement 3: Handle Yes confirmation and dispatch context notification
                    self.banner = "NOTIFYING CAREGIVER..."
                    def notify_job():
                        audio = self.net.notify("Caregiver alert: User requested help regarding: " + self.vision_text)
                        self.audio.play_b64(audio)
                        time.sleep(2)
                        with self.lock:
                            self._set(MENU, "MENU")
                    threading.Thread(target=notify_job, daemon=True).start()
            elif key == ord('2') or key == 2:
                if self.state == MENU:
                    self.start_vision()
                elif self.state == MORSE:
                    self.morse.trigger_instant_phrase("Help!")
                elif self.state == VISION_CONFIRM:
                    self.speak("Okay, dismissing.")
                    self._set(MENU, "MENU")
            elif key == ord('3') or key == 3:
                if self.state == MENU:
                    self.cmd, self.cand, self.cand_n = "stop", None, 0
                    self.mob_enter = now
                    self._set(MOBILITY, "MOBILITY MODE")
                    
    def _menu(self, events, mouth, now):
        pass

    def _mobility(self, events, gaze, now):
        for kind, dur in events:
            if kind == "blink" and dur < QUICK_BLINK_MAX_S:
                self.quick.append(now)
                if self._quick_count(now, DOUBLE_BLINK_WINDOW_S) >= 2:
                    self.mob.clear()
                    self._send_cmd("stop")
                    self._set(MENU, "MENU")
                    return
        if now - self.mob_enter < MOBILITY_ARM_S:
            self._send_cmd("stop")
            return
        g = self._norm(gaze) if gaze else None
        target = GAZE_TO_COMMAND.get(g, "stop")
        immediate = (g == "closed")
        if target == self.cand:
            self.cand_n += 1
        else:
            self.cand, self.cand_n = target, 1
        if immediate or self.cand_n >= MOBILITY_STABLE_FRAMES:
            self._send_cmd(target)

    def _morse(self, events, mouth, now):
        for kind, dur in events:
            if kind == "blink":
                self.morse.on_blink(dur)
        if self.morse.on_mouth(mouth, now) == "exit":
            self._set(MENU, "MENU")
        else:
            self.banner = "MORSE MODE (%s)" % ("phrases" if self.morse.submode == "phrases" else "A-Z")

    # Requirement 2: Dwell-Time Selection Engine
    def _vision_dwell_step(self, gaze, now):
        g = self._norm(gaze) if gaze else "straight"
        if g not in ("left", "straight", "right"):
            g = "straight"

        if g == self.dwell_zone:
            elapsed = now - self.dwell_start_time
            remaining = max(0.0, VISION_DWELL_TARGET_S - elapsed)
            self.banner = f"VISION: Targeting {g.upper()} [{remaining:.1f}s]"
            if elapsed >= VISION_DWELL_TARGET_S:
                target_region = g
                self.dwell_zone = None
                self.trigger_vision_capture(target_region)
        else:
            self.dwell_zone = g
            self.dwell_start_time = now
            self.banner = f"VISION: Targeting {g.upper()} [{VISION_DWELL_TARGET_S:.1f}s]"

    def start_vision(self):
        frame = self.outward.latest()
        if frame is None:
            self.speak("The outward camera is not available.")
            self._set(MENU, "MENU (no outward camera)")
            return
        # Enter vision state, resetting dwell parameters for zone selection
        self.dwell_zone = None
        self.dwell_start_time = time.time()
        self._set(VISION, "VISION: Look Left, Center, or Right (3s Dwell)")

    def trigger_vision_capture(self, region):
        frame = self.outward.latest()
        if frame is None:
            self.speak("The outward camera is not available.")
            self._set(MENU, "MENU (no outward camera)")
            return
        
        self.last_open_gaze = region
        h, w = frame.shape[:2]
        idx = {"left": 0, "straight": 1, "right": 2}.get(region, 1)
        third = w // 3
        x1, x2 = idx * third, (w if idx == 2 else (idx + 1) * third)
        roi = frame[:, x1:x2].copy()
        snap = frame.copy()
        cv2.rectangle(snap, (x1, 0), (x2 - 1, h - 1), (0, 255, 0), 3)
        self.vision_token += 1
        token = self.vision_token
        
        self.speak(f"Analyzing {region} region")
        self._set(VISION, f"VISION: analyzing ({region})")
        threading.Thread(target=self._vision_job, args=(token, roi, snap, region), daemon=True).start()

    def _vision_job(self, token, roi, snap, region):
        try:
            _, enc = cv2.imencode(".jpg", roi, [cv2.IMWRITE_JPEG_QUALITY, 90])
            res = self.net.vision(enc.tobytes(), region)
        except Exception as e:
            res = {"speech": "Sorry, the analysis failed.", "audio_b64": None, "error": str(e)}
        with self.lock:
            if token != self.vision_token or self.state not in (VISION, VISION_CONFIRM):
                return                                 
            self.audio.play_b64(res.get("audio_b64"))
            self.vision_text = res.get("speech", "")
            cv2.putText(snap, self.vision_text[:90], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            self.vision_view = (snap, time.time() + 15)
            self._set(VISION_CONFIRM, "NOTIFY CAREGIVER? [1 = YES | 2 = NO]")

    def trigger_estop(self):
        prev = self.state
        self.vision_token += 1                         
        self.audio.interrupt()
        self.mob.clear()
        self.cmd = "stop"
        self.morse.enter()
        self._set(LOCKED, "E-STOP LOCKED - open mouth 2s to reset")

        def job():
            try:
                self.audio.play_b64(self.net.estop(prev).get("audio_b64"))
            except Exception as e:
                print("[estop] server call failed:", e)
        threading.Thread(target=job, daemon=True).start()

def draw(frame, ctrl, dets, img_wh, status, fps_gaze):
    h, w = frame.shape[:2]
    iw, ih = img_wh
    for d in dets:
        x, y, bw, bh = d["box"]
        x, y, bw, bh = int(x * w / iw), int(y * h / ih), int(bw * w / iw), int(bh * h / ih)
        col = (0, 0, 255) if "closed" in d["label"] else (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), col, 2)
        cv2.putText(frame, "%s %.2f" % (d["label"], d["confidence"]), (x, max(y - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

    colour = {MENU: (255, 255, 0), MOBILITY: (0, 255, 0), VISION: (255, 0, 255),
          VISION_CONFIRM: (0, 165, 255), MORSE: (0, 200, 255), LOCKED: (0, 0, 255)}[ctrl.state]
    cv2.rectangle(frame, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.putText(frame, ctrl.banner, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
    cv2.putText(frame, f"Blinks: {ctrl.total_blinks}", (w - 150, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, "server: " + status, (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    lines = []
    if ctrl.state == MENU:
        lines = ["Press 1 = Morse Mode / Emergency Mode", "Press 2 = Agentic Vision",
                 "Press 3 = Mobility", "Eyes closed 8s = E-STOP"]
    elif ctrl.state == MOBILITY:
        lines = ["Gaze: %s" % (ctrl.last_open_gaze or "-"), "Command: %s" % ctrl.cmd.upper(),
                 "Double blink = back to menu"]
    elif ctrl.state == MORSE:
        m = ctrl.morse
        table = PHRASES if m.submode == "phrases" else ALPHABET
        y = 60
        x = 10
        for k, v in table.items():
            cv2.putText(frame, "%s:%s" % (k, v), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            if m.submode == "phrases":
                y += 24
            else:
                x += 90
                if x > w - 90:
                    x, y = 10, y + 22
        lines = ["Binary: " + m.binary, "Text: " + m.decoded,
                 "Short blink=0  Long blink(3s)=1", "Mouth 2s=confirm  Mouth 4s=switch/exit"]
    elif ctrl.state == LOCKED:
        lines = ["E-STOP ACTIVE - caregiver alerted", "Open mouth for 2s to reset"]
    elif ctrl.state == VISION:
        lines = ["Look Left / Center / Right to select region (3s Dwell)"]
    elif ctrl.state == VISION_CONFIRM:
        lines = ["Press 1 = Notify Caregiver (YES)", "Press 2 = Dismiss (NO)"]
    for i, t in enumerate(lines):
        cv2.putText(frame, t, (10, h - 30 - (len(lines) - 1 - i) * 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)

def main():
    import queue
    import io
    net = Net()
    audio = Audio()
    mob = MobilityLink(net)
    print("Initializing outward camera...")
    outward = OutwardCam(OUTWARD_CAM_INDEX)
    print("Outward camera initialized.")
    
    ctrl = Controller(net, audio, mob, outward)
    inf = Inference(net)

    print("Initializing inward camera...")
    cap = cv2.VideoCapture(INWARD_CAM_INDEX)
    print("Inward camera initialized.")
    print("IrisAgent client running. Press 'q' to quit the app.")
    dets, img_wh, vision_open = [], (SEND_WIDTH, 480), False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        inf.set_frame(frame.copy(), ctrl.server_mode)

        res = inf.take_result()
        if res:
            mode, data = res
            if mode == ctrl.server_mode and "error" not in data:     
                dets = data.get("detections", [])
                img_wh = (data.get("img_w", SEND_WIDTH), data.get("img_h", 480))
                ctrl.step(data, time.time())

        draw(frame, ctrl, dets, img_wh, inf.status, None)
        cv2.imshow("IrisAgent", frame)

        # Requirement 1: Dual-Window Parallel Display implementation
        outward_frame = outward.latest()
        if ctrl.state in (VISION, VISION_CONFIRM) and outward_frame is not None:
            # Draw vertical dividing lines for LEFT, STRAIGHT, RIGHT zones
            oh, ow = outward_frame.shape[:2]
            v_zone = outward_frame.copy()
            third_w = ow // 3
            cv2.line(v_zone, (third_w, 0), (third_w, oh), (255, 0, 0), 2)
            cv2.line(v_zone, (third_w * 2, 0), (third_w * 2, oh), (255, 0, 0), 2)
            cv2.putText(v_zone, "LEFT ZONE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(v_zone, "STRAIGHT ZONE", (third_w + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(v_zone, "RIGHT ZONE", (third_w * 2 + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("Environment View", v_zone)
            vision_open = True
        else:
            if vision_open:
                try:
                    cv2.destroyWindow("Environment View")
                except Exception:
                    pass
                vision_open = False

        # Show captured image snapshot context if available during confirm phase
        v = ctrl.vision_view
        if v and time.time() < v[1] and ctrl.state == VISION_CONFIRM:
            cv2.imshow("Agentic Vision Result", v[0])
        elif not vision_open:
            try:
                cv2.destroyWindow("Agentic Vision Result")
            except Exception:
                pass

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in (ord('1'), ord('2'), ord('3')):
            ctrl.handle_key(key, time.time())

    mob.clear()
    net.mobility("stop")
    cap.release()
    cv2.destroyAllWindows() 

if __name__ == "__main__":
    main()
