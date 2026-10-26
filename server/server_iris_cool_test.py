import uuid

from google import genai
from google.genai import types
import base64
import json
import logging
import os
import threading
import time
from pydantic import BaseModel
import boto3
import cv2
import numpy as np
import onnxruntime as ort
from botocore.config import Config
from flask import Flask, jsonify, request

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("iris")

REGION = os.getenv("AWS_REGION", "ap-south-1")
SNS_TOPIC_ARN = os.getenv("SNS_TOPIC_ARN", "arn:aws:sns:ap-south-1:905418218600:IrisAgent-Emergency")          
SNS_PHONE_NUMBER = os.getenv("SNS_PHONE_NUMBER", "9841942345")    
ESTOP_SMS_COOLDOWN_S = float(os.getenv("ESTOP_SMS_COOLDOWN_S", "30"))
IOT_ENDPOINT = os.getenv("IOT_ENDPOINT", "")            
IOT_TOPIC = os.getenv("IOT_TOPIC", "wheelchair/command")
MOBILITY_WATCHDOG_S = float(os.getenv("MOBILITY_WATCHDOG_S", "2.0"))
POLLY_VOICE = os.getenv("POLLY_VOICE", "Kajal")
POLLY_ENGINE = os.getenv("POLLY_ENGINE", "neural")
YOLO11_ONNX = os.getenv("YOLO11_ONNX", "eye_gaze_blink_mouthopen_best.onnx") 
GAZE_ONNX = os.getenv("GAZE_ONNX", "eye_gaze_model_2.onnx")    
GAZE_EVERY_N = int(os.getenv("GAZE_EVERY_N", "3"))           
CONF_THRESHOLD = 0.15 
DYNAMO_TABLE_NAME = os.getenv("DYNAMO_TABLE_NAME", "IrisAgentLogs")


log.info("Loading YOLO and gaze ONNX models with OpenCV DNN (COOL-compatible)...")
# YOLO inference uses OpenCV DNN. Run this script in the COOL Python environment
# (OpenCV loaded from /opt/cool/python_3.12/site-packages/cv2/python-3.12).
yolo11_net = cv2.dnn.readNetFromONNX(YOLO11_ONNX)
yolo11_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
yolo11_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
log.info("YOLO model loaded through OpenCV DNN.")

# COOL OpenCV 4.12 currently fails to import this gaze ONNX model.
# Keep gaze inference on ONNX Runtime CPU; YOLO remains on COOL OpenCV DNN.
_gaze_opts = ort.SessionOptions()
_gaze_opts.intra_op_num_threads = 2
_gaze_opts.inter_op_num_threads = 1
_gaze_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
_gaze_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
gaze_sess = ort.InferenceSession(
    GAZE_ONNX, sess_options=_gaze_opts, providers=["CPUExecutionProvider"]
)
_gaze_input_name = gaze_sess.get_inputs()[0].name
log.info("Gaze model loaded through ONNX Runtime CPU.")

YOLO11_CLASSES = ["Mouth_open", "closed eyes", "opened eyes"]
GAZE_CLASSES = {0: "straight", 1: "left", 2: "right", 3: "closed"}

_aws_cfg = Config(region_name=REGION, retries={"max_attempts": 2}, read_timeout=30, connect_timeout=5)
_clients = {}
_clients_lock = threading.Lock()

def aws(name, **kw):
    with _clients_lock:
        if name not in _clients:
            _clients[name] = boto3.client(name, config=_aws_cfg, **kw)
        return _clients[name]

app = Flask(__name__)
inference_lock = threading.Lock()
log.info("Inference backends: COOL OpenCV DNN for YOLO; ONNX Runtime CPU for gaze.")

STATE = {
    "cmd": "stop", 
    "last_predict": 0.0, 
    "last_mode": "menu", 
    "last_estop": 0.0, 
    "frame_counter": 0, 
    "last_mouth_open": False,
    "last_valid_eye_box": None,
    "last_vision_text": "",
    "last_vision_time": 0.0  # <--- Added to track rate limits
}

def yolo11_pipeline(frame):
    h, w = frame.shape[:2]
    img = cv2.cvtColor(cv2.resize(frame, (640, 640)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.ascontiguousarray(np.transpose(img, (2, 0, 1))[None, ...]) 
    yolo11_net.setInput(blob)
    output = yolo11_net.forward()
    out = np.squeeze(output)
    if out.ndim == 2 and out.shape[0] < out.shape[1]:
        out = out.T  

    scores = out[:, 4:]                      
    class_ids = np.argmax(scores, axis=1)    
    confidences = scores[np.arange(out.shape[0]), class_ids] 
    mask = confidences >= CONF_THRESHOLD
    if not np.any(mask):
        return [], False, False, None

    filtered_boxes = out[mask, 0:4]
    filtered_confs = confidences[mask]
    filtered_class_ids = class_ids[mask]
    xf, yf = w / 640.0, h / 640.0
    raw_boxes = []
    
    for i in range(filtered_boxes.shape[0]):
        cx, cy, bw, bh = filtered_boxes[i]
        left = int((cx - bw / 2) * xf)
        top = int((cy - bh / 2) * yf)
        width = int(bw * xf)
        height = int(bh * yf)
        raw_boxes.append([max(0, left), max(0, top), width, height])

    conf_list = filtered_confs.tolist()
    class_list = filtered_class_ids.tolist()
    indices = cv2.dnn.NMSBoxes(raw_boxes, conf_list, CONF_THRESHOLD, nms_threshold=0.4)
    
    dets = []
    eyes_closed = False
    mouth_open = False
    best_eye_box = None
    best_eye_conf = 0.0

    if len(indices) > 0:
        flat_indices = indices.flatten() if hasattr(indices, 'flatten') else [i for i in indices]
        for idx in flat_indices:
            x, y, wb, hb = raw_boxes[idx]
            label = YOLO11_CLASSES[class_list[idx]]
            conf = conf_list[idx]
            dets.append({"box": [x, y, wb, hb], "label": label, "confidence": conf})
            if label == "closed eyes":
                eyes_closed = True
            elif label == "Mouth_open":
                mouth_open = True
            if label in ("closed eyes", "opened eyes") and conf > best_eye_conf:
                best_eye_conf = conf
                best_eye_box = [x, y, x + wb, y + hb]
            
    return dets, eyes_closed, mouth_open, best_eye_box

def run_gaze(frame, box):
    x1, y1, x2, y2 = box
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    blob = cv2.dnn.blobFromImage(roi, 1.0 / 127.5, (299, 299), (127.5, 127.5, 127.5), swapRB=True, crop=False)
    gaze_output = gaze_sess.run(None, {_gaze_input_name: blob})[0]
    return GAZE_CLASSES.get(int(np.argmax(gaze_output)), "straight")

def publish_command(cmd, reason="gaze"):
    STATE["cmd"] = cmd
    payload = json.dumps({"command": cmd, "reason": reason, "ts": int(time.time() * 1000)})
    if not IOT_ENDPOINT:
        return {"status": "skipped", "detail": "IOT_ENDPOINT not set"}
    try:
        aws("iot-data", endpoint_url="https://" + IOT_ENDPOINT).publish(topic=IOT_TOPIC, qos=1, payload=payload)
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

def send_sms(message):
    try:
        sns = aws("sns")
        if SNS_TOPIC_ARN:
            sns.publish(TopicArn=SNS_TOPIC_ARN, Message=message, Subject="IrisAgent EMERGENCY")
        elif SNS_PHONE_NUMBER:
            sns.publish(PhoneNumber=SNS_PHONE_NUMBER, Message=message)
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

def polly_b64(text):
    if not text:
        return None
    try:
        r = aws("polly").synthesize_speech(Text=text, OutputFormat="mp3", VoiceId=POLLY_VOICE, Engine=POLLY_ENGINE)
        log_to_dynamodb("SPEECH_SYNTHESIS", f"Text synthesized: {text}")
        return base64.b64encode(r["AudioStream"].read()).decode()
    except Exception as e:
        return None

try:
    gemini_client = genai.Client()
    log.info("Google Gemini GenAI Client initialized successfully.")
except Exception as e:
    gemini_client = None

VISION_PROMPT = """You are the vision assistant of a wheelchair user who cannot speak or move. 
        The image is the part of the room they are looking at (the {region} part of their view). 
        Identify the main object they are most likely looking at and infer what they probably want.
        Reply ONLY with JSON: {{"object": "<short name>", "intent": "<likely need>", 
        "speech": "<1-2 short sentences spoken to the user, start with 'I see ...' 
        and end with a yes/no question such as 'Would you like me to notify a caregiver?'>"}}"""

class VisionResponseSchema(BaseModel):
    object: str
    intent: str
    speech: str

def analyse_roi(jpeg_bytes, region):
    if not gemini_client:
        return {"object": "", "intent": "", "speech": "Gemini API missing.", "error": "Missing API Key"}
    try:
        image_part = types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg")
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[image_part, VISION_PROMPT.format(region=region)],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=VisionResponseSchema,
                temperature=0.2
            ),
        )
        result_json = json.loads(response.text)
        return {
            "object": result_json.get("object", ""),
            "intent": result_json.get("intent", ""),
            "speech": result_json.get("speech", "")
        }
    except Exception as e:
        return {"object": "", "intent": "", "speech": "Analysis failed.", "error": str(e)}

def log_to_dynamodb(event_type, details):
    """Safely and asynchronously write key event logs to DynamoDB."""
    try:
        dynamodb = aws("dynamodb")
        dynamodb.put_item(
            TableName=DYNAMO_TABLE_NAME,
            Item={
                "LogID": {"S": str(uuid.uuid4())},
                "Timestamp": {"N": str(int(time.time() * 1000))},
                "EventType": {"S": event_type},
                "Details": {"S": str(details)}
            }
        )
    except Exception as e:
        log.error(f"Failed to write log to DynamoDB: {e}")

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        mode = request.form.get("mode", "menu")
        frame = cv2.imdecode(np.frombuffer(request.files["image"].read(), np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Invalid image"}), 400
        
        h, w = frame.shape[:2]
        STATE["last_predict"], STATE["last_mode"] = time.time(), mode
        out = {"status": "success", "mode": mode, "img_w": w, "img_h": h, "detections": [],
               "eyes_closed": False, "mouth_open": False, "gaze": None}

        with inference_lock:
            STATE["frame_counter"] += 1
            dets, eyes_closed, mouth_open, best_eye_box = yolo11_pipeline(frame)
            
            out["eyes_closed"] = eyes_closed
            out["mouth_open"] = mouth_open
            out["detections"] += dets
            STATE["last_mouth_open"] = mouth_open

            if best_eye_box:
                STATE["last_valid_eye_box"] = best_eye_box
            elif STATE["last_valid_eye_box"]:
                best_eye_box = STATE["last_valid_eye_box"]

            # Requirement 1 & 2: Track gaze continuously in 'vision' mode as well as 'mobility' and 'menu'
            run_gaze_now = (mode in ("mobility", "vision"))
            if mode == "menu":
                run_gaze_now = STATE["frame_counter"] % GAZE_EVERY_N == 0
            
            if run_gaze_now and best_eye_box:
                gaze = run_gaze(frame, best_eye_box)
                out["gaze"] = gaze
                
                if mode == "mobility":
                    out["eyes_closed"] = eyes_closed or (gaze == "closed")

        return jsonify(out)
    except Exception as e:
        log.exception("predict error")
        return jsonify({"error": str(e)}), 500

@app.route("/mobility", methods=["POST"])
def mobility():
    cmd = (request.get_json(force=True) or {}).get("command", "stop")
    log_to_dynamodb("MOBILITY_COMMAND", f"Wheelchair command executed: {cmd}")
    if cmd not in ("left", "right", "forward", "stop"):
        return jsonify({"error": "bad command"}), 400
    return jsonify(publish_command(cmd))

@app.route("/estop", methods=["POST"])
def estop():
    mode = (request.get_json(force=True, silent=True) or {}).get("mode", "unknown")
    log_to_dynamodb("ESTOP_TRIGGERED", f"Emergency safety stop activated in {mode} mode.")
    iot = publish_command("stop", reason="estop")

    now = time.time()
    if now - STATE["last_estop"] >= ESTOP_SMS_COOLDOWN_S:
        STATE["last_estop"] = now
        sms = send_sms("EMERGENCY: The wheelchair user's eyes have been closed for over 8 seconds "
                       "(system was in %s mode). Please check on them immediately. Time: %s"
                       % (mode, time.strftime("%Y-%m-%d %H:%M:%S")))
    else:
        sms = {"status": "skipped", "detail": "cooldown"}
    return jsonify({"iot": iot, "sms": sms,
                    "audio_b64": polly_b64("Emergency. Your caregiver has been alerted.")})

@app.route("/vision", methods=["POST"])
def vision():
    img = request.files["image"].read()
    region = request.form.get("region", "centre")
    
    current_time = time.time()
    # Prevent hitting 429 by enforcing a local 12-second minimum window between Gemini calls
    if current_time - STATE["last_vision_time"] < 12.0 and STATE["last_vision_text"]:
        log.info("Vision request rate-limited locally to protect free tier quota. Returning cached context.")
        cached_speech = STATE["last_vision_text"]
        return jsonify({
            "object": "Cached Object",
            "intent": "Assistance",
            "speech": cached_speech,
            "audio_b64": polly_b64(cached_speech)
        })

    STATE["last_vision_time"] = current_time
    result = analyse_roi(img, region)
    
    # Handle API-level fallback gracefully if 429 occurs anyway
    if "error" in result and "429" in str(result["error"]):
        fallback_speech = "I am processing too many requests right now. Please wait a moment before trying again."
        result = {
            "object": "Rate Limited",
            "intent": "Wait",
            "speech": fallback_speech,
            "audio_b64": polly_b64(fallback_speech)
        }

    result["audio_b64"] = polly_b64(result["speech"])
    
    with inference_lock:
        STATE["last_vision_text"] = result.get("speech", "")
        
    return jsonify(result)


@app.route("/speak", methods=["POST"])
def speak():
    text = (request.get_json(force=True) or {}).get("text", "")
    return jsonify({"audio_b64": polly_b64(text)})

@app.route("/notify", methods=["POST"])
def notify():
    data = request.get_json(force=True, silent=True) or {}
    msg_context = data.get("message", "Assistance requested.")
    
    # Requirement 3: Bulletproof Context Handshake assembling cached text context for SNS SMS delivery
    with inference_lock:
        cached_text = STATE["last_vision_text"]

    if cached_text:
        msg_context = f"Caregiver Alert: User requires assistance regarding observed context -> {cached_text}"
    log_to_dynamodb("NOTIFICATION_SENT", f"SMS notification sent: {msg_context}")
    sms = send_sms(msg_context)
    return jsonify({
        "sms": sms, 
        "audio_b64": polly_b64("Your caregiver has been notified.")
    })

def _watchdog():
    while True:
        time.sleep(0.5)
        if STATE["cmd"] != "stop":
            stale = time.time() - STATE["last_predict"] > MOBILITY_WATCHDOG_S
            if stale or STATE["last_mode"] not in ("mobility", "vision"):
                log.warning("Watchdog: stopping wheelchair")
                publish_command("stop", reason="watchdog")

threading.Thread(target=_watchdog, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5001")), threaded=True)