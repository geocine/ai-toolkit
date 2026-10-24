# Face check for one dataset (photo checks, section 12.3), run by the Go
# server as a job: python photo_faces.py <.job_config.json>. It finds faces
# with YuNet, measures them, embeds them with SFace and writes faces.json
# next to the config: per-photo measurements plus the face-to-face cosine
# matrix. Face templates stay in this process's memory and are never
# written. Logs carry file names and numbers only. The Go judge decides
# every flag; this script only measures.
#
# Needs numpy, Pillow and OpenCV >= 4.10 (no torch, no onnxruntime).
# python -I photo_faces.py --selftest [--models DIR] checks an install.
import base64
import hashlib
import io
import json
import math
import os
import signal
import sqlite3
import struct
import sys
import tempfile
import time
import traceback
import urllib.request

ALGO = 1  # photocheck.FaceAlgo
MODEL_NAME = "yunet2023mar+sface2021dec"  # photocheck.FaceModel

# The pins Go passes in config.models (internal/photocheck/pyjob Models);
# --selftest uses this copy. pyjob_test keeps the two in step.
ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models/"
DEFAULT_MODELS = {
    "yunet4": {
        "url": ZOO + "face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        "bytes": 232589,
    },
    "yunet5": {
        "url": ZOO + "face_detection_yunet/face_detection_yunet_2026may.onnx",
        "sha256": "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0",
        "bytes": 229738,
    },
    "sface": {
        "url": ZOO + "face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "sha256": "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
        "bytes": 38696353,
    },
    "ppocr": {
        "url": ZOO + "text_detection_ppocr/text_detection_en_ppocrv3_2023may.onnx",
        "sha256": "03f550c6b406fda8bf54bd8327815f6c7e2edd98cea02348c93d879254366587",
        "bytes": 2423490,
    },
}

MIN_CV2 = (4, 10)
MIN_SCORE = 0.6  # faces below this are dropped (the judge's candidate score)
MAX_FACES = 8  # per photo, largest first
MAX_EMBED = 6  # per photo, largest first
EMBED_MIN_ALIGN = 40  # SFace templates of smaller faces are poor
ALIGN_LONG = 1280  # A: the alignment image
DETECT_LONG = 640  # D: YuNet's input
DETECT_RETRY = 320  # D again when nothing was found
SHARP_EDGE = 1024  # S: area-equivalent edge, the section 3 scale
TEXT_LONG = 736  # PP-OCR DB input
TEXT_MIN_SCORE = 0.3  # the judge counts boxes from 0.6; lower ones help calibration
MAX_TEXT = 32
DOWNLOAD_TIMEOUT = 60
CHECK_BYTES = 4 << 20  # a download polls stop and return_to_queue this often

# numpy, cv2 and Pillow load in load_libs(), so the helpers that need
# only the standard library import without them.
np = cv2 = Image = ImageOps = None
BOX = None


class Fail(Exception):
    """A failure with a message meant for the user (the job's info)."""


class Stop(BaseException):
    """The server asked the job to stop (status) with this info. A
    BaseException, like KeyboardInterrupt, so the per-model and per-photo
    `except Exception` handlers let it through."""

    def __init__(self, status, info):
        super().__init__(info)
        self.status, self.info = status, info


def log(msg):
    print(msg, flush=True)


# ---- the job's row --------------------------------------------------------


class Job:
    """The job's row in the server's database, written the way upstream's
    BaseCaptioner does: one connection per operation, BEGIN IMMEDIATE,
    retried while the database is locked. With no database or no
    AITK_JOB_ID (a manual run) every call is a no-op."""

    def __init__(self, db_path, job_id):
        ok = bool(db_path) and bool(job_id) and os.path.exists(db_path)
        self.db = db_path if ok else None
        self.id = job_id

    def _run(self, fn, tries=6):
        delay = 0.5
        for attempt in range(tries):
            try:
                conn = sqlite3.connect(self.db, timeout=30.0)
                conn.isolation_level = None
                try:
                    return fn(conn)
                finally:
                    conn.close()
            except sqlite3.OperationalError as e:
                if "locked" not in str(e) or attempt == tries - 1:
                    raise
                log(f"[AITK] Database locked, retrying in {delay:.1f}s")
                time.sleep(delay)
                delay *= 2

    def update(self, **cols):
        if not self.db:
            return
        sets = ", ".join(f"{k} = ?" for k in cols)  # keys are this file's literals

        def fn(conn):
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(f"UPDATE Job SET {sets} WHERE id = ?", (*cols.values(), self.id))
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")

        self._run(fn)

    def status(self, status, info):
        self.update(status=status, info=info)

    def check(self):
        """Raises Stop when the server set stop or return_to_queue."""
        if not self.db:
            return

        def fn(conn):
            return conn.execute("SELECT stop, return_to_queue FROM Job WHERE id = ?", (self.id,)).fetchone()

        row = self._run(fn)
        if row is None:
            return
        if truthy(row[0]):
            raise Stop("stopped", "Job stopped")
        if truthy(row[1]):
            raise Stop("queued", "Job queued")


def truthy(v):
    return v in (1, True, "1", "true")


# ---- standard-library helpers (imported by the Go test) --------------------


def pack_f16(values):
    """base64 of little-endian binary16 values (FaceMatrix.cos)."""
    vals = [max(-1.0, min(1.0, float(v))) for v in values]
    return base64.b64encode(struct.pack("<%de" % len(vals), *vals)).decode("ascii")


def unpack_f16(b64):
    raw = base64.b64decode(b64)
    return list(struct.unpack("<%de" % (len(raw) // 2), raw))


def parse_version(v):
    """(major, minor) of a cv2.__version__ such as '4.10.0' or '5.0.0-pre'."""
    out = []
    for part in str(v).split(".")[:2]:
        digits = ""
        for ch in part:
            if not ch.isdigit():
                break
            digits += ch
        out.append(int(digits or 0))
    while len(out) < 2:
        out.append(0)
    return tuple(out)


def fit_long(w, h, long):
    """Size whose long side is at most long; never grows."""
    m = max(w, h)
    if m <= long:
        return w, h
    f = long / m
    return max(1, round(w * f)), max(1, round(h * f))


def fit_area(w, h, edge):
    """Size whose area-equivalent edge sqrt(w*h) is at most edge; never grows."""
    f = min(1.0, edge / math.sqrt(w * h))
    return max(1, round(w * f)), max(1, round(h * f))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_atomic(path, data):
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix="." + os.path.basename(path) + ".", dir=d)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


# ---- models ---------------------------------------------------------------


class Models:
    """The opencv_zoo files, fetched once into dir and checked by sha256."""

    def __init__(self, conf, job):
        conf = conf or {}
        self.dir = conf.get("dir") or default_models_dir()
        self.spec = {k: dict(v) for k, v in DEFAULT_MODELS.items()}
        for k, v in conf.items():
            if isinstance(v, dict) and v.get("url"):
                self.spec[k] = v
        self.job = job

    def path(self, key):
        return os.path.join(self.dir, os.path.basename(self.spec[key]["url"]))

    def missing(self, key):
        p = self.path(key)
        if not os.path.isfile(p):
            return True
        want = self.spec[key].get("sha256")
        return bool(want) and sha256_file(p) != want

    def announce(self, keys):
        """Says what is about to be downloaded, in the job's info."""
        need = [k for k in keys if self.missing(k)]
        size = sum(int(self.spec[k].get("bytes") or 0) for k in need)
        if need:
            msg = "Downloading face models" + (f" ({max(1, round(size / 1e6))} MB)" if size else "")
            log(msg)
            self.job.check()
            self.job.status("running", msg)

    def ensure(self, key):
        p = self.path(key)
        if not self.missing(key):
            return p
        spec = self.spec[key]
        name = os.path.basename(p)
        os.makedirs(self.dir, exist_ok=True)
        log(f"downloading {name}")
        fd, tmp = tempfile.mkstemp(prefix="." + name + ".", dir=self.dir)
        try:
            h = hashlib.sha256()
            n = polled = 0
            req = urllib.request.Request(spec["url"], headers={"User-Agent": "aitk-photo-faces"})
            with os.fdopen(fd, "wb") as out, urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as r:
                for chunk in iter(lambda: r.read(1 << 20), b""):
                    h.update(chunk)
                    out.write(chunk)
                    n += len(chunk)
                    if n - polled >= CHECK_BYTES:
                        self.job.check()  # Stop leaves through the finally
                        polled = n
            want = spec.get("sha256")
            if want and h.hexdigest() != want:
                raise Fail(f"The face model {name} didn't match its checksum. Try again later.")
            os.replace(tmp, p)
            log(f"downloaded {name}: {n} bytes")
            return p
        except Fail:
            raise
        except Exception as e:
            raise Fail(f"Couldn't download the face model {name}: {e}") from e
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    def detector(self, version):
        """YuNet: 2023mar first; on OpenCV 5.x the dynamic-shape 2026may
        when 2023mar does not load or run there."""
        keys = ["yunet4"] + (["yunet5"] if version[0] >= 5 and "yunet5" in self.spec else [])
        errors = []
        for key in keys:
            try:
                det = cv2.FaceDetectorYN.create(self.ensure(key), "", (320, 320), MIN_SCORE, 0.3, 5000)
                # a non-square probe: a fixed-shape model fails here on 5.x
                det.setInputSize((480, 360))
                det.detect(np.zeros((360, 480, 3), np.uint8))
                return key, det
            except Fail:
                raise
            except Exception as e:
                errors.append(f"{os.path.basename(self.path(key))}: {e}")
                log(f"face detector {key} did not load: {e}")
        raise Fail(f"Couldn't load the face detector on OpenCV {cv2.__version__} ({'; '.join(errors)}).")

    def recognizer(self):
        try:
            return cv2.FaceRecognizerSF.create(self.ensure("sface"), "")
        except Fail:
            raise
        except Exception as e:
            raise Fail(f"Couldn't load the face recognizer on OpenCV {cv2.__version__}: {e}") from e

    def text(self):
        """PP-OCR DB, or None when this OpenCV cannot run it (text_checked
        stays false)."""
        if "ppocr" not in self.spec or not hasattr(cv2, "dnn_TextDetectionModel_DB"):
            log("text detection is not available in this OpenCV")
            return None
        try:
            m = cv2.dnn_TextDetectionModel_DB(cv2.dnn.readNet(self.ensure("ppocr")))
            m.setBinaryThreshold(0.3)
            m.setPolygonThreshold(0.5)
            m.setUnclipRatio(2.0)
            m.setMaxCandidates(200)
            return m
        except Fail as e:
            log(f"text detection is off: {e}")
        except Exception as e:
            log(f"text detection did not load: {e}")
        return None


def default_models_dir():
    root = os.environ.get("MODELS_PATH", "").strip()
    if root:
        return os.path.join(root, "opencv_zoo")
    return os.path.join(tempfile.gettempdir(), "aitk-opencv_zoo")


def load_libs():
    global np, cv2, Image, ImageOps, BOX
    try:
        import numpy
        from PIL import Image as image_mod, ImageOps as imageops_mod
    except ImportError as e:
        raise Fail(f"Checking faces needs numpy and Pillow on this server: {e}") from e
    try:
        import cv2 as cv2_mod
    except ImportError as e:
        raise Fail(f"Checking faces needs OpenCV 4.10 or newer; this server has none ({e}).") from e
    np, cv2, Image, ImageOps = numpy, cv2_mod, image_mod, imageops_mod
    BOX = getattr(getattr(Image, "Resampling", Image), "BOX")
    log(f"cv2 {cv2.__version__} at {cv2.__file__}")
    v = parse_version(cv2.__version__)
    if v < MIN_CV2:
        raise Fail(f"Checking faces needs OpenCV 4.10 or newer; this server has {cv2.__version__}.")
    return v


# ---- measuring (numpy, Pillow, cv2) ---------------------------------------


def luma(rgb):
    """Y = (19595R + 38470G + 7471B + 32768) >> 16, as float32 (section 3)."""
    c = rgb.astype(np.uint32)
    y = (19595 * c[:, :, 0] + 38470 * c[:, :, 1] + 7471 * c[:, :, 2] + 32768) >> 16
    return y.astype(np.float32)


def box_down(plane, w, h):
    """Area average of a float plane to w x h (Pillow's BOX filter)."""
    if plane.shape[1] == w and plane.shape[0] == h:
        return plane
    return np.asarray(Image.fromarray(plane.astype(np.float32)).resize((w, h), BOX), dtype=np.float32)


def laplacian(s):
    """The 3x3 Gaussian [1 2 1]/4 along rows then columns, then the 3x3
    Laplacian, both over the valid region: L[y, x] sits at S[y+2, x+2]."""
    s = s.astype(np.float64)
    t = (s[:, :-2] + 2 * s[:, 1:-1] + s[:, 2:]) / 4
    g = (t[:-2, :] + 2 * t[1:-1, :] + t[2:, :]) / 4
    return g[1:-1, :-2] + g[1:-1, 2:] + g[:-2, 1:-1] + g[2:, 1:-1] - 4 * g[1:-1, 1:-1]


def frame_sharp(s, lap):
    """Nearest-rank p90 of per-tile Laplacian variance: Go's sharpness."""
    h, w = s.shape
    if w < 5 or h < 5:
        return 0.0
    lh, lw = lap.shape
    nx, ny = min(max(w // 32, 1), 8), min(max(h // 32, 1), 8)
    vals = []
    for ty in range(ny):
        y0, y1 = ty * lh // ny, (ty + 1) * lh // ny
        for tx in range(nx):
            x0, x1 = tx * lw // nx, (tx + 1) * lw // nx
            tile = lap[y0:y1, x0:x1]
            if tile.size:
                vals.append(max(0.0, float(np.mean(tile * tile) - np.mean(tile) ** 2)))
    if not vals:
        return 0.0
    vals.sort()
    return vals[int(math.ceil(0.9 * len(vals))) - 1]


def box_stats(s, lap, box):
    """sharp (Laplacian variance), mean_y and clip_hi inside a box of
    fractions, on S."""
    h, w = s.shape
    x0 = max(0, int(math.floor(box[0] * w)))
    y0 = max(0, int(math.floor(box[1] * h)))
    x1 = min(w, int(math.ceil((box[0] + box[2]) * w)))
    y1 = min(h, int(math.ceil((box[1] + box[3]) * h)))
    if x1 <= x0 or y1 <= y0:
        return 0.0, 0.0, 0.0
    px = s[y0:y1, x0:x1]
    mean_y = float(px.mean())
    clip_hi = float((px >= 250).mean())
    lv = lap[max(0, y0 - 2):max(0, y1 - 2), max(0, x0 - 2):max(0, x1 - 2)]
    sharp = 0.0
    if lv.size >= 9:
        sharp = max(0.0, float(np.mean(lv * lv) - np.mean(lv) ** 2))
    return sharp, mean_y, clip_hi


def resized_rgb(im, w, h):
    if im.size == (w, h):
        return im
    return im.resize((w, h), BOX)


def detect(det, im_a, long):
    """YuNet on A scaled to long; rows in that copy's pixels, its size."""
    w, h = fit_long(im_a.size[0], im_a.size[1], long)
    d = np.ascontiguousarray(np.asarray(resized_rgb(im_a, w, h))[:, :, ::-1])
    det.setInputSize((w, h))
    _, rows = det.detect(d)
    return ([] if rows is None else list(rows)), w, h


def clamp01(v):
    return min(1.0, max(0.0, float(v)))


def detect_text(model, a_bgr):
    """PP-OCR DB boxes as [x, y, w, h, score] fractions of the image."""
    ah, aw = a_bgr.shape[:2]
    w, h = fit_long(aw, ah, TEXT_LONG)
    w, h = max(32, round(w / 32) * 32), max(32, round(h / 32) * 32)
    img = cv2.resize(a_bgr, (w, h), interpolation=cv2.INTER_AREA)
    model.setInputParams(1.0 / 255.0, (w, h), (122.67891434, 116.66876762, 104.00698793))
    boxes, scores = model.detect(img)
    out = []
    for quad, score in zip(boxes, scores):
        score = float(score)
        if score < TEXT_MIN_SCORE:
            continue
        q = np.asarray(quad, dtype=np.float64).reshape(-1, 2)
        x0, y0 = clamp01(q[:, 0].min() / w), clamp01(q[:, 1].min() / h)
        x1, y1 = clamp01(q[:, 0].max() / w), clamp01(q[:, 1].max() / h)
        if x1 > x0 and y1 > y0:
            out.append([round(x0, 4), round(y0, 4), round(x1 - x0, 4), round(y1 - y0, 4), round(score, 4)])
    out.sort(key=lambda b: -b[4])
    return out[:MAX_TEXT]


def to_8bit(im):
    """16-bit and 32-bit greyscale as 8-bit L, the high byte as Go reads a
    Gray16: convert("RGB") clips those modes, so a gray16 PNG would turn
    almost white. Other modes are returned as they are."""
    if im.mode.startswith("I;16"):
        a = np.asarray(im).astype(np.uint32) >> 8
    elif im.mode in ("I", "F"):
        a = np.asarray(im, dtype=np.float64)
        hi = float(a.max()) if a.size else 0.0
        if hi > 255:  # a 16-bit PNG that older Pillow opens as I
            a = a / 257.0 if hi <= 65535 else a * (255.0 / hi)
        a = np.rint(a)
    else:
        return im
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "L")


def measure(data, det, rec, text_model):
    """One photo's FaceData (without algo) and its face templates, in the
    order of FaceData.faces for the embedded ones."""
    im = Image.open(io.BytesIO(data))
    nw, nh = im.size
    try:
        orientation = int(im.getexif().get(0x0112, 1) or 1)
    except Exception:
        orientation = 1
    if orientation in (5, 6, 7, 8):
        nw, nh = nh, nw
    if im.format == "JPEG":
        im.draft("RGB", (ALIGN_LONG, ALIGN_LONG))
    im = to_8bit(ImageOps.exif_transpose(im)).convert("RGB")

    # A: alignment image; S: sharpness copy (section 3 scale) of its luma.
    aw, ah = fit_long(im.size[0], im.size[1], ALIGN_LONG)
    im_a = resized_rgb(im, aw, ah)
    a_rgb = np.asarray(im_a)
    a_bgr = np.ascontiguousarray(a_rgb[:, :, ::-1])
    sw, sh = fit_area(aw, ah, SHARP_EDGE)
    s = box_down(luma(a_rgb), sw, sh)
    lap = laplacian(s) if sw >= 5 and sh >= 5 else np.zeros((0, 0))
    fs = frame_sharp(s, lap)

    rows, dw, dh = detect(det, im_a, DETECT_LONG)
    if not rows and max(dw, dh) > DETECT_RETRY:
        rows, dw, dh = detect(det, im_a, DETECT_RETRY)
    rows = [r for r in rows if float(r[14]) >= MIN_SCORE]
    rows.sort(key=lambda r: -float(r[2]) * float(r[3]))
    rows = rows[:MAX_FACES]

    faces, templates = [], []
    for r in rows:
        x0, y0 = clamp01(r[0] / dw), clamp01(r[1] / dh)
        x1, y1 = clamp01((r[0] + r[2]) / dw), clamp01((r[1] + r[3]) / dh)
        box = [x0, y0, x1 - x0, y1 - y0]
        kps = [[round(clamp01(r[4 + 2 * i] / dw), 5), round(clamp01(r[5 + 2 * i] / dh), 5)] for i in range(5)]
        side = round(max(box[2] * nw, box[3] * nh))
        align_side = round(max(box[2] * aw, box[3] * ah))
        sharp, mean_y, clip_hi = box_stats(s, lap, box)
        face = {
            "box": [round(v, 5) for v in box],
            "kps": kps,
            "score": round(float(r[14]), 4),
            "side": int(side),
            "align_side": int(align_side),
            "sharp": round(sharp, 4),
            "mean_y": round(mean_y, 2),
            "clip_hi": round(clip_hi, 4),
            "embedded": False,
        }
        if rec is not None and len(templates) < MAX_EMBED and align_side >= EMBED_MIN_ALIGN:
            row = np.asarray(r, dtype=np.float32).copy()
            row[0:14:2] *= aw / dw
            row[1:14:2] *= ah / dh
            feat = np.asarray(rec.feature(rec.alignCrop(a_bgr, row)), dtype=np.float32).reshape(-1)
            n = float(np.linalg.norm(feat))
            if n > 0:
                templates.append(feat / n)
                face["embedded"] = True
        faces.append(face)

    fd = {"faces": faces, "frame_sharp": round(fs, 4), "text_checked": False}
    if text_model is not None:
        try:
            boxes = detect_text(text_model, a_bgr)
            fd["text_checked"] = True
            if boxes:
                fd["text"] = boxes
        except Exception as e:
            log(f"text detection failed: {e}")
    return fd, templates


def cos_upper(templates):
    """The upper triangle (i < j, row-major) of the cosine matrix of
    L2-normalised templates, as base64 binary16, one row at a time."""
    if len(templates) < 2:
        return ""
    v = np.stack(templates).astype(np.float32)
    out = bytearray()
    for i in range(len(v) - 1):
        row = np.clip(v[i + 1:] @ v[i], -1.0, 1.0)
        out += row.astype("<f2").tobytes()
    return base64.b64encode(bytes(out)).decode("ascii")


# ---- the job --------------------------------------------------------------


def run(p, job_dir, job):
    files = p.get("files") or []
    threads = int(p.get("threads") or 2)
    text = bool(p.get("text"))
    out = os.path.join(job_dir, os.path.basename(p.get("out") or "faces.json"))
    dataset = p.get("dataset_dir") or ""

    version = load_libs()
    cv2.setNumThreads(max(1, threads))
    job.check()  # a stop pressed while the job started
    models = Models(p.get("models"), job)
    models.announce(["yunet4", "sface"] + (["ppocr"] if text else []))
    det_key, det = models.detector(version)
    rec = models.recognizer()
    text_model = models.text() if text else None
    model_name = MODEL_NAME if det_key == "yunet4" else "yunet2026may+sface2021dec"

    job.check()  # so "running" never overwrites the server's "stopping"
    job.update(total_steps=len(files), step=0)
    job.status("running", f"Checking faces in {len(files)} photos")
    log(f"checking {len(files)} photos with {det_key}, text {'on' if text_model else 'off'}")

    items, skipped, ids, templates = {}, [], [], []
    nfaces = 0
    for step, f in enumerate(files, 1):
        job.check()
        name, sha = f.get("file") or "", (f.get("sha256") or "").lower()
        path = os.path.join(dataset, name)
        try:
            with open(path, "rb") as fh:
                data = fh.read()
        except OSError:
            log(f"{name}: missing, skipped")
            skipped.append(name)
            job.update(step=step)
            continue
        if sha and hashlib.sha256(data).hexdigest() != sha:
            log(f"{name}: changed since the check started, skipped")
            skipped.append(name)
            job.update(step=step)
            continue
        if not sha:
            sha = hashlib.sha256(data).hexdigest()
        if sha in items:  # an exact duplicate: one entry, one set of matrix rows
            log(f"{name}: same photo as an earlier file, skipped")
            job.update(step=step)
            continue
        try:
            fd, temps = measure(data, det, rec, text_model)
            k = 0
            for i, face in enumerate(fd["faces"]):
                if face["embedded"]:
                    ids.append(f"{sha}:{i}")
                    templates.append(temps[k])
                    k += 1
            nfaces += len(fd["faces"])
            log(f"{name}: {len(fd['faces'])} faces, {k} embedded")
        except Exception as e:
            msg = "can't read this image" if isinstance(e, Image.UnidentifiedImageError) else (str(e) or e.__class__.__name__)
            log(f"{name}: error: {msg}")
            fd = {"faces": [], "frame_sharp": 0, "text_checked": False, "error": msg}
        fd = {"algo": ALGO, **fd}
        items[sha] = fd
        job.update(step=step)

    result = {
        "algo": ALGO,
        "cv2": cv2.__version__,
        "model": model_name,
        "items": items,
        "identity": {"faces": ids, "cos": cos_upper(templates)},
        "skipped": skipped,
    }
    templates.clear()
    write_atomic(out, json.dumps(result, separators=(",", ":")).encode("utf-8"))
    msg = f"Checked {len(items)} photos, {nfaces} faces"
    log(msg)
    job.status("completed", msg)


# ---- selftest -------------------------------------------------------------


def selftest(args):
    """Checks this install: OpenCV version, a detector that loads (fetched
    into --models DIR when missing), no face on a flat grey image, and the
    binary16 matrix round trip."""
    models_dir = None
    if "--models" in args:
        i = args.index("--models")
        if i + 1 < len(args):
            models_dir = args[i + 1]
    try:
        version = load_libs()
        a, b, c = np.array([1, 0, 0], np.float32), np.array([0.6, 0.8, 0], np.float32), np.array([0, 0.6, 0.8], np.float32)
        got = unpack_f16(cos_upper([a, b, c]))
        want = [0.6, 0.0, 0.48]
        if len(got) != 3 or any(abs(g - w) > 1e-3 for g, w in zip(got, want)):
            raise Fail(f"binary16 round trip gave {got}, want {want}")
        if unpack_f16(pack_f16(want)) != got:
            raise Fail("pack_f16 and numpy binary16 disagree")
        g16 = Image.frombytes("I;16", (256, 1), np.arange(0, 65536, 256, dtype="<u2").tobytes())
        mean = float(np.asarray(to_8bit(g16).convert("RGB")).mean())
        if abs(mean - 127.5) > 1:
            raise Fail(f"a 16-bit grey ramp reads with mean {mean:.1f}, want 127.5")
        models = Models({"dir": models_dir} if models_dir else None, Job(None, None))
        log(f"models in {models.dir}")
        key, det = models.detector(version)
        buf = io.BytesIO()
        Image.new("RGB", (256, 256), (128, 128, 128)).save(buf, "PNG")
        fd, temps = measure(buf.getvalue(), det, None, None)
        if fd["faces"] or temps:
            raise Fail(f"{len(fd['faces'])} faces on a flat grey image")
        if fd["frame_sharp"] != 0:
            raise Fail(f"frame_sharp {fd['frame_sharp']} on a flat grey image")
        log(f"selftest ok: OpenCV {cv2.__version__}, detector {key}")
        return 0
    except Fail as e:
        log(f"selftest failed: {e}")
        return 1


def main(argv):
    if "--selftest" in argv[1:]:
        return selftest(argv[1:])
    if len(argv) < 2:
        log("usage: photo_faces.py <job_config.json> | --selftest [--models DIR]")
        return 2
    cfg_path = os.path.abspath(argv[1])
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    p = cfg["config"]["process"][0]
    job = Job(p.get("sqlite_db_path"), (os.environ.get("AITK_JOB_ID") or "").strip())

    def interrupt(*_):
        raise KeyboardInterrupt

    if hasattr(signal, "SIGBREAK"):  # Windows: the stop button's console break
        signal.signal(signal.SIGBREAK, interrupt)
    try:
        run(p, os.path.dirname(cfg_path), job)
        return 0
    except Stop as s:
        log(s.info)
        job.status(s.status, s.info)
        return 0
    except KeyboardInterrupt:
        log("Job stopped")
        job.status("stopped", "Job stopped")
        return 0
    except Fail as e:
        log(f"error: {e}")
        job.status("error", str(e))
        return 1
    except Exception as e:
        traceback.print_exc()
        job.status("error", f"Face check failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
