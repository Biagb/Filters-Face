import math
import os
import random
import threading
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

try:
    import winsound
except Exception:
    winsound = None


class FaceFilterApp:
    def __init__(self, camera_index: int = 0, max_faces: int = 8):
        self.camera_index = camera_index
        self.max_faces = max_faces
        self.camera_backend_name = "Unknown"

        self.backend = "solutions"
        self.face_mesh = None
        self.face_landmarker = None
        self._init_mediapipe_backend()

        self.filter_mode = 1
        self.mode_names = {
            0: "Normal",
            1: "Beauty",
            2: "Warm",
            3: "Cool",
            4: "B&W",
            5: "Sunglasses",
        }

        self.background_mode = 0
        self.background_names = {
            0: "Original",
            1: "Blur",
            2: "Sky",
            3: "Sunset",
        }

        self.last_time = time.time()
        self.fps = 0.0
        self.last_mode_change_time = 0.0
        self.frame_timestamp_ms = 0
        self.last_background_change_time = 0.0

        self.last_wink_time = 0.0
        self.last_mouth_open_time = 0.0
        self.last_smile_time = 0.0
        self.gesture_status = ""
        self.gesture_status_until = 0.0
        self.confetti_until = 0.0
        self.emotion_label = "Neutral"
        self.symmetry_score = 100.0

        self.ear_threshold = 0.20
        self.ear_trigger_seconds = 2.0
        self.current_ear = 0.0
        self.low_ear_start_time = None
        self.alarm_active = False
        self._alarm_stop_event = threading.Event()
        self._alarm_thread = None

        self.confetti_particles = [
            {
                "x": random.random(),
                "y": random.random(),
                "vx": random.uniform(-0.003, 0.003),
                "vy": random.uniform(0.004, 0.015),
                "r": random.randint(2, 6),
                "color": (
                    random.randint(40, 255),
                    random.randint(40, 255),
                    random.randint(40, 255),
                ),
            }
            for _ in range(140)
        ]

    def _init_mediapipe_backend(self):
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            self.backend = "solutions"
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=self.max_faces,
                refine_landmarks=True,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6,
            )
            return

        self.backend = "tasks"
        model_path = self._ensure_face_landmarker_model()
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=self.max_faces,
            min_face_detection_confidence=0.6,
            min_face_presence_confidence=0.6,
            min_tracking_confidence=0.6,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def _ensure_face_landmarker_model(self) -> str:
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, "face_landmarker.task")

        if os.path.exists(model_path):
            return model_path

        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
        urllib.request.urlretrieve(url, model_path)
        return model_path

    @staticmethod
    def _landmarks_to_points(face_landmarks, width: int, height: int) -> np.ndarray:
        points = []
        landmarks = face_landmarks.landmark if hasattr(face_landmarks, "landmark") else face_landmarks
        for lm in landmarks:
            px = int(lm.x * width)
            py = int(lm.y * height)
            px = min(max(px, 0), width - 1)
            py = min(max(py, 0), height - 1)
            points.append((px, py))
        return np.array(points, dtype=np.int32)

    @staticmethod
    def _create_face_mask(shape, face_points: np.ndarray) -> np.ndarray:
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        hull = cv2.convexHull(face_points)
        cv2.fillConvexPoly(mask, hull, 255)
        mask = cv2.GaussianBlur(mask, (41, 41), 0)
        return mask

    @staticmethod
    def _blend_with_mask(base: np.ndarray, filtered: np.ndarray, mask: np.ndarray) -> np.ndarray:
        alpha = (mask.astype(np.float32) / 255.0)[..., None]
        blended = base.astype(np.float32) * (1.0 - alpha) + filtered.astype(np.float32) * alpha
        return np.clip(blended, 0, 255).astype(np.uint8)

    def _cycle_filter(self, direction: int):
        total = len(self.mode_names)
        self.filter_mode = (self.filter_mode + direction) % total
        self.last_mode_change_time = time.time()

    def _cycle_background(self):
        total = len(self.background_names)
        self.background_mode = (self.background_mode + 1) % total
        self.last_background_change_time = time.time()

    def _set_gesture_status(self, text: str, duration: float = 1.0):
        self.gesture_status = text
        self.gesture_status_until = time.time() + duration

    @staticmethod
    def _eye_aspect_ratio(points: np.ndarray, outer_idx: int, inner_idx: int, top_idx: int, bottom_idx: int) -> float:
        eye_width = np.linalg.norm(points[outer_idx] - points[inner_idx])
        eye_height = np.linalg.norm(points[top_idx] - points[bottom_idx])
        if eye_width < 1e-5:
            return 0.0
        return float(eye_height / eye_width)

    def _handle_gesture_triggers(self, points: np.ndarray):
        now = time.time()

        mouth_width = np.linalg.norm(points[61] - points[291])
        mouth_height = np.linalg.norm(points[13] - points[14])
        smile_ratio = mouth_width / max(mouth_height, 1.0)

        face_span = np.linalg.norm(points[33] - points[263])
        mouth_open_ratio = mouth_height / max(face_span, 1.0)

        left_ear = self._eye_aspect_ratio(points, 33, 133, 159, 145)
        right_ear = self._eye_aspect_ratio(points, 263, 362, 386, 374)

        left_eye_center = ((points[33] + points[133]) // 2).astype(np.int32)
        right_eye_center = ((points[263] + points[362]) // 2).astype(np.int32)
        left_mouth_corner = points[61]
        right_mouth_corner = points[291]

        left_corner_to_eye = np.linalg.norm(left_mouth_corner - left_eye_center)
        right_corner_to_eye = np.linalg.norm(right_mouth_corner - right_eye_center)
        avg_corner_to_eye_norm = ((left_corner_to_eye + right_corner_to_eye) / 2.0) / max(face_span, 1.0)

        if mouth_open_ratio > 0.22 and avg_corner_to_eye_norm > 0.43:
            self.emotion_label = "Surprised"
        elif avg_corner_to_eye_norm < 0.41 and mouth_width / max(face_span, 1.0) > 0.40:
            self.emotion_label = "Smiling"
        else:
            self.emotion_label = "Neutral"

        if left_ear < 0.18 and right_ear > 0.24 and now - self.last_wink_time > 1.2:
            self._cycle_filter(+1)
            self.last_wink_time = now
            self._set_gesture_status("Wink detected: next filter")

        if mouth_open_ratio > 0.20 and now - self.last_mouth_open_time > 1.2:
            self._cycle_background()
            self.last_mouth_open_time = now
            self._set_gesture_status("Mouth open: next background")

        if smile_ratio > 2.3 and mouth_height > 6 and now - self.last_smile_time > 1.8:
            self.last_smile_time = now
            self.confetti_until = now + 1.1
            self._set_gesture_status("Smile detected: confetti!")

    @staticmethod
    def _create_sky_background(shape) -> np.ndarray:
        h, w = shape[:2]
        y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        top = np.array([255, 210, 120], dtype=np.float32)
        bottom = np.array([150, 220, 255], dtype=np.float32)
        grad = top * (1.0 - y) + bottom * y
        background = np.repeat(grad[:, None, :], w, axis=1)
        return background.astype(np.uint8)

    @staticmethod
    def _create_sunset_background(shape) -> np.ndarray:
        h, w = shape[:2]
        y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        top = np.array([255, 120, 60], dtype=np.float32)
        bottom = np.array([80, 30, 160], dtype=np.float32)
        grad = top * (1.0 - y) + bottom * y
        background = np.repeat(grad[:, None, :], w, axis=1)

        center = (int(w * 0.8), int(h * 0.22))
        radius = max(20, int(min(h, w) * 0.08))
        sun_overlay = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.circle(sun_overlay, center, radius, (120, 235, 255), -1, cv2.LINE_AA)
        background = cv2.addWeighted(background.astype(np.uint8), 1.0, sun_overlay, 0.7, 0)
        return background.astype(np.uint8)

    def _apply_ar_background(self, frame: np.ndarray, face_mask: np.ndarray) -> np.ndarray:
        if self.background_mode == 0:
            return frame

        if self.background_mode == 1:
            bg = cv2.GaussianBlur(frame, (0, 0), sigmaX=22, sigmaY=22)
        elif self.background_mode == 2:
            bg = self._create_sky_background(frame.shape)
        else:
            bg = self._create_sunset_background(frame.shape)

        inv_mask = cv2.bitwise_not(face_mask)
        alpha_bg = (inv_mask.astype(np.float32) / 255.0)[..., None]
        mixed = frame.astype(np.float32) * (1.0 - alpha_bg) + bg.astype(np.float32) * alpha_bg
        return np.clip(mixed, 0, 255).astype(np.uint8)

    def _draw_confetti(self, frame: np.ndarray) -> np.ndarray:
        if time.time() > self.confetti_until:
            return frame

        h, w = frame.shape[:2]
        overlay = frame.copy()
        for particle in self.confetti_particles:
            particle["x"] += particle["vx"]
            particle["y"] += particle["vy"]

            if particle["x"] < 0.0:
                particle["x"] = 1.0
            elif particle["x"] > 1.0:
                particle["x"] = 0.0

            if particle["y"] > 1.0:
                particle["y"] = 0.0
                particle["x"] = random.random()

            x = int(particle["x"] * w)
            y = int(particle["y"] * h)
            cv2.circle(overlay, (x, y), particle["r"], particle["color"], -1, cv2.LINE_AA)

        return cv2.addWeighted(overlay, 0.38, frame, 0.62, 0)

    def _alarm_loop(self):
        while not self._alarm_stop_event.is_set():
            if winsound is not None:
                winsound.Beep(2200, 350)
                if self._alarm_stop_event.is_set():
                    break
                winsound.Beep(1700, 350)
            else:
                time.sleep(0.7)

    def _start_alarm(self):
        if self.alarm_active:
            return
        self.alarm_active = True
        self._alarm_stop_event.clear()
        self._alarm_thread = threading.Thread(target=self._alarm_loop, daemon=True)
        self._alarm_thread.start()

    def _stop_alarm(self):
        if not self.alarm_active:
            return
        self.alarm_active = False
        self._alarm_stop_event.set()

    def _compute_average_ear(self, points: np.ndarray) -> float:
        left_ear = self._eye_aspect_ratio(points, 33, 133, 159, 145)
        right_ear = self._eye_aspect_ratio(points, 263, 362, 386, 374)
        return float((left_ear + right_ear) / 2.0)

    def _update_ear_alarm(self, points: np.ndarray):
        now = time.time()
        self.current_ear = self._compute_average_ear(points)

        if self.current_ear < self.ear_threshold:
            if self.low_ear_start_time is None:
                self.low_ear_start_time = now
            elif now - self.low_ear_start_time >= self.ear_trigger_seconds:
                self._start_alarm()
        else:
            self.low_ear_start_time = None
            self._stop_alarm()

    @staticmethod
    def _compute_symmetry_score(points: np.ndarray) -> float:
        nose_center_x = float(points[1][0])

        left_eye_indices = [33, 133, 159, 145]
        right_eye_indices = [263, 362, 386, 374]

        left_distances = [abs(float(points[idx][0]) - nose_center_x) for idx in left_eye_indices]
        right_distances = [abs(float(points[idx][0]) - nose_center_x) for idx in right_eye_indices]

        mean_left = float(np.mean(left_distances))
        mean_right = float(np.mean(right_distances))

        baseline = max((mean_left + mean_right) / 2.0, 1.0)
        asymmetry_ratio = abs(mean_left - mean_right) / baseline
        score = 100.0 * (1.0 - asymmetry_ratio)
        return float(np.clip(score, 0.0, 100.0))

    def _apply_beauty(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        smooth = cv2.bilateralFilter(frame, d=9, sigmaColor=80, sigmaSpace=80)
        warm = smooth.astype(np.float32)
        warm[:, :, 2] *= 1.08
        warm[:, :, 1] *= 1.03
        warm[:, :, 0] *= 0.95
        warm = np.clip(warm, 0, 255).astype(np.uint8)
        return self._blend_with_mask(frame, warm, mask)

    def _apply_warm(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        warm = frame.astype(np.float32)
        warm[:, :, 2] *= 1.12
        warm[:, :, 1] *= 1.06
        warm[:, :, 0] *= 0.92
        warm = cv2.convertScaleAbs(warm, alpha=1.02, beta=4)
        return self._blend_with_mask(frame, warm, mask)

    def _apply_cool(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        cool = frame.astype(np.float32)
        cool[:, :, 0] *= 1.15
        cool[:, :, 1] *= 1.02
        cool[:, :, 2] *= 0.92
        cool = cv2.convertScaleAbs(cool, alpha=1.0, beta=-2)
        return self._blend_with_mask(frame, cool, mask)

    def _apply_bw(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return self._blend_with_mask(frame, gray_bgr, mask)

    @staticmethod
    def _draw_sunglasses(frame: np.ndarray, points: np.ndarray) -> np.ndarray:
        left_eye_outer = points[33]
        right_eye_outer = points[263]
        left_eye_inner = points[133]
        right_eye_inner = points[362]

        eye_center_left = ((left_eye_outer + left_eye_inner) // 2).astype(np.int32)
        eye_center_right = ((right_eye_outer + right_eye_inner) // 2).astype(np.int32)

        eye_distance = np.linalg.norm(eye_center_right - eye_center_left)
        if eye_distance < 5:
            return frame

        width = int(eye_distance * 0.95)
        lens_w = max(20, int(width * 0.44))
        lens_h = max(15, int(lens_w * 0.65))

        angle = math.degrees(
            math.atan2(
                int(eye_center_right[1] - eye_center_left[1]),
                int(eye_center_right[0] - eye_center_left[0]),
            )
        )

        overlay = frame.copy()
        color = (25, 25, 25)

        cv2.ellipse(overlay, tuple(eye_center_left), (lens_w // 2, lens_h // 2), angle, 0, 360, color, -1)
        cv2.ellipse(overlay, tuple(eye_center_right), (lens_w // 2, lens_h // 2), angle, 0, 360, color, -1)

        bridge_center = ((eye_center_left + eye_center_right) // 2).astype(np.int32)
        bridge_w = max(8, int(lens_w * 0.25))
        bridge_h = max(4, int(lens_h * 0.18))

        box = cv2.boxPoints(((float(bridge_center[0]), float(bridge_center[1])), (float(bridge_w), float(bridge_h)), float(angle)))
        box = np.int32(box)
        cv2.fillConvexPoly(overlay, box, color)

        blended = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

        left_temple = points[127]
        right_temple = points[356]
        cv2.line(blended, tuple(eye_center_left), tuple(left_temple), color, 2, cv2.LINE_AA)
        cv2.line(blended, tuple(eye_center_right), tuple(right_temple), color, 2, cv2.LINE_AA)

        return blended

    @staticmethod
    def _draw_blush(frame: np.ndarray, points: np.ndarray) -> np.ndarray:
        left_cheek = points[205]
        right_cheek = points[425]

        cheek_dist = np.linalg.norm(left_cheek - right_cheek)
        radius = max(8, int(cheek_dist * 0.06))

        overlay = np.zeros_like(frame)
        cv2.circle(overlay, tuple(left_cheek), radius, (140, 100, 220), -1, cv2.LINE_AA)
        cv2.circle(overlay, tuple(right_cheek), radius, (140, 100, 220), -1, cv2.LINE_AA)

        overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=8, sigmaY=8)
        return cv2.addWeighted(frame, 1.0, overlay, 0.22, 0)

    def _apply_sunglasses_mode(self, frame: np.ndarray, face_points: np.ndarray, mask: np.ndarray) -> np.ndarray:
        output = self._draw_sunglasses(frame, face_points)
        output = self._draw_blush(output, face_points)
        output = self._apply_beauty(output, mask)
        return output

    def _process_one_face(self, frame: np.ndarray, face_points: np.ndarray) -> np.ndarray:
        mask = self._create_face_mask(frame.shape, face_points)

        if self.filter_mode == 0:
            return frame
        if self.filter_mode == 1:
            return self._apply_beauty(frame, mask)
        if self.filter_mode == 2:
            return self._apply_warm(frame, mask)
        if self.filter_mode == 3:
            return self._apply_cool(frame, mask)
        if self.filter_mode == 4:
            return self._apply_bw(frame, mask)
        if self.filter_mode == 5:
            return self._apply_sunglasses_mode(frame, face_points, mask)

        return frame

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = frame.copy()
        face_landmarks_list = []
        face_points_list = []

        if self.backend == "solutions":
            result = self.face_mesh.process(rgb)
            if result.multi_face_landmarks:
                face_landmarks_list = result.multi_face_landmarks
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            self.frame_timestamp_ms += 33
            detection_result = self.face_landmarker.detect_for_video(mp_image, self.frame_timestamp_ms)
            if detection_result.face_landmarks:
                face_landmarks_list = detection_result.face_landmarks

        if face_landmarks_list:
            h, w = output.shape[:2]
            combined_face_mask = np.zeros((h, w), dtype=np.uint8)

            for face_landmarks in face_landmarks_list:
                points = self._landmarks_to_points(face_landmarks, w, h)
                face_points_list.append(points)

                single_face_mask = self._create_face_mask(output.shape, points)
                combined_face_mask = cv2.max(combined_face_mask, single_face_mask)

                output = self._process_one_face(output, points)

            output = self._apply_ar_background(output, combined_face_mask)

            self._handle_gesture_triggers(face_points_list[0])
            self.symmetry_score = self._compute_symmetry_score(face_points_list[0])
            self._update_ear_alarm(face_points_list[0])
        else:
            self.emotion_label = "Neutral"
            self.symmetry_score = 0.0
            self.current_ear = 0.0
            self.low_ear_start_time = None
            self._stop_alarm()
            output = self._apply_ar_background(output, np.zeros(output.shape[:2], dtype=np.uint8))

        current_time = time.time()
        delta = current_time - self.last_time
        if delta > 0:
            self.fps = 1.0 / delta
        self.last_time = current_time

        cv2.putText(
            output,
            f"Mode: {self.mode_names.get(self.filter_mode, 'Unknown')}  |  BG: {self.background_names.get(self.background_mode, 'Unknown')}  |  Faces: {len(face_landmarks_list)}  |  MP: {self.backend}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (50, 230, 50),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            f"FPS: {self.fps:.1f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (50, 230, 50),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            f"Emotion: {self.emotion_label}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 220, 80),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            f"Symmetry: {self.symmetry_score:.1f}/100",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (170, 255, 170),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            f"EAR: {self.current_ear:.3f}",
            (10, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (180, 230, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            f"Camera: index {self.camera_index} via {self.camera_backend_name}",
            (10, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
        if self.alarm_active:
            cv2.putText(
                output,
                "ALARM: Eyes closed > 2s",
                (10, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        if len(face_landmarks_list) == 0:
            cv2.putText(
                output,
                "No face detected. Move closer / increase light.",
                (10, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 180, 255),
                2,
                cv2.LINE_AA,
            )

        if time.time() - self.last_mode_change_time < 1.0:
            cv2.putText(
                output,
                f"Switched to: {self.mode_names.get(self.filter_mode, 'Unknown')}",
                (10, 155),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if time.time() - self.last_background_change_time < 1.0:
            cv2.putText(
                output,
                f"Background: {self.background_names.get(self.background_mode, 'Unknown')}",
                (10, 185),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (220, 220, 255),
                2,
                cv2.LINE_AA,
            )

        if time.time() < self.gesture_status_until:
            cv2.putText(
                output,
                self.gesture_status,
                (10, 215),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (100, 255, 255),
                2,
                cv2.LINE_AA,
            )

        output = self._draw_confetti(output)

        cv2.putText(
            output,
            "Keys: N/P or Arrows = Filter | B = Background | 0..5 direct mode | Q Exit",
            (10, output.shape[0] - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return output

    def _open_camera(self):
        backends = [
            ("DSHOW", cv2.CAP_DSHOW),
            ("MSMF", cv2.CAP_MSMF),
            ("ANY", cv2.CAP_ANY),
        ]

        for backend_name, backend in backends:
            cap = cv2.VideoCapture(self.camera_index, backend)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    self.camera_backend_name = backend_name
                    return cap
            cap.release()

        return None

    def run(self):
        cap = self._open_camera()
        if cap is None:
            raise RuntimeError("Cannot open webcam. Check camera index/permissions.")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        window_name = "Live Face Filters"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                filtered_frame = self.process_frame(frame)
                cv2.imshow(window_name, filtered_frame)

                key_raw = cv2.waitKeyEx(1)
                key = key_raw & 0xFF

                if key in (ord("q"), ord("Q"), 27):
                    break

                numpad_map = {
                    96: 0,
                    97: 1,
                    98: 2,
                    99: 3,
                    100: 4,
                    101: 5,
                }

                if ord("0") <= key <= ord("5"):
                    self.filter_mode = key - ord("0")
                    self.last_mode_change_time = time.time()
                elif key_raw in numpad_map:
                    self.filter_mode = numpad_map[key_raw]
                    self.last_mode_change_time = time.time()
                elif key in (ord("n"), ord("N")) or key_raw == 2555904:
                    self._cycle_filter(+1)
                elif key in (ord("p"), ord("P")) or key_raw == 2424832:
                    self._cycle_filter(-1)
                elif key in (ord("b"), ord("B")):
                    self._cycle_background()
        finally:
            self._stop_alarm()
            cap.release()
            cv2.destroyAllWindows()
            if self.face_mesh is not None:
                self.face_mesh.close()
            if self.face_landmarker is not None:
                self.face_landmarker.close()


def main():
    app = FaceFilterApp(camera_index=0, max_faces=8)
    app.run()


if __name__ == "__main__":
    main()
