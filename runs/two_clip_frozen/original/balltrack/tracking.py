"""Non-neural candidates, camera motion, and a shared constant-velocity tracker."""
from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass
class Candidate:
    box: tuple
    score: float

    @property
    def center(self):
        x, y, w, h = self.box
        return np.array([x + w / 2, y + h / 2], dtype=float)


class CameraMotion:
    """Sparse optical flow + robust affine camera approximation, no learned features."""
    def __init__(self, cfg):
        self.cfg, self.previous = cfg, None

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (480, 270))
        affine = np.eye(2, 3, dtype=float)
        cut = False
        previous, self.previous = self.previous, small
        if previous is None:
            return affine, cut
        difference = float(np.mean(cv2.absdiff(previous, small)))
        cut = difference > self.cfg['cut_difference']
        if cut or not self.cfg['enabled']:
            return affine, cut
        points = cv2.goodFeaturesToTrack(previous, 150, 0.02, 12)
        if points is None or len(points) < 10:
            return affine, cut
        current, status, _ = cv2.calcOpticalFlowPyrLK(previous, small, points, None)
        if current is None:
            return affine, cut
        good = status.ravel().astype(bool)
        if good.sum() < 10:
            return affine, cut
        m, inliers = cv2.estimateAffinePartial2D(points[good], current[good], method=cv2.RANSAC)
        if m is not None and inliers.mean() > 0.5 and 0.9 < np.linalg.det(m[:, :2]) < 1.1:
            s = np.diag([frame.shape[1] / 480, frame.shape[0] / 270])
            affine[:, :2] = s @ m[:, :2] @ np.linalg.inv(s)
            affine[:, 2] = s @ m[:, 2]
        return affine, cut


class ClassicalDetector:
    def __init__(self, cfg):
        self.cfg = cfg

    def detect(self, frame):
        c = self.cfg
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Bright AND dark small structures; hue/whiteness is not a hard condition.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (c['kernel'], c['kernel']))
        response = cv2.max(cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel),
                           cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel))
        mask = (response >= c['contrast']).astype(np.uint8) * 255
        # Tiny closing joins fragmented ball pixels; opening could erase tiny balls.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        candidates = []
        for i in range(1, count):
            x, y, w, h, area = stats[i]
            aspect, fill = max(w / h, h / w), area / (w * h)
            if not c['min_area'] <= area <= c['max_area'] or aspect > c['max_aspect'] or fill < c['min_fill']:
                continue
            roi = response[y:y+h, x:x+w]
            contrast = float(roi[labels[y:y+h, x:x+w] == i].mean()) / 255
            # Broad shape preference tolerates elongated motion blur.
            score = 0.60 * contrast + 0.25 * fill + 0.15 / aspect
            if score >= c['min_score']:
                candidates.append(Candidate((float(x), float(y), float(w), float(h)), score))
        return sorted(candidates, key=lambda d: d.score, reverse=True)[:c['max_candidates']]


class BallTracker:
    def __init__(self, cfg, width, height, fps):
        self.cfg, self.width, self.height = cfg, width, height
        self.dt = 1 / fps
        self.max_gap = max(0, round(cfg['max_gap_seconds'] * fps))
        self.gate = cfg['gate_diagonal_fraction'] * math.hypot(width, height)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self.F = np.eye(4)
        self.F[0, 2] = self.F[1, 3] = self.dt
        self.reset()

    def reset(self):
        self.state = None
        self.P = np.diag([25., 25., 10000., 10000.])
        self.gap, self.size = 0, (1., 1.)

    def step(self, candidates, affine=None, cut=False):
        if cut:
            self.reset()
        c = self.cfg
        chosen = None
        if self.state is not None:
            if affine is not None:
                self.state[:2] = affine[:, :2] @ self.state[:2] + affine[:, 2]
                self.state[2:] = affine[:, :2] @ self.state[2:]
                transform = np.zeros((4, 4))
                transform[:2, :2] = transform[2:, 2:] = affine[:, :2]
                self.P = transform @ self.P @ transform.T
            self.state = self.F @ self.state
            dt = self.dt
            g = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
            self.P = self.F @ self.P @ self.F.T + c['process_noise'] * (g @ g.T)
            eligible = [(np.linalg.norm(d.center - self.state[:2]), d) for d in candidates]
            eligible = [(distance, d) for distance, d in eligible if distance <= self.gate * (1 + self.gap * 0.15)]
            if eligible:
                chosen = min(eligible, key=lambda pair: pair[0] / self.gate - 0.4 * pair[1].score)[1]
        elif candidates:
            best = max(candidates, key=lambda d: d.score)
            if best.score >= c['min_acquire_score']:
                chosen = best
        if chosen is not None:
            if self.state is None:
                self.state = np.r_[chosen.center, 0., 0.]
            else:
                R = np.eye(2) * c['measurement_noise']
                gain = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + R)
                self.state += gain @ (chosen.center - self.H @ self.state)
                a = np.eye(4) - gain @ self.H
                self.P = a @ self.P @ a.T + gain @ R @ gain.T
            self.size, self.gap = chosen.box[2:], 0
            # Observation geometry stays measured; filter only drives association/prediction.
            return 'observed', chosen.box, chosen.score
        if self.state is not None:
            self.gap += 1
            cx, cy = self.state[:2]
            if self.gap <= self.max_gap and 0 <= cx < self.width and 0 <= cy < self.height:
                w, h = self.size
                return 'predicted', (cx - w / 2, cy - h / 2, w, h), None
            self.reset()
        return 'missing', None, None
