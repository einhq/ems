"""The face models' geometry: the five-point alignment and the head pose read from the points.

Needs numpy and OpenCV, so it runs in the container and skips elsewhere."""

import math
import os
import sys

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ems", "modelworker"))

import face_models  # noqa: E402

TEMPLATE = face_models.ARCFACE_TEMPLATE


def _turned(points, angle_deg, scale, shift):
    a = math.radians(angle_deg)
    rotation = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return np.asarray(points) @ rotation.T * scale + shift


def test_the_template_aligns_onto_itself():
    matrix = face_models.similarity(TEMPLATE, TEMPLATE)
    assert np.allclose(matrix, [[1, 0, 0], [0, 1, 0]], atol=1e-9)


def test_a_turned_scaled_and_shifted_face_aligns_back_onto_the_template():
    points = _turned(TEMPLATE, 30, 2.5, [400, 120])
    matrix = face_models.similarity(points, TEMPLATE)
    back = points @ matrix[:, :2].T + matrix[:, 2]
    assert np.allclose(back, TEMPLATE, atol=1e-9)


def test_points_with_no_spread_do_not_align():
    assert face_models.similarity([[5, 5]] * 5, TEMPLATE) is None


def test_the_aligned_crop_is_112_pixels_square():
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    assert face_models.align(cv2, img, _turned(TEMPLATE, -10, 3, [300, 200])).shape == (112, 112, 3)


def test_a_face_level_with_the_camera_has_no_turn_or_tilt():
    # The template's nose sits 0.5 of the way down from the eyes to the mouth: neither raised nor
    # lowered.
    pose = face_models.pose(TEMPLATE)
    assert abs(pose["yaw"]) < 1 and abs(pose["roll"]) < 1 and abs(pose["pitch"]) < 2


def test_a_tilted_face_reads_as_roll_alone():
    pose = face_models.pose(_turned(TEMPLATE, 15, 4, [200, 200]))
    assert pose["roll"] == pytest.approx(15 + face_models.pose(TEMPLATE)["roll"], abs=1e-6)
    assert abs(pose["yaw"]) < 1


def test_a_nose_towards_the_pictures_left_is_a_negative_yaw():
    points = TEMPLATE.copy()
    points[2, 0] -= 10
    assert face_models.pose(points)["yaw"] < -20


def test_a_nose_low_towards_the_mouth_is_a_negative_pitch():
    points = TEMPLATE.copy()
    points[2, 1] += 10
    assert face_models.pose(points)["pitch"] < -10


def test_eyes_on_one_spot_give_no_pose():
    points = TEMPLATE.copy()
    points[1] = points[0]
    assert face_models.pose(points) is None
