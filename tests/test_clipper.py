import unittest

import cv2
import numpy as np

from shorts_generator.local.clipper import (
    _pick_active_face,
    _pick_locked_face,
    _update_face_activities,
)


class FaceLockTests(unittest.TestCase):
    def test_lock_does_not_jump_to_a_larger_distant_face(self):
        current = (90, 10, 40, 40)
        larger_distant = (400, 10, 100, 100)

        selected = _pick_locked_face(
            [larger_distant, current],
            locked_x=110,
            reacquire=False,
            max_distance=80,
        )

        self.assertEqual(selected, current)

    def test_reacquire_uses_largest_face(self):
        selected = _pick_locked_face(
            [(90, 10, 40, 40), (400, 10, 100, 100)],
            locked_x=None,
            reacquire=True,
            max_distance=80,
        )

        self.assertEqual(selected, (400, 10, 100, 100))

    def test_active_speaker_can_take_over_the_lock(self):
        old_speaker = (90, 10, 80, 80)
        active_speaker = (400, 10, 80, 80)

        selected = _pick_active_face(
            [old_speaker, active_speaker],
            [0.005, 0.025],
            locked_x=130,
            reacquire=False,
            max_distance=100,
        )

        self.assertEqual(selected, active_speaker)

    def test_small_motion_does_not_steal_the_lock(self):
        old_speaker = (90, 10, 80, 80)
        other_face = (400, 10, 80, 80)

        selected = _pick_active_face(
            [old_speaker, other_face],
            [0.005, 0.010],
            locked_x=130,
            reacquire=False,
            max_distance=100,
        )

        self.assertEqual(selected, old_speaker)

    def test_mouth_motion_registers_as_activity(self):
        face = (20, 20, 60, 60)
        tracks = []
        still = np.zeros((100, 100), dtype=np.uint8)
        moving_mouth = still.copy()
        moving_mouth[52:76, 31:69] = 255

        _update_face_activities(cv2, still, [face], tracks)
        activities = _update_face_activities(cv2, moving_mouth, [face], tracks)

        self.assertGreater(activities[0], 0.012)


if __name__ == "__main__":
    unittest.main()
