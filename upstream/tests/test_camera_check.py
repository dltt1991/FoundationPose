import unittest
from check_realsense import summarize_frame_timing, needs_usb_permission_hint


class CameraCheckTests(unittest.TestCase):
    def test_repeated_and_skipped_stream_frames_are_reported_separately(self):
        info = summarize_frame_timing([[10, 20], [10, 21], [11, 21], [13, 22]], [0, .1, .2, .3])
        self.assertEqual(info['color']['duplicate_transitions'], 1)
        self.assertEqual(info['color']['skipped_frame_numbers'], 1)
        self.assertEqual(info['depth']['duplicate_transitions'], 1)
        self.assertAlmostEqual(info['color']['fresh_fps'], 2/.3)
        self.assertAlmostEqual(info['frameset_fps'], 3/.3)

    def test_backwards_frame_number_is_preserved_as_diagnostic(self):
        info = summarize_frame_timing([[10, 20], [9, 21], [11, 22]], [0, .1, .2])
        self.assertEqual(info['color']['backward_transitions'], 1)

    def test_permission_hint_is_not_shown_after_successful_start(self):
        self.assertFalse(needs_usb_permission_hint(True, 'Camera frame numbers are not advancing monotonically.'))
        self.assertFalse(needs_usb_permission_hint(False, 'Frames did not arrive within 10000'))
        self.assertTrue(needs_usb_permission_hint(False, 'No device connected'))


if __name__ == '__main__':
    unittest.main()
