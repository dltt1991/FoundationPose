import threading
import unittest
from live_pipeline import LivePipeline, LatestSlot


class PipelineTests(unittest.TestCase):
    def test_replay_preserves_original_time_instead_of_accelerating_motion(self):
        from live_runner import replay_offset
        first = '1581120424100262102'
        second = str(int(first)+200_000_000)
        self.assertAlmostEqual(replay_offset(second, first, 1, 0), .2)
        self.assertAlmostEqual(replay_offset(second, first, 1, 30), 1/30)

    def test_latest_slot_overwrites_backlog(self):
        slot = LatestSlot()
        for value in range(100):
            slot.publish(value)
        version, value = slot.wait(0)
        self.assertEqual((version, value), (100, 99))
        slot.close()
        self.assertIsNone(slot.wait(version))

    def test_capture_continues_during_slow_initialization(self):
        captured = threading.Event()
        seen = []

        def source():
            for value in range(100):
                yield value
            captured.set()

        def infer(value):
            if value == -1:
                self.assertTrue(captured.wait(2), 'Capture blocked behind inference')
            seen.append(value)
            return value * 2

        pipeline = LivePipeline(source(), infer, initial=-1).start()
        pipeline.join()
        pipeline.raise_if_failed()
        self.assertEqual(seen, [-1, 99])
        _, result = pipeline.results.wait(0)
        self.assertEqual(result.value, 198)
        self.assertGreater(pipeline.stats()['dropped'], 0)

    def test_worker_failure_reaches_main_thread(self):
        def fail(value):
            raise ValueError('bad depth')

        pipeline = LivePipeline(iter([1]), fail, initial=0).start()
        pipeline.join()
        with self.assertRaisesRegex(ValueError, 'bad depth'):
            pipeline.raise_if_failed()


if __name__ == '__main__':
    unittest.main()
