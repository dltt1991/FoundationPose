import unittest
import threading
from stage_profile import StageProfile, stage


class StageProfileTests(unittest.TestCase):
    def test_unrelated_thread_does_not_inherit_active_profile(self):
        ticks = iter([0., 1.])
        p = StageProfile(clock=lambda: next(ticks))

        def worker():
            with stage('other-thread'):
                pass

        with p:
            thread = threading.Thread(target=worker)
            thread.start()
            thread.join()
            with stage('main-thread'):
                pass
        self.assertEqual(set(p.report()['stages']), {'main-thread'})

    def test_nested_time_is_not_double_counted(self):
        ticks = iter([0., 1., 3., 5.])
        syncs = []
        p = StageProfile(synchronize=lambda: syncs.append(1), clock=lambda: next(ticks))
        with p:
            with stage('register'):
                with stage('network'):
                    pass
        report = p.report()
        self.assertEqual(report['stages']['register']['inclusive_seconds'], 5.)
        self.assertEqual(report['stages']['register']['exclusive_seconds'], 3.)
        self.assertEqual(report['stages']['register/network']['inclusive_seconds'], 2.)
        self.assertEqual(len(syncs), 4)

    def test_disabled_profile_does_not_synchronize(self):
        with stage('disabled'):
            pass
        p = StageProfile(synchronize=lambda: self.fail('unexpected synchronization'))
        self.assertEqual(p.report()['stages'], {})

    def test_repeated_stages_aggregate_and_exceptions_restore_context(self):
        ticks = iter([0., 1., 2., 4.])
        p = StageProfile(clock=lambda: next(ticks))
        with p:
            with stage('render'):
                pass
            with self.assertRaises(ValueError):
                with stage('render'):
                    raise ValueError('render failed')
        item = p.report()['stages']['render']
        self.assertEqual(item['calls'], 2)
        self.assertEqual(item['inclusive_seconds'], 3.)
        self.assertEqual(item['failed_calls'], 1)
        with stage('outside'):
            pass
        self.assertNotIn('outside', p.report()['stages'])


if __name__ == '__main__':
    unittest.main()
