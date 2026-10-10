"""Regression tests for the two pipeline bugs found after the first full run.

Run with:  python -m unittest discover -s tests -v
(The real-data test is skipped when data/sleep-cassette has no files.)
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.config import CASSETTE_DIR
from src.data import (
    assert_no_person_overlap,
    compute_crop_window,
    discover_subjects,
    get_subject_split,
    load_subject_epochs,
    person_id,
)

# Full Sleep-EDF cassette recording list (PhysioNet RECORDS): 153 recordings, 78 people.
ALL_IDS = """
4001 4002 4011 4012 4021 4022 4031 4032 4041 4042 4051 4052 4061 4062 4071 4072 4081 4082 4091 4092
4101 4102 4111 4112 4121 4122 4131 4141 4142 4151 4152 4161 4162 4171 4172 4181 4182 4191 4192
4201 4202 4211 4212 4221 4222 4231 4232 4241 4242 4251 4252 4261 4262 4271 4272 4281 4282 4291 4292
4301 4302 4311 4312 4321 4322 4331 4332 4341 4342 4351 4352 4362 4371 4372 4381 4382
4401 4402 4411 4412 4421 4422 4431 4432 4441 4442 4451 4452 4461 4462 4471 4472 4481 4482 4491 4492
4501 4502 4511 4512 4522 4531 4532 4541 4542 4551 4552 4561 4562 4571 4572 4581 4582 4591 4592
4601 4602 4611 4612 4621 4622 4631 4632 4641 4642 4651 4652 4661 4662 4671 4672
4701 4702 4711 4712 4721 4722 4731 4732 4741 4742 4751 4752 4761 4762 4771 4772
4801 4802 4811 4812 4821 4822
""".split()


class CropWindow(unittest.TestCase):
    # 10 h recording: 6 h Wake, 1 h of sleep stages, 3 h Wake (30 s annotations merged).
    DESC = ["Sleep stage W", "Sleep stage 1", "Sleep stage 2", "Sleep stage R", "Sleep stage W"]
    ONSET = [0.0, 21600.0, 22000.0, 24000.0, 25200.0]
    DUR = [21600.0, 400.0, 2000.0, 1200.0, 10800.0]

    def test_wake_does_not_define_sleep_period(self):
        # The original bug: leading Wake counted as "sleep", so tmin was 0 and nothing was cropped.
        tmin, tmax = compute_crop_window(self.DESC, self.ONSET, self.DUR, recording_end=36000.0)
        self.assertEqual(tmin, 21600.0 - 1800.0)
        self.assertEqual(tmax, 25200.0 + 1800.0)

    def test_window_is_clamped_to_recording(self):
        tmin, tmax = compute_crop_window(self.DESC, self.ONSET, self.DUR, recording_end=26000.0)
        self.assertEqual(tmax, 26000.0)
        tmin, _ = compute_crop_window(self.DESC, [0.0, 600.0, 900.0, 1500.0, 2000.0], self.DUR, 36000.0)
        self.assertEqual(tmin, 0.0)

    def test_no_sleep_means_no_crop(self):
        self.assertIsNone(compute_crop_window(["Sleep stage W"], [0.0], [100.0], 100.0))

    def test_ignores_movement_and_unknown(self):
        window = compute_crop_window(
            ["Sleep stage W", "Movement time", "Sleep stage ?", "Sleep stage 2"],
            [0.0, 5000.0, 5100.0, 9000.0],
            [5000.0, 30.0, 30.0, 600.0],
            recording_end=20000.0,
        )
        self.assertEqual(window, (9000.0 - 1800.0, 9600.0 + 1800.0))


class PersonSplit(unittest.TestCase):
    def test_same_person_never_in_two_splits(self):
        train, val, test = get_subject_split(ALL_IDS)
        assert_no_person_overlap(train, val, test)  # raises on leakage

    def test_covers_every_recording_exactly_once(self):
        splits = get_subject_split(ALL_IDS)
        flat = [s for part in splits for s in part]
        self.assertEqual(sorted(flat), sorted(ALL_IDS))

    def test_roughly_60_20_20_by_person(self):
        train, val, test = get_subject_split(ALL_IDS)
        n_people = {person_id(s) for s in ALL_IDS}
        self.assertEqual(len(n_people), 78)
        sizes = [len({person_id(s) for s in part}) for part in (train, val, test)]
        self.assertEqual(sum(sizes), 78)
        self.assertAlmostEqual(sizes[0] / 78, 0.6, delta=0.03)
        self.assertAlmostEqual(sizes[1] / 78, 0.2, delta=0.03)

    def test_deterministic(self):
        self.assertEqual(get_subject_split(ALL_IDS), get_subject_split(list(ALL_IDS)))

    def test_leakage_detector_fires(self):
        with self.assertRaises(AssertionError):
            assert_no_person_overlap(["4581"], [], ["4582"])  # two nights, one person


@unittest.skipUnless(list(CASSETTE_DIR.glob("SC*-PSG.edf")), "no local Sleep-EDF data")
class RealRecording(unittest.TestCase):
    def test_crop_actually_trims_wake(self):
        subject = discover_subjects(CASSETTE_DIR)[0]
        epochs, labels = load_subject_epochs(subject)
        self.assertEqual(epochs.shape[1], 3000)
        wake_share = float(np.mean(labels == 0))
        self.assertLess(wake_share, 0.5, f"Wake share {wake_share:.2f}: crop is not trimming")
        sleep_idx = np.where(labels != 0)[0]
        # at most ~30 min (60 epochs) of leading and trailing Wake around sleep
        self.assertLessEqual(sleep_idx[0], 61)
        self.assertLessEqual(len(labels) - 1 - sleep_idx[-1], 61)

    def test_cache_roundtrip(self):
        import tempfile

        subject = discover_subjects(CASSETTE_DIR)[0]
        with tempfile.TemporaryDirectory() as d:
            e1, l1 = load_subject_epochs(subject, cache_dir=Path(d))
            self.assertTrue(list(Path(d).glob("*.npz")))
            e2, l2 = load_subject_epochs(subject, cache_dir=Path(d))
        np.testing.assert_array_equal(e1, e2)
        np.testing.assert_array_equal(l1, l2)


if __name__ == "__main__":
    unittest.main()
