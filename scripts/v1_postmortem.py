"""Post-mortem of the FIRST full run (archived in results/archive_v1_recording_split).

That run had two pipeline bugs, both fixed in pipeline v2 (see PIPELINE_VERSION):
  1. The split was by recording id, not by person. Sleep-EDF cassette has two
     nights per person (ids SC4ssN: ss = person, N = night), so a person's
     other night could be in training.
  2. The intended 30-minute crop around sleep was a no-op (Wake annotations
     were counted as sleep), so Wake was ~69% of test epochs.

This script quantifies both from the archived predictions, without retraining.
It is kept as a record of what the bugs did; v2 results do not need it.

Usage: python scripts/v1_postmortem.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src.config import SEED
from src.data import person_id

ARCHIVE = Path(__file__).resolve().parent.parent / "results" / "archive_v1_recording_split"
SEEDS = (42, 43, 44)
MARGIN = 60  # 30 min of 30 s epochs

# Cassette recording list (PhysioNet RECORDS): 153 recordings, 78 people.
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


def load(model, seed):
    return np.load(ARCHIVE / f"{model}_seed{seed}_preds.npz")


def v1_split(ids):
    """The original (buggy) split: by recording id."""
    train, temp = train_test_split(sorted(ids), train_size=0.6, random_state=SEED)
    val, test = train_test_split(temp, train_size=0.5, random_state=SEED)
    return train, val, test


def crop_mask(labels, subject_ids):
    keep = np.zeros(len(labels), dtype=bool)
    for s in np.unique(subject_ids):
        idx = np.where(subject_ids == s)[0]
        sleep = np.where(labels[idx] != 0)[0]
        if len(sleep):
            keep[idx[max(0, sleep[0] - MARGIN): min(len(idx), sleep[-1] + MARGIN + 1)]] = True
    return keep


train, val, test = v1_split(ALL_IDS)
saved_test = sorted(set(load("cnn", SEEDS[0])["subject_ids"].tolist()))
assert sorted(test) == saved_test, "id list does not reproduce the archived test split"

# ---- 1. person leakage
train_people = {person_id(s) for s in train}
leaked = {s for s in test if person_id(s) in train_people}
clean = set(test) - leaked
print(f"{len(ALL_IDS)} recordings from {len({person_id(s) for s in ALL_IDS})} people")
print(f"test recordings whose person also has a TRAIN recording: {len(leaked)} / {len(test)}\n")

for model in ["cnn", "transformer"]:
    f1_l, f1_c, per_rec = [], [], {}
    for seed in SEEDS:
        d = load(model, seed)
        sid, y, p = d["subject_ids"], d["labels"], d["preds"]
        a, b = np.isin(sid, list(leaked)), np.isin(sid, list(clean))
        f1_l.append(f1_score(y[a], p[a], average="macro"))
        f1_c.append(f1_score(y[b], p[b], average="macro"))
        for s in np.unique(sid):
            i = sid == s
            per_rec.setdefault(s, []).append(f1_score(y[i], p[i], average="macro"))
    L = [np.mean(per_rec[s]) for s in leaked]
    C = [np.mean(per_rec[s]) for s in clean]
    pval = stats.ttest_ind(L, C, equal_var=False).pvalue
    print(f"{model:12s} macro F1  leaked-person {np.mean(f1_l):.3f}  clean {np.mean(f1_c):.3f}  (Welch p = {pval:.2f})")

# ---- 2. Wake-dominated test set
print()
d0 = load("cnn", SEEDS[0])
keep0 = crop_mask(d0["labels"], d0["subject_ids"])
print(f"Wake share of test epochs: {100 * (d0['labels'] == 0).mean():.1f}% as run, "
      f"{100 * (d0['labels'][keep0] == 0).mean():.1f}% after cropping (kept {100 * keep0.mean():.1f}% of epochs)")
for model in ["cnn", "transformer"]:
    acc, f1 = [], []
    for seed in SEEDS:
        d = load(model, seed)
        k = crop_mask(d["labels"], d["subject_ids"])
        acc.append((accuracy_score(d["labels"], d["preds"]), accuracy_score(d["labels"][k], d["preds"][k])))
        f1.append((f1_score(d["labels"], d["preds"], average="macro"), f1_score(d["labels"][k], d["preds"][k], average="macro")))
    acc, f1 = np.mean(acc, 0), np.mean(f1, 0)
    print(f"{model:12s} accuracy {acc[0]:.3f} -> {acc[1]:.3f} | macro F1 {f1[0]:.3f} -> {f1[1]:.3f}")
