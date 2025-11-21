"""Prepare non-IID hospital datasets from NIH Chest X-ray14 for federated learning."""

import os
import random

import numpy as np
import pandas as pd
from datasets import DatasetDict, concatenate_datasets, load_dataset
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split

RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

HF_DS = "BahaaEldin0/NIH-Chest-Xray-14"

# Target proportions: A=45%, B=31%, C=19%, D=5%
TARGETS = {"A": 0.45, "B": 0.31, "C": 0.19, "D": 0.04}
SPLIT_RATIOS = (0.8, 0.1, 0.1)  # train/eval/test (patient-wise)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTDIR = os.path.join(ROOT_DIR, "xray/raw")
os.makedirs(OUTDIR, exist_ok=True)

NO_FINDING = "No Finding"

# Label "preferences" to shape each silo (strong biases for non-IID)
PREF_A = {"Effusion": 8.0, "Edema": 8.0, "Atelectasis": 5.0}
PREF_B = {"Mass": 8.0, "Nodule": 8.0, "Pneumothorax": 6.0}
PREF_C = {"Hernia": 12.0, "Fibrosis": 8.0, "Emphysema": 8.0}


def patient_summary(df):
    """Create a summary table of patients with aggregated metadata."""
    rows = []
    for pid, g in df.groupby("Patient ID"):
        rows.append(
            {
                "id": pid,
                "n_images": len(g),
                "median_age": int(np.median(g["Patient Age"])),
                "view": list(g["View Position"])[0],
                "gender": list(g["Patient Gender"])[0],
                "labels_union": set([el for els in g["label"] for el in els]),
            }
        )
    return pd.DataFrame(rows)


def score_patient(row, silo):
    """Score a patient"""
    score = 0.0
    age = row["median_age"]
    view = row["view"]
    labs = row["labels_union"]
    gender = row["gender"]

    if silo == "A":
        # Portable inpatient: AP, older males, fluid-related
        if view == "PA":
            score -= 10000.0

        # Strong age preference for elderly
        if age >= 60:
            score += 8.0
        elif age < 40:
            score -= 100.0
        else:
            score += (age - 40) / 5.0  # gradual increase 40-60

        # Gender bias: more males
        if gender == "M":
            score += 5.0

        for lb, w in PREF_A.items():
            if lb in labs:
                score += w

    elif silo == "B":
        # Outpatient PA clinic: PA, younger females, nodules/mass
        if view == "AP":
            score -= 10000.0

        # Strong age preference for working age
        if 20 <= age <= 65:
            score += 10.0

        # Gender bias: more females
        if gender == "F":
            score += 5.0

        for lb, w in PREF_B.items():
            if lb in labs:
                score += w

    elif silo == "C":
        for lb, w in PREF_C.items():
            if lb in labs:
                score += w

        if labs == {NO_FINDING}:
            score -= 5.0

    elif silo == "D":
        # ICU / Critical Care: AP, age extremes, multi-morbidity
        if view == "AP":
            score += 8.0

        # Strong preference for age extremes ONLY
        if age < 25:
            score += 10.0
        elif age > 70:
            score += 10.0
        elif 30 <= age <= 65:
            score -= 15.0  # strong penalty for middle age

        # Multi-morbidity: ICU patients have multiple findings
        n_findings = len([lb for lb in labs if lb != NO_FINDING])
        if n_findings >= 2:
            score += 10.0
        elif n_findings == 1:
            score += 2.0

        # Rare/severe conditions
        for lb in ["Emphysema", "Fibrosis", "Hernia", "Pneumothorax"]:
            if lb in labs:
                score += 3.0

        # Exclude healthy patients (no one in ICU is "No Finding")
        if labs == {NO_FINDING}:
            score -= 10.0

    return score


def report_split_stats(name, patient_ids, df_images, ptab):
    """Generate comprehensive statistics for a data split."""
    # Filter patient summary for this split
    patients = ptab[ptab["id"].isin(patient_ids)]

    n_patients = len(patients)
    n_images = len(df_images)

    # Age statistics
    ages = patients["median_age"]
    age_stats = {
        "mean": ages.mean(),
        "median": ages.median(),
        "std": ages.std(),
        "min": ages.min(),
        "max": ages.max(),
    }

    # View position distribution
    view_counts = patients["view"].value_counts()
    view_pct = {v: 100 * count / n_patients for v, count in view_counts.items()}

    # Gender distribution
    gender_counts = patients["gender"].value_counts()

    # Label distribution - count patients with each label
    all_labels = set()
    for labels in patients["labels_union"]:
        all_labels.update(labels)

    label_counts = {}
    for label in all_labels:
        count = sum(1 for labels in patients["labels_union"] if label in labels)
        label_counts[label] = count

    # Sort and get top 5
    top_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    print(f"\n{'=' * 60}")
    print(f"{name}")
    print(f"{'=' * 60}")
    print(
        f"Patients: {n_patients:,} | Images: {n_images:,} ({n_images / n_patients:.1f} img/patient)"
    )
    print(
        f"\nAge: mean={age_stats['mean']:.1f} median={age_stats['median']:.0f} std={age_stats['std']:.1f} range=[{age_stats['min']}-{age_stats['max']}]"
    )
    print("\nView Position:")
    for view, pct in sorted(view_pct.items()):
        print(f"  {view}: {view_counts[view]:,} ({pct:.1f}%)")
    print("\nGender:")
    for gender, count in sorted(gender_counts.items()):
        print(f"  {gender}: {count:,} ({100 * count / n_patients:.1f}%)")
    print("\nTop 5 Labels (by patient count):")
    for label, count in top_labels:
        print(f"  {label}: {count:,} ({100 * count / n_patients:.1f}%)")


def main():
    """Main dataset preparation logic."""
    ds_dict = load_dataset(HF_DS)

    print("Building patient summary ...")
    # Keep images in the full dataset for later
    dsets_with_images = concatenate_datasets(
        [ds_dict["train"], ds_dict["valid"], ds_dict["test"]]
    )

    # Create metadata-only version for scoring
    dsets = concatenate_datasets(
        [
            ds_dict["train"].remove_columns(["image"]),
            ds_dict["valid"].remove_columns(["image"]),
            ds_dict["test"].remove_columns(["image"]),
        ]
    )

    df = dsets.to_pandas()

    # Remove invalid rows with age > 100 and track valid indices
    valid_mask = df["Patient Age"] <= 100
    valid_indices = valid_mask[valid_mask].index.tolist()
    df = df[valid_mask].reset_index(drop=True)

    # Also filter the dataset with images using indices (much faster than filter)
    dsets_with_images = dsets_with_images.select(valid_indices)

    print("Building patient summary ...")
    ptab = patient_summary(df)

    # Score, rank-normalize, and weight all patients for all silos
    print("Scoring patients for all silos ...")
    for silo in ["A", "B", "C", "D"]:
        score = ptab.apply(lambda r, s=silo: score_patient(r, s), axis=1)
        norm = rankdata(score) / len(ptab)  # [0,1]
        norm[norm < 0.2] = 1e-6
        ptab[f"score_{silo}"] = norm * TARGETS[silo]

    # Assign patients to silos probabilistically
    print("Assigning patients to silos probabilistically ...")
    rng = np.random.default_rng(RNG_SEED)
    assigned_silo = []
    for _, row in ptab.iterrows():
        weights = [row["score_A"], row["score_B"], row["score_C"], row["score_D"]]
        assigned = rng.choice(["A", "B", "C", "D"], p=np.array(weights) / sum(weights))
        assigned_silo.append(assigned)
    ptab["assigned_silo"] = assigned_silo

    # Report silo assignment and characteristics
    for silo in ["A", "B", "C", "D"]:
        silo_pids = set(ptab[ptab["assigned_silo"] == silo]["id"])
        silo_df = df[df["Patient ID"].isin(silo_pids)]
        report_split_stats(f"Silo {silo}", silo_pids, silo_df, ptab)

    # Split patients (not images!) into train/eval/test per silo
    print("\n" + "=" * 60)
    print("Creating hospital datasets ...")
    print("=" * 60)

    # Build all splits as datasets first
    splits = {}
    for silo in ["A", "B", "C"]:
        silo_pids = list(ptab[ptab["assigned_silo"] == silo]["id"])
        train_p, temp = train_test_split(
            silo_pids, test_size=0.2, random_state=RNG_SEED
        )
        eval_p, test_p = train_test_split(temp, test_size=0.5, random_state=RNG_SEED)

        # Create datasets for each split
        for split_type, pids in [
            ("train", train_p),
            ("eval", eval_p),
            ("test", test_p),
        ]:
            split_mask = df["Patient ID"].isin(pids)
            split_indices = df[split_mask].index.tolist()
            splits[f"{silo}_{split_type}"] = dsets_with_images.select(split_indices)
            print(
                f"  Created split {silo}_{split_type}: {len(pids)} patients, {len(splits[f'{silo}_{split_type}'])} images"
            )

    # Silo D is OOD test only
    d_pids = list(ptab[ptab["assigned_silo"] == "D"]["id"])
    d_mask = df["Patient ID"].isin(d_pids)
    d_indices = df[d_mask].index.tolist()
    splits["D_test"] = dsets_with_images.select(d_indices)
    print(
        f"  Created split D_test: {len(d_pids)} patients, {len(splits['D_test'])} images"
    )

    # Create and save Hospital datasets
    print("\n" + "=" * 60)
    print("Saving hospital datasets ...")
    print("=" * 60)

    # Hospital A
    hospital_a = DatasetDict({"train": splits["A_train"], "eval": splits["A_eval"]})
    hospital_a.save_to_disk(os.path.join(OUTDIR, "HospitalA"))
    print(
        f"✓ HospitalA saved: train={len(hospital_a['train'])}, eval={len(hospital_a['eval'])}"
    )

    # Hospital B
    hospital_b = DatasetDict({"train": splits["B_train"], "eval": splits["B_eval"]})
    hospital_b.save_to_disk(os.path.join(OUTDIR, "HospitalB"))
    print(
        f"✓ HospitalB saved: train={len(hospital_b['train'])}, eval={len(hospital_b['eval'])}"
    )

    # Hospital C
    hospital_c = DatasetDict({"train": splits["C_train"], "eval": splits["C_eval"]})
    hospital_c.save_to_disk(os.path.join(OUTDIR, "HospitalC"))
    print(
        f"✓ HospitalC saved: train={len(hospital_c['train'])}, eval={len(hospital_c['eval'])}"
    )

    # Test dataset
    test = DatasetDict(
        {
            "test_A": splits["A_test"],
            "test_B": splits["B_test"],
            "test_C": splits["C_test"],
            "test_D": splits["D_test"],
        }
    )
    test.save_to_disk(os.path.join(OUTDIR, "Test"))
    print(
        f"✓ Test saved: test_A={len(test['test_A'])}, test_B={len(test['test_B'])}, test_C={len(test['test_C'])}, test_D={len(test['test_D'])}"
    )
    print("\n✓ Done! Hospital datasets created at:", os.path.abspath(OUTDIR))


if __name__ == "__main__":
    main()
