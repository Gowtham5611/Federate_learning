import json
from pathlib import Path

def inspect_metadata(dataset_name):
    meta_dir = Path("data/processed/ibm_aml") / dataset_name / "metadata"
    print(f"\n=================================================================")
    print(f"DATASET METADATA AUDIT: {dataset_name}")
    print(f"=================================================================")
    
    splits_p = meta_dir / "splits.json"
    if splits_p.exists():
        s_data = json.loads(splits_p.read_text(encoding="utf-8"))
        print(f"Splits Metadata ({splits_p.name}):")
        print("  Target Column:", s_data.get("target"))
        print("  Chronology Holds:", s_data.get("chronology_holds"))
        print("  Boundaries:", s_data.get("boundaries"))
        print("  Train Last Timestamp:", s_data.get("train_last_timestamp"))
        print("  Val Last Timestamp:", s_data.get("validation_last_timestamp"))
        for s in s_data.get("per_split", []):
            print(f"    Split: {s['name']}")
            print(f"      Rows: {s['rows']:,}")
            print(f"      Positive (Laundering): {s['positive']:,}")
            print(f"      Negative (Legitimate): {s['negative']:,}")
            print(f"      Positive Rate: {s['positive_rate']:.8f}")
            print(f"      Share: {s['share']:.6f}")
            print(f"      Time Window: {s['first_timestamp']} -> {s['last_timestamp']}")
            
    feat_p = meta_dir / "feature_metadata.json"
    if feat_p.exists():
        f_data = json.loads(feat_p.read_text(encoding="utf-8"))
        cols = f_data.get("columns", [])
        num_features = len(cols)
        print(f"\nFeature Metadata ({feat_p.name}):")
        print(f"  Feature Count: {num_features}")
        print(f"  Sample Features: {[c['name'] for c in cols[:8]]}")
        print(f"  Last Features: {[c['name'] for c in cols[-5:]]}")

def main():
    inspect_metadata("HI-Small")
    inspect_metadata("LI-Medium")

if __name__ == "__main__":
    main()

