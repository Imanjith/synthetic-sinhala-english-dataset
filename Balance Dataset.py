import pandas as pd
import sys

def print_ratios(df, name):
    total = len(df)
    if total == 0:
        print(f"Dataset {name} is empty.")
        return

    gt_count = len(df[df['method'] == 'ground_truth'])
    others_count = total - gt_count

    gt_ratio = gt_count / total
    others_ratio = others_count / total

    print(f"--- {name} Statistics ---")
    print(f"Total rows: {total}")
    print(f"Ground Truth: {gt_count} ({gt_ratio:.2%})")
    print(f"Others: {others_count} ({others_ratio:.2%})")
    print(f"Ratio (GT : Others) -> 1 : {others_count / gt_count if gt_count > 0 else 'inf':.2f}")
    print("-" * 30)

def main():
    input_file = 'synthetic_hallucinations_exhaustives_final.csv'
    output_file = 'balanced_dataset.csv'

    try:
        print(f"Reading {input_file}...")
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        return

    # Print original stats
    print_ratios(df, "Original Dataset")

    # Split dataset
    df_gt = df[df['method'] == 'ground_truth']
    df_others = df[df['method'] != 'ground_truth']

    # Determine target count (limit to the smaller group size)
    min_count = min(len(df_gt), len(df_others))
    print(f"Balancing to {min_count} rows per group...")

    # Sample from both to be safe, though usually one is exact
    # 'random_state=42' for reproducibility
    df_gt_sampled = df_gt.sample(n=min_count, random_state=42) if len(df_gt) > min_count else df_gt
    df_others_sampled = df_others.sample(n=min_count, random_state=42)

    # Concatenate and shuffle
    balanced_df = pd.concat([df_gt_sampled, df_others_sampled])
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Print new stats
    print_ratios(balanced_df, "Balanced Dataset")

    # Save to CSV
    balanced_df.to_csv(output_file, index=False)
    print(f"\nSaved balanced dataset to {output_file}")

if __name__ == "__main__":
    main()
