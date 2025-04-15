import json
import os
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATASETS = {
    "GoEmotions": "data/processed/goemotions_cleaned.json",
    "Positive Reddit": "data/processed/positive/reddit_positive_scored.json",
    "Negative Reddit": "data/processed/labeled_negative/reddit_cleaned.json"
}

GRAPH_DIR = "graphs"
os.makedirs(GRAPH_DIR, exist_ok=True)

def analyze_and_plot_labels(dataset_path, dataset_name):
    with open(dataset_path, "r") as f:
        data = json.load(f)

    label_counter = Counter()

    for entry in data:
        labels = entry.get("labels", [])
        for label in labels:
            label_counter[label] += 1

    df = pd.DataFrame(label_counter.items(), columns=["Label", "Count"])
    df = df.sort_values(by="Count", ascending=False)
    df["Percentage"] = 100 * df["Count"] / df["Count"].sum()

    df.to_csv(f"{GRAPH_DIR}/{dataset_name.lower().replace(' ', '_')}_labels.csv", index=False)

    # Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Count", y="Label", data=df, palette="viridis")
    plt.title(f"{dataset_name} - Label Frequency")
    plt.xlabel("Count")
    plt.ylabel("Label")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, f"{dataset_name.lower().replace(' ', '_')}_labels.png"))
    plt.close()

    print(f"[✓] Processed: {dataset_name}")
    return df


def main():
    all_dataframes = []
    for name, path in DATASETS.items():
        df = analyze_and_plot_labels(path, name)
        all_dataframes.append(df.assign(Source=name))

    combined_df = pd.concat(all_dataframes)
    combined_df.to_csv(f"{GRAPH_DIR}/label_distribution_summary.csv", index=False)
    print("\nAll graphs and summaries saved to:", GRAPH_DIR)

if __name__ == "__main__":
    main()
