import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from wordcloud import WordCloud
import os
from datetime import datetime
import numpy as np
from pandas import DataFrame
from typing import Any



with open("data/processed/goemotions_cleaned.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)
df = df[df['labels'].apply(lambda x: isinstance(x, list) and len(x) > 0)]

# Now that df exists, fit the MultiLabelBinarizer
mlb = MultiLabelBinarizer()
binary_matrix: np.ndarray = mlb.fit_transform(df["labels"])
columns: list[str] = list(mlb.classes_)

binary_df: DataFrame = pd.DataFrame(binary_matrix, columns=columns)
correlation_matrix: DataFrame = binary_df.corr()

# --- Setup ---
sns.set(style="whitegrid")
os.makedirs("reports", exist_ok=True)

# --- GoEmotions Analysis ---
with open("data/processed/goemotions_cleaned.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)
df = df[df['labels'].apply(lambda x: isinstance(x, list) and len(x) > 0)]

with open("data/processed/reddit_scored.json", "r") as f:
    reddit = json.load(f)

reddit_df = pd.DataFrame([{
    "text": post["text"],
    "subreddit": post.get("subreddit", "unknown"),
    "length": len(post["text"].split()),
    "anxiety_score": post.get("anxiety_score", 0),

} for post in reddit if "anxiety_score" in post])


# --- Plot 1: Emotion Frequency ---
all_labels = [label for sublist in df['labels'] for label in sublist]
label_counts = Counter(all_labels)

plt.figure(figsize=(12, 6))
sns.barplot(x=list(label_counts.keys()), y=list(label_counts.values()))
plt.xticks(rotation=45)
plt.title("GoEmotions: Emotion Frequency")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("reports/emotion_frequency.png")
plt.show()

# --- Plot 2: Remade Emotion Co-occurrence Heatmap ---
plt.figure(figsize=(14, 12))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Upper triangle mask
sns.heatmap(
    correlation_matrix,
    mask=mask,
    cmap="RdBu_r",
    center=0,
    linewidths=0.5,
    square=True,
    cbar_kws={"shrink": 0.8},
    annot=True,
    fmt=".2f",
    annot_kws={"size": 7},
)
plt.title("GoEmotions: Emotion Co-occurrence Correlation (Improved)")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("reports/emotion_correlation_cleaned.png")
plt.show()


# --- Plot 3: Quadrant-Style, Less Cluttered ---
filtered = reddit_df[
    (reddit_df["length"] <= 1500) &
    (reddit_df["anxiety_score"] <= 100) &
    (reddit_df["anxiety_score"] > 0)
]

median_len = filtered["length"].median()
median_anx = filtered["anxiety_score"].median()

plt.figure(figsize=(12, 8))

# Add density-based background using hexbin
plt.hexbin(filtered["length"], filtered["anxiety_score"], gridsize=50, cmap="Blues", mincnt=1)

# Overlay jittered scatterplot
sns.stripplot(
    data=filtered,
    x="length",
    y="anxiety_score",
    color="black",
    alpha=0.2,
    jitter=0.25,
    size=2
)

# Quadrant lines
plt.axhline(median_anx, color="gray", linestyle="--", linewidth=1)
plt.axvline(median_len, color="gray", linestyle="--", linewidth=1)

# Quadrant labels
plt.text(median_len + 50, median_anx + 5, "Long + High Anxiety", fontsize=10, color="crimson")
plt.text(median_len - 350, median_anx + 5, "Short + High Anxiety", fontsize=10, color="purple")
plt.text(median_len - 350, median_anx - 10, "Short + Low Anxiety", fontsize=10, color="steelblue")
plt.text(median_len + 50, median_anx - 10, "Long + Low Anxiety", fontsize=10, color="green")

# Labels and layout
plt.title("Anxiety Score vs Post Length (Quadrant View + Density)")
plt.xlabel("Post Length (Words)")
plt.ylabel("Anxiety Score")
plt.grid(True)
plt.tight_layout()
plt.savefig("reports/anxiety_quadrant_density.png")
plt.show()

# --- Plot 3.5: Bubble Plot for Anxiety Score vs Post Length (Bubble Size = Frequency) ---
# Group by length and anxiety score to count frequency of each combination
bubble_data = filtered.groupby(["length", "anxiety_score"]).size().reset_index(name="frequency")

# Normalize frequency for bubble size (bubbles won't be too large)
max_freq = bubble_data["frequency"].max()
bubble_data["size"] = bubble_data["frequency"] / max_freq * 500  # Scale to get a good bubble size

plt.figure(figsize=(12, 8))

# Plot the bubble plot with color representing anxiety score, and size representing frequency
sns.scatterplot(
    data=bubble_data,
    x="length",
    y="anxiety_score",
    size="size",
    hue="anxiety_score",
    sizes=(20, 200),  # Controls min/max bubble size
    palette="coolwarm",  # Color map
    alpha=0.7,
    legend=False
)

# Add labels and styling
plt.title("Anxiety Score vs Post Length (Bubble Plot - Size: Post Frequency)")
plt.xlabel("Post Length (Words)")
plt.ylabel("Anxiety Score")
plt.grid(True)
plt.tight_layout()
plt.savefig("reports/anxiety_bubble_plot.png")
plt.show()


# --- Fixed Plot 4: Subreddit vs Average Anxiety Score ---
avg_anxiety = reddit_df.groupby("subreddit")["anxiety_score"].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(data=avg_anxiety, x="subreddit", y="anxiety_score", color="skyblue")
plt.title("Average Anxiety Score by Subreddit")
plt.ylabel("Average Anxiety Score")
plt.xlabel("Subreddit")
plt.tight_layout()
plt.savefig("reports/subreddit_vs_anxiety.png")
plt.show()



# --- Plot 5: Word Cloud for High Anxiety Posts ---
high_anx = reddit_df[reddit_df["anxiety_score"] > 0.7]
all_text = " ".join(high_anx["text"].values)
wc = WordCloud(width=800, height=400, background_color="white").generate(all_text)

plt.figure(figsize=(12, 6))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("Top Words in High Anxiety Posts")
plt.tight_layout()
plt.savefig("reports/wordcloud_high_anxiety.png")
plt.show()

possible_emotions = ["anxiety_score", "anger_score", "sadness_score"]
emotion_cols = [col for col in possible_emotions if col in reddit_df.columns]

# --- Plot 6: Distribution of Emotion Scores ---
plt.figure(figsize=(10, 6))
for col in emotion_cols:
    sns.kdeplot(reddit_df[col], label=col)
plt.title("Distribution of Emotion Scores")
plt.xlabel("Score")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("reports/emotion_score_distributions.png")
plt.show()

# --- Plot 7: Emotion Intensity by Hour ---
if "hour" in reddit_df.columns:
    hourly_avg = reddit_df.groupby("hour")[emotion_cols].mean().reset_index().melt(id_vars="hour")

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=hourly_avg, x="hour", y="value", hue="variable", marker="o")
    plt.title("Emotion Scores by Hour of Day")
    plt.xlabel("Hour (UTC)")
    plt.ylabel("Average Score")
    plt.tight_layout()
    plt.savefig("reports/hourly_emotion_trends.png")
    plt.show()
