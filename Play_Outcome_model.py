# play_outcome_model.py

import pandas as pd
import numpy as np
from collections import defaultdict

class PlayOutcomeModel:
    def __init__(self, csv_path="processed_offensive_plays.csv"):
        self.df = pd.read_csv(csv_path)

        # Round yardline and ydstogo for binning
        self.df["yardline_bin"] = (self.df["yardline_100"] // 10) * 10
        self.df["ydstogo_bin"] = (self.df["ydstogo"] // 2) * 2

        # Index the data by simplified keys
        self.play_map = defaultdict(list)
        for _, row in self.df.iterrows():
            key = (row["down"], row["ydstogo_bin"], row["yardline_bin"], row["play_type"])
            self.play_map[key].append(row["yards_gained"])

    def sample_yards(self, down, ydstogo, yardline_100, play_type):
        key = (
            down,
            int(ydstogo // 2) * 2,
            int(yardline_100 // 10) * 10,
            play_type
        )

        # Try exact match first
        samples = self.play_map.get(key, [])

        # Loosen condition if no match
        if len(samples) < 3:
            key_loose = (down, int(ydstogo // 2) * 2, None, play_type)
            samples = [
                row["yards_gained"]
                for _, row in self.df.iterrows()
                if row["down"] == down and row["ydstogo_bin"] == int(ydstogo // 2) * 2 and row["play_type"] == play_type
            ]

        # Fallback to small random value if nothing works
        if len(samples) == 0:
            return np.random.randint(-2, 3)

        return int(np.random.choice(samples))
