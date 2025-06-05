import nfl_data_py as nfl
import pandas as pd

# Step 1: Download play-by-play data for 2020–2023 seasons
seasons = [2020, 2021, 2022, 2023]
pbp = nfl.import_pbp_data(seasons)

# Step 2: Filter for offensive plays only (run or pass) and exclude no-plays (penalties, timeouts, etc.)
# Filter to run and pass plays only
offense = pbp[
    (pbp['play_type'].isin(['pass', 'run']))
].copy()

offense.dropna(subset=['down', 'ydstogo', 'yardline_100', 'yards_gained'], inplace=True)
offense = offense[offense['down'].between(1, 4)]
offense = offense[offense['yardline_100'].between(1, 99)]
offense = offense[offense['ydstogo'] > 0]

# Step 3: Keep only relevant columns
offense = offense[[
    'game_id', 'posteam', 'defteam', 'quarter_seconds_remaining',
    'qtr', 'down', 'ydstogo', 'yardline_100',
    'play_type', 'yards_gained', 'score_differential',
    'pass_length', 'run_location', 'epa', 'success'
]]

# Step 4: Clean data by dropping or filtering invalid entries
offense.dropna(subset=['down', 'ydstogo', 'yardline_100', 'yards_gained'], inplace=True)
offense = offense[offense['down'].between(1, 4)]
offense = offense[offense['yardline_100'].between(1, 99)]
offense = offense[offense['ydstogo'] > 0]

# Step 5: Reset index for clean storage
offense.reset_index(drop=True, inplace=True)

# Step 6: Save cleaned data to CSV for later use
offense.to_csv("processed_offensive_plays.csv", index=False)

# Optional: preview
print(offense.head())
