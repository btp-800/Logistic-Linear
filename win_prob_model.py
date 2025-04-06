import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from typing import Tuple

def update_csv_with_new_data(csv_file: str, new_data: pd.DataFrame) -> None:
  # Appends new daily data to the existing CSV
  existing_data = pd.read_csv(csv_file)
  updated_data = pd.concat([existing_data, new_data], ignore_index=True)
  updated_data.to_csv(csv_file, index=False)

def train_models(csv_file: str) -> Tuple[LogisticRegression, LinearRegression, LinearRegression]:
  data = pd.read_csv(csv_file)

  # Basic feature engineering
  # Label: home_win = 1 if home_score > away_score else 0
  data['home_win'] = (data['home_score'] > data['away_score']).astype(int)
  data['baseruns_diff'] = data['home_baseruns'] - data['away_baseruns']
  data['score_diff'] = data['home_score'] - data['away_score']
  data['total_runs'] = data['home_score'] + data['away_score']

  # Drop rows with missing or suspicious data
  data = data.dropna(subset=['home_baseruns','away_baseruns','home_score','away_score'])

  # Logistic Regression for win probability
  lr_win = LogisticRegression()
  X_lr = data[['baseruns_diff']]
  y_lr = data['home_win']
  lr_win.fit(X_lr, y_lr)

  # Linear Regression for run differential
  lr_diff = LinearRegression()
  X_diff = data[['baseruns_diff']]
  y_diff = data['score_diff']
  lr_diff.fit(X_diff, y_diff)

  # Linear Regression for total runs
  lr_total = LinearRegression()
  X_total = data[['home_baseruns','away_baseruns']]  # or use baseruns_diff, etc.
  y_total = data['total_runs']
  lr_total.fit(X_total, y_total)

  return lr_win, lr_diff, lr_total

def probability_to_moneyline(p: float) -> float:
  if p == 0:
    return 9999
  if p == 1:
    return -9999
  if p >= 0.5:
    return -100 * (p / (1 - p))
  return 100 * ((1 - p) / p)

def compute_fair_odds(
  lr_win: LogisticRegression,
  lr_diff: LinearRegression,
  lr_total: LinearRegression,
  home_baseruns: float,
  away_baseruns: float
) -> dict:
  prob = lr_win.predict_proba([[home_baseruns - away_baseruns]])[0, 1]
  moneyline = probability_to_moneyline(prob)

  run_diff = lr_diff.predict([[home_baseruns - away_baseruns]])[0]
  total_pred = lr_total.predict([[home_baseruns, away_baseruns]])[0]

  # Runline typically set around the predicted difference
  # This is a simplistic approach; real lines often use half-runs, etc.
  runline = round(run_diff, 1)

  return {
    'win_probability': round(prob, 3),
    'fair_moneyline': round(moneyline, 1),
    'runline': runline,
    'predicted_total': round(total_pred, 1)
  }

def main():
  csv_file = 'games.csv'
  # Example of updating CSV with new data
  # new_data = pd.DataFrame([{
  #   'date': '2023-09-15',
  #   'home_team': 'TeamA',
  #   'away_team': 'TeamB',
  #   'home_score': 5,
  #   'away_score': 3,
  #   'home_baseruns': 4.2,
  #   'away_baseruns': 3.7
  # }])
  # update_csv_with_new_data(csv_file, new_data)

  lr_win, lr_diff, lr_total = train_models(csv_file)

  # Example of making a prediction for a new game
  home_baseruns_est = 4.3
  away_baseruns_est = 3.8
  odds_info = compute_fair_odds(lr_win, lr_diff, lr_total, home_baseruns_est, away_baseruns_est)

  print("Win Probability:", odds_info['win_probability'])
  print("Fair Moneyline:", odds_info['fair_moneyline'])
  print("Runline:", odds_info['runline'])
  print("Predicted Total Runs:", odds_info['predicted_total'])

if __name__ == '__main__':
  main()
