import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from typing import Tuple
from io import StringIO

def update_csv_with_new_data(csv_file: str, new_data: pd.DataFrame) -> None:
  existing_data = pd.read_csv(csv_file)
  updated_data = pd.concat([existing_data, new_data], ignore_index=True)
  updated_data.to_csv(csv_file, index=False)

def train_models(data: pd.DataFrame) -> Tuple[LogisticRegression, LinearRegression, LinearRegression]:
  data['home_win'] = (data['home_score'] > data['away_score']).astype(int)
  data['baseruns_diff'] = data['home_baseruns'] - data['away_baseruns']
  data['score_diff'] = data['home_score'] - data['away_score']
  data['total_runs'] = data['home_score'] + data['away_score']
  data = data.dropna(subset=['home_baseruns','away_baseruns','home_score','away_score'])

  lr_win = LogisticRegression()
  X_lr = data[['baseruns_diff']]
  y_lr = data['home_win']
  lr_win.fit(X_lr, y_lr)

  lr_diff = LinearRegression()
  X_diff = data[['baseruns_diff']]
  y_diff = data['score_diff']
  lr_diff.fit(X_diff, y_diff)

  lr_total = LinearRegression()
  X_total = data[['home_baseruns','away_baseruns']]
  y_total = data['total_runs']
  lr_total.fit(X_total, y_total)

  return lr_win, lr_diff, lr_total

def probability_to_moneyline(p: float) -> float:
  if p <= 0:
    return 9999
  if p >= 1:
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

  return {
    'win_probability': round(prob, 3),
    'fair_moneyline': round(moneyline, 1),
    'runline': round(run_diff, 1),
    'predicted_total': round(total_pred, 1)
  }

def main():
  # Sample CSV data with 20 rows
  sample_csv = """date,home_team,away_team,home_score,away_score,home_baseruns,away_baseruns
2023-09-01,TeamA,TeamB,3,2,3.4,2.7
2023-09-02,TeamA,TeamB,5,1,4.2,2.1
2023-09-03,TeamA,TeamB,2,3,3.1,3.9
2023-09-04,TeamA,TeamB,7,6,6.3,5.5
2023-09-05,TeamA,TeamB,4,4,4.0,3.9
2023-09-06,TeamB,TeamA,6,3,5.7,2.8
2023-09-07,TeamB,TeamA,2,5,2.4,4.8
2023-09-08,TeamB,TeamA,1,3,2.2,3.7
2023-09-09,TeamB,TeamA,4,8,4.3,7.6
2023-09-10,TeamB,TeamA,2,2,3.1,3.3
2023-09-11,TeamC,TeamD,3,5,3.8,4.6
2023-09-12,TeamC,TeamD,1,1,2.1,2.4
2023-09-13,TeamC,TeamD,4,0,4.2,1.9
2023-09-14,TeamC,TeamD,2,3,2.9,3.7
2023-09-15,TeamC,TeamD,6,4,5.2,3.9
2023-09-16,TeamD,TeamC,3,5,3.4,5.1
2023-09-17,TeamD,TeamC,6,7,5.1,6.5
2023-09-18,TeamD,TeamC,2,1,2.7,2.4
2023-09-19,TeamD,TeamC,5,3,4.8,3.2
2023-09-20,TeamD,TeamC,0,2,1.2,2.5
"""

  df = pd.read_csv(StringIO(sample_csv))
  lr_win, lr_diff, lr_total = train_models(df)

  # Example usage
  home_baseruns_est = 4.5
  away_baseruns_est = 3.0
  odds_info = compute_fair_odds(lr_win, lr_diff, lr_total, home_baseruns_est, away_baseruns_est)

  print("Win Probability:", odds_info['win_probability'])
  print("Fair Moneyline:", odds_info['fair_moneyline'])
  print("Runline:", odds_info['runline'])
  print("Predicted Total Runs:", odds_info['predicted_total'])

if __name__ == '__main__':
  main()
