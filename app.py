from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load and prepare the dataset
df = pd.read_csv('dataset/ipl.csv')

# Drop unnecessary columns
df.drop(['mid', 'venue', 'batsman', 'bowler', 'striker', 'non-striker'], axis=1, inplace=True)

# Consider only data after 5 overs
df = df[df['overs'] >= 5.0]

# Keep only consistent teams
consistent_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
                    'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
                    'Delhi Daredevils', 'Sunrisers Hyderabad']
df = df[(df['bat_team'].isin(consistent_teams)) & (df['bowl_team'].isin(consistent_teams))]

# Convert date column
df['date'] = pd.to_datetime(df['date'])

# Rename columns for consistency (if needed)
df.rename(columns={'runs_in_prev_5': 'runs_last_5', 'wickets_in_prev_5': 'wickets_last_5'}, inplace=True)

# One-Hot Encoding for bat_team and bowl_team
encoded_df = pd.get_dummies(df, columns=['bat_team', 'bowl_team'])

# Rearranging the columns for consistency
column_order = ['date'] + [col for col in encoded_df.columns if col.startswith('bat_team_') or col.startswith('bowl_team_')] + ['overs', 'runs', 'wickets', 'runs_last_5', 'wickets_last_5', 'total']
encoded_df = encoded_df[column_order]

# Splitting data based on year to avoid data leakage
X_train = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year <= 2016]
X_test = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year >= 2017]
y_train = encoded_df[encoded_df['date'].dt.year <= 2016]['total'].values
y_test = encoded_df[encoded_df['date'].dt.year >= 2017]['total'].values

# Removing the 'date' column from training and testing sets
X_train.drop(labels='date', axis=1, inplace=True)
X_test.drop(labels='date', axis=1, inplace=True)

# Train the model
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)

# Function to predict score
def predict_score(batting_team, bowling_team, overs, runs, wickets, runs_last_5, wickets_last_5):
    # Create input feature vector
    temp_array = []

    # Batting Team One-Hot Encoding (Fixed Order)
    for team in consistent_teams:
        temp_array.append(1 if team == batting_team else 0)

    # Bowling Team One-Hot Encoding (Fixed Order)
    for team in consistent_teams:
        temp_array.append(1 if team == bowling_team else 0)

    # Append numeric inputs
    temp_array.extend([overs, runs, wickets, runs_last_5, wickets_last_5])

    # Convert to numpy array and reshape for prediction
    temp_array = np.array([temp_array])

    # Predict Score
    predicted_score = int(linear_regressor.predict(temp_array)[0])
    return predicted_score

# Flask Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        batting_team = request.form['batting_team']
        bowling_team = request.form['bowling_team']
        overs = float(request.form['overs'])
        runs = int(request.form['runs'])
        wickets = int(request.form['wickets'])
        runs_last_5 = int(request.form['runs_last_5'])  # Fixed key
        wickets_last_5 = int(request.form['wickets_last_5'])  # Fixed key

        result = predict_score(batting_team, bowling_team, overs, runs, wickets, runs_last_5, wickets_last_5)

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
