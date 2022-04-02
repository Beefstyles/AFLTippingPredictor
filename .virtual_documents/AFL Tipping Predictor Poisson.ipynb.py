import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from scipy.stats import poisson, skellam
import os


# Load training data
afl_results_2009_to_2018 = pd.read_csv(r"C:\Users\Beefsports\Documents\GitHub\AFLTippingPredictor\SourceData\AFL2009to2018.csv", encoding = "ISO-8859-1")
afl_results_2009_to_2018_results = pd.DataFrame(afl_results_2009_to_2018)


afl_results_2009_to_2018.head()


# Only take columns we need for this
afl_results_2009_to_2018_results = afl_results_2009_to_2018[['Date', 'Home Team', 'Away Team', 'Home Score', 'Away Score', 'Home Odds', 'Away Odds', 'Play Off Game?']] 
afl_results_2009_to_2018_results = afl_results_2009_to_2018_results.rename(columns={'Home Team': 'HomeTeam', 'Away Team': 'AwayTeam', 'Play Off Game?':'FinalsGame', 'Home Score': 'HomeScore', 'Away Score': 'AwayScore'})


# Drop all finals games from this model as footy tipping is only completed for the home and away season
afl_results_2009_to_2018_results = afl_results_2009_to_2018_results[afl_results_2009_to_2018_results.FinalsGame get_ipython().getoutput("= 'Y']")
afl_results_2009_to_2018_results = afl_results_2009_to_2018_results.drop(columns='FinalsGame')


# Split out Date into Day, Month, Year columns by using - as a delimiter
afl_results_2009_to_2018_results[['Day','Month', 'Year']] = afl_results_2009_to_2018_results.Date.str.split("-",expand=True)
afl_results_2009_to_2018_results = afl_results_2009_to_2018_results.drop(columns='Date')


# Move Year, Month, Day to the front of the dataframe
afl_results_2009_to_2018_results = afl_results_2009_to_2018_results[ ['Day'] + [ col for col in afl_results_2009_to_2018_results.columns if col get_ipython().getoutput("= 'Day' ] ]")
afl_results_2009_to_2018_results = afl_results_2009_to_2018_results[ ['Month'] + [ col for col in afl_results_2009_to_2018_results.columns if col get_ipython().getoutput("= 'Month' ] ]")
afl_results_2009_to_2018_results = afl_results_2009_to_2018_results[ ['Year'] + [ col for col in afl_results_2009_to_2018_results.columns if col get_ipython().getoutput("= 'Year' ] ]")


# Convert odds to percentage
afl_results_2009_to_2018_results['HomeOddsPercent'] = [1 / home_odds for home_odds in afl_results_2009_to_2018_results['Home Odds']]
afl_results_2009_to_2018_results['AwayOddsPercent'] = [1 / home_odds for home_odds in afl_results_2009_to_2018_results['Away Odds']]


# Rename some of the teams so they match the testing dataset
afl_results_2009_to_2018_results = afl_results_2009_to_2018_results.replace('Adelaide','Adelaide Crows')
afl_results_2009_to_2018_results = afl_results_2009_to_2018_results.replace('Brisbane','Brisbane Lions')
afl_results_2009_to_2018_results = afl_results_2009_to_2018_results.replace('Geelong','Geelong Cats')
afl_results_2009_to_2018_results = afl_results_2009_to_2018_results.replace('Gold Coast','Gold Coast Suns')
afl_results_2009_to_2018_results = afl_results_2009_to_2018_results.replace('Sydney','Sydney Swans')
afl_results_2009_to_2018_results = afl_results_2009_to_2018_results.replace('West Coast','West Coast Eagles')


afl_results_2009_to_2018_results.head()


# Check to confirm if Home Score and Away Score are numbers as expected
afl_results_2009_to_2018_results.applymap(np.isreal)


# Need to coerce the Year into a numeric value for use in filtering below
afl_results_2009_to_2018_results['Year'] = pd.to_numeric(afl_results_2009_to_2018_results['Year'],errors='coerce')

# Due to the up and down nature of footy teams over the years I'll take just the last five years of results for the model - just as a guess
afl_results_2014_to_2018_results = afl_results_2009_to_2018_results[(afl_results_2009_to_2018_results['Year'] >= 14) &
                                                                    (afl_results_2009_to_2018_results['Year'] <= 18)]
afl_results_2014_to_2018_results.head()


# Importing the tools required for the Poisson regression model - average points scored
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Below if home team then make HomeTeam team and AwayTeam opponent and vice versa
afl_tipping_model_data = pd.concat([afl_results_2014_to_2018_results[['HomeTeam','AwayTeam','HomeScore']].assign(home=1).rename(
            columns={'HomeTeam':'team','AwayTeam':'opponent','HomeScore':'score'}),
           afl_results_2014_to_2018_results[['AwayTeam','HomeTeam','AwayScore']].assign(home=0).rename(
            columns={'AwayTeam':'team', 'HomeTeam':'opponent','AwayScore':'score'})])

afl_tipping_model_data.head()


# Coerce score to numeric
afl_tipping_model_data['score'] = pd.to_numeric(afl_tipping_model_data['score'],errors='coerce')
afl_tipping_model_data.head()


# Creates a poisson model using the statsmodels generalised linear model (glm) with score as the dependent variable and
# home (i.e. if home team or not), team and opponent as input variables (R-style syntax), data is above 
# and family specifies as Poisson
# .fit() fits a generalised linear model for a given family

afl_2014_to_2018_poisson_model = smf.glm(formula="score ~ home + team + opponent", data=afl_tipping_model_data, 
                        family=sm.families.Poisson()).fit()
# Gives a glm regression result summary
afl_2014_to_2018_poisson_model.summary()


# Make a function to simulate any match

def simulate_match(tipping_model, homeTeam, awayTeam, max_score=250):
    home_score_avg = tipping_model.predict(pd.DataFrame(data={'team': homeTeam,
                                                             'opponent': awayTeam,'home':1},
                                                       index=[1])).values[0]
    away_score_avg = tipping_model.predict(pd.DataFrame(data={'team': awayTeam,
                                                             'opponent': homeTeam,
                                                             'home':0},
                                                       index=[1])).values[0]
    # List comprehension to calculate the predicted team matrix (Rows are home team, Columns are away team and diagonal is chance of a draw)
    team_pred = [[poisson.pmf(i, team_avg) for i in range (0, max_score+1)] for team_avg in [home_score_avg, away_score_avg]]
    return(np.outer(np.array(team_pred[0]), np.array(team_pred[1])))


# Load testing data
afl_results_2019 = pd.read_csv(r"C:\Users\Beefsports\Documents\GitHub\AFLTippingPredictor\SourceData\afl-2019-AUSEasternStandardTime_results.csv", encoding = "ISO-8859-1")
afl_results_2019_results = pd.DataFrame(afl_results_2019)
afl_results_2019_results.head()


afl_results_2019_scores = afl_results_2019[['Home Team', 'Away Team', 'Result']] # Only take Home Team, Away Team and Result
afl_results_2019_scores = afl_results_2019_scores.rename(columns={'Home Team': 'HomeTeam', 'Away Team': 'AwayTeam', 'Result':'FullScore'})
afl_results_2019_scores.head()


afl_results_2019_split_scores = pd.DataFrame(afl_results_2019_scores) # Make pandas dataframe
# Split out FullScore into Home Score and AwayScore columns by using - as a delimiter
afl_results_2019_split_scores[['HomeScore','AwayScore']] = afl_results_2019_split_scores.FullScore.str.split(" - ",expand=True)
afl_results_2019_split_scores.head()


# Drop FullScore column because we don't need it anymore
afl_results_2019_split_scores = afl_results_2019_split_scores.drop(columns='FullScore')
afl_results_2019_split_scores.head()


# Need to coerce the HomeScore and AwayScore results to numeric values as previously were strings - won't work with the poisson model
afl_results_2019_split_scores['HomeScore'] = pd.to_numeric(afl_results_2019_split_scores['HomeScore'],errors='coerce')
afl_results_2019_split_scores['AwayScore'] = pd.to_numeric(afl_results_2019_split_scores['AwayScore'],errors='coerce')

# afl_results_2019_split_scores.applymap(np.isreal)


# Set the base assumption that the home team wins to compare against
afl_results_2019_split_scores['PredictedWinnerHomeGroundAdv'] = 'Home'


afl_results_2019_split_scores['HomeOddsPercentPred'] = [np.sum(np.tril(simulate_match(afl_2014_to_2018_poisson_model, homeTeam, awayTeam), -1)) for (homeTeam, awayTeam) 
                                                           in zip(afl_results_2019_split_scores['HomeTeam'], afl_results_2019_split_scores['AwayTeam'])]

afl_results_2019_split_scores['AwayOddsPercentPred'] = [np.sum(np.triu(simulate_match(afl_2014_to_2018_poisson_model, homeTeam, awayTeam), 1)) for (homeTeam, awayTeam) 
                                                           in zip(afl_results_2009_to_2018_results['HomeTeam'], afl_results_2019_split_scores['AwayTeam'])]
afl_results_2019_split_scores.head()


afl_results_2019_split_scores['PredictedWinnerModel'] = ['Home' if homeoddspred >= awayoddspred
                                                    else 'Away' for (homeoddspred, awayoddspred) in zip(afl_results_2019_split_scores['HomeOddsPercentPred'], afl_results_2019_split_scores['AwayOddsPercentPred'])]
afl_results_2019_split_scores.head()


# Add actual winner to the test data column using a list comprehension

afl_results_2019_split_scores['ActualWinner'] = ['Home' if homescore > awayscore
                                                    else 'Away' if homescore < awayscore
                                                    else 'Draw' for (homescore, awayscore)
                                                    in zip(afl_results_2019_split_scores['HomeScore'],
                                                           afl_results_2019_split_scores['AwayScore'])]


afl_results_2019_split_scores['PredictionCorrect'] = ['Yes' if PredictedWinner == ActualWinner
                                                    else 'No' for (PredictedWinner, ActualWinner) in zip(afl_results_2019_split_scores['PredictedWinnerModel'],
                                                                                                         afl_results_2019_split_scores['ActualWinner'])]
afl_results_2019_split_scores['PredictionCorrectHomeGround'] = ['Yes' if PredictedWinnerHomeGroundAdv == ActualWinner
                                                    else 'No' for (PredictedWinnerHomeGroundAdv, ActualWinner) in zip(afl_results_2019_split_scores['PredictedWinnerHomeGroundAdv'],
                                                                                                                      afl_results_2019_split_scores['ActualWinner'])]
afl_results_2019_split_scores.head()


afl_results_2019_split_scores.to_excel("afl_results_2019_split_scores.xlsx", sheet_name='2019_results_predicted')


afl_results_results_2019_pred_correct_map = afl_results_2019_split_scores.apply(lambda x: True if x['PredictionCorrect'] == 'Yes' else False, axis = 1)
afl_results_results_2019_pred_correct = len(afl_results_results_2019_pred_correct_map[afl_results_results_2019_pred_correct_map == True].index)
print(afl_results_results_2019_pred_correct)


afl_results_results_2019_pred_homegroud_correct_map = afl_results_2019_split_scores.apply(lambda x: True if x['PredictionCorrectHomeGround'] == 'Yes' else False, axis = 1)
afl_results_results_2019_pred_homegroud_correct = len(afl_results_results_2019_pred_homegroud_correct_map[afl_results_results_2019_pred_correct_map == True].index)
print(afl_results_results_2019_pred_homegroud_correct)


afl_results_2019_predicted_pivot_pred = afl_results_2019_split_scores.pivot_table(afl_results_2019_split_scores, columns = ['PredictionCorrect'],
                                                                                  aggfunc='size')
afl_results_2019_predicted_pivot_home = afl_results_2019_split_scores.pivot_table(afl_results_2019_split_scores, columns = ['PredictionCorrectHomeGround'],
                                                                                  aggfunc='size')

afl_results_2019_predicted_pivot_pred.loc["Total"] = afl_results_2019_predicted_pivot_pred.sum()
afl_results_2019_predicted_pivot_home.loc["Total"] = afl_results_2019_predicted_pivot_home.sum()


afl_results_2019_predicted_pivot_pred['PercentageCorrectPoisson'] = afl_results_2019_predicted_pivot_pred['Yes'] / afl_results_2019_predicted_pivot_pred['Total']
afl_results_2019_predicted_pivot_pred.head()


afl_results_2019_predicted_pivot_home['PercentageCorrectPoisson'] = afl_results_2019_predicted_pivot_home['Yes'] / afl_results_2019_predicted_pivot_home['Total']
afl_results_2019_predicted_pivot_home.head()


combined_prediction_results_table = pd.concat([afl_results_2019_predicted_pivot_pred, afl_results_2019_predicted_pivot_home], axis = 1)
combined_prediction_results_table.columns=['PoissonPrediction','HomeGroundPrediction']
combined_prediction_results_table.head()
