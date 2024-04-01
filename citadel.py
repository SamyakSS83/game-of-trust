import csv
import numpy as np
from hmmlearn.hmm import MultinomialHMM
from sklearn.cluster import KMeans

# Load the data from the CSV file
with open('input_game.csv') as f:
    reader = csv.reader(f)
    data = list(reader)

# Extract the relevant columns and convert the actions to integers (0 for TRUST, 1 for CHEAT)
game_data = [(int(row[0]), int(row[1]), int(row[2]), int(row[3])) for row in data[1:]]

# Initialize an empty list to store the observations for each player
observations = []

# Initialize an empty list to store the HMMs for each player
hmm_models = []

# Iterate through the games and players, creating an HMM and extracting the observations for each player
for game_id, p1_id, p2_id, _ in game_data:
    if len(observations) < p1_id:
        observations.append([])
    if len(observations) < p2_id:
        observations.append([])
    actions = [row[4] for row in game_data if row[0] == game_id and (row[1] == str(p1_id) or row[2] == str(p1_id))]
    observations[p1_id].append(actions)
    actions = [row[4] for row in game_data if row[0] == game_id and (row[1] == str(p2_id) or row[2] == str(p2_id))]
    observations[p2_id].append(actions)

# Create an HMM for each player
for i, obs in enumerate(observations):
    obs = [int(o) for o in sum(obs, [])]
    hmm = MultinomialHMM(n_components=2, covariance_type='diag')
    hmm.fit(np.array(obs).reshape(-1, 1))
    hmm_models.append(hmm)

# Extract the emission probabilities for each player's actions
emissions = []
for hmm in hmm_models:
    emissions.append(hmm.emissionprob([[0], [1]])[0])

# Use KMeans clustering to assign players to distinct strategies
kmeans = KMeans(n_clusters=len(emissions), random_state=0).fit(emissions)

# Print the number of distinct strategies (i.e., the number of clusters)
print(len(set(kmeans.labels_)))