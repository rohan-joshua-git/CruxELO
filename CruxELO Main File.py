##IMPORTS##
#import boardlib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import nltk
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

##KILTERBOARD DATA LOADING AND PRE-PROCESSING##
try:
    kilterboard_df = pd.read_csv('kilterboard_logbook.csv')
    print("Kilterboard logbook loaded successfully.")

except FileNotFoundError:
    print("Error: 'kilterboard_logbook.csv' not found. Please ensure the file is uploaded to Colab and try again.")
    
    #Synthetic data(Demonstration Purposes)
    print("Creating synthetic data for demonstration.")
    n_records = 100 #Amount of fake data
    base_comments = [
        "This climb was amazing! So much fun and great holds.",
        "Terrible setting, felt way harder than a V5. Very frustrating.",
        "Solid V6, definitely a project for me. The crux was brutal.",
        "Interesting movement, but the beta was unclear. Had to try many times.",
        "Super easy for its grade, almost a flash. Good warm-up.",
        "Dyno was too big, felt impossible for my height. Bad design.",
        "Loved the variety of holds, very creative route. High quality.",
        "Slippery foot holds, dangerous fall potential. Not well maintained.",
        "A classic V7, took a few tries but worth it. The top out was scary but fun.",
        "Overgraded. Definitely a V4 at most. Disappointing.",
        "Neutral feelings, nothing special, nothing bad. Just a climb.",
        "The volume was spinning, almost fell. Needs fixing ASAP.",
        "Great problem, very technical and required precise movements.",
        "Too crimpy for my fingers, not my style. Still a good challenge though.",
        "Fantastic V8, felt strong today and sent it! Psyched!",
        "Challenging but rewarding. A true test of strength and endurance.",
        "This felt like a V6, not a V5. Still fun, but definitely sandbagged.",
        "Too many people on the route, hard to focus. Crowded gym.",
        "Awesome compression, loved the body tension required.",
        "The slopers were impossible, couldn\'t hold on. Weak.",
        "Decent climb, nothing to write home about.",
        "Very dynamic, I liked the big moves.",
        "Could use more chalk, the holds were greasy.",
        "Perfect climb for practicing flags. Enjoyed it."
    ]
    
    data = {
        "ClimberID": np.random.randint(100, 500, n_records),
        "ClimbID": np.random.randint(1000, 2000, n_records),
        "Timestamp": [datetime(2023, 1, 1) + timedelta(days=int(d), hours=int(h), minutes=int(m))
                      for d, h, m in zip(np.random.randint(0, 365, n_records),
                                         np.random.randint(0, 24, n_records),
                                         np.random.randint(0, 60, n_records))],
        "Outcome": np.random.choice(["Send", "Fail"], n_records, p=[0.5, 0.5]),
        "LoggedGrade": np.random.choice(["V4", "V5", "V5-", "V5/6", "V6", "V6+", "V7", "V7-", "V8", "V8/9", "V9"], n_records, p=[0.1, 0.2, 0.1, 0.05, 0.15, 0.05, 0.1, 0.05, 0.1, 0.05, 0.05]),
        "OfficialGrade": np.random.choice(["V4", "V5", "V6", "V7", "V8", "V9"], n_records),
        "Comments": np.random.choice(base_comments, n_records)
    }
    
    kilterboard_df = pd.DataFrame(data)
    kilterboard_df["Comments"] = kilterboard_df["Comments"].apply(lambda x: x + np.random.choice([
        "Really enjoyed the sequence.",
        "The first move was hard.",
        "Needs better lighting.",
        "This is my new favorite!",
        "Not a fan of the starting position.",
        "Took me ages to figure it out.",
        "Fun challenge.",
        "Highly recommend!",
        "Would climb again.",
        "Felt dangerous.",
        "A bit boring.",
        "So much fun!",
        "Hated it.",
        "Very creative.",
        "Standard route."
    ], 1)[0] if np.random.rand() > 0.5 else x)

print("\nInitial DataFrame Head:")
print(kilterboard_df.head())
print("\nDataFrame Info:")
print(kilterboard_df.info())

#Renaming for Clarity
kilterboard_df = kilterboard_df.rename(columns={
    "ClimberID": "ClimberID",
    "ClimbID": "ClimbID",
    "date": "Timestamp",
    "is_ascent": "Outcome",
    "logged_grade": "LoggedGrade",
    "displayed_grade": "OfficialGrade",
    "comment": "Comments" 
})

#Boolean to Natural Language
if 'Outcome' in kilterboard_df.columns and kilterboard_df['Outcome'].dtype == 'bool':
    kilterboard_df['Outcome'] = kilterboard_df['Outcome'].apply(lambda x: 'Send' if x else 'Fail')

#Variable data consistency check
kilterboard_df['Timestamp'] = pd.to_datetime(kilterboard_df['Timestamp'])
kilterboard_df['Comments'] = kilterboard_df['Comments'].fillna('')

print("\nDataFrame after renaming and initial processing:")
print(kilterboard_df.head())
print(kilterboard_df.info())

# NLTK Downloads
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)
print("\nNLTK resources downloaded.")

##USER COMMENTS PROCESSING##
stop_words = set(stopwords.words('english'))

#Process words
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = text.strip()  # Remove whitespace
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space

    #Tokenise words
    tokens = word_tokenize(text)

    #Remove stop words
    filtered_tokens = [word for word in tokens if word not in stop_words]

    return filtered_tokens

# Apply pre-processing function
kilterboard_df['Preprocessed_Comments'] = kilterboard_df['Comments'].apply(preprocess_text)

print("Comments preprocessed through a combined pipeline successfully.")
print(kilterboard_df[['Comments', 'Preprocessed_Comments']].head())

##GRADE CONVERSIONS - for future commercial use##
def convert_grade_to_numerical(grade_str):
    if not isinstance(grade_str, str):
        return np.nan

    grade_str = grade_str.strip().upper()

    if not grade_str.startswith('V'):
        return np.nan

    numeric_part = grade_str[1:]

    try:
        if '-' in numeric_part:
            base_grade = float(numeric_part.replace('-', ''))
            return base_grade - 0.5
        elif '+' in numeric_part:
            base_grade = float(numeric_part.replace('+', ''))
            return base_grade + 0.5
        elif '/' in numeric_part:
            grades = numeric_part.split('/')
            if len(grades) == 2:
                return (float(grades[0]) + float(grades[1])) / 2
            else:
                return np.nan
        else:
            return float(numeric_part)
    except ValueError:
        return np.nan

kilterboard_df['Numerical_LoggedGrade'] = kilterboard_df['LoggedGrade'].apply(convert_grade_to_numerical)
kilterboard_df['Numerical_OfficialGrade'] = kilterboard_df['OfficialGrade'].apply(convert_grade_to_numerical)

print("Logged and Official grades converted to numerical format successfully.")
print(kilterboard_df[['LoggedGrade', 'Numerical_LoggedGrade', 'OfficialGrade', 'Numerical_OfficialGrade']].head())
print(kilterboard_df[['Numerical_LoggedGrade', 'Numerical_OfficialGrade']].describe())

##SENTIMENT ANALYSIS##

# Initialize VADER sentiment analyser
sia = SentimentIntensityAnalyzer()

def get_vader_sentiment(text):
    return sia.polarity_scores(text)["compound"]

#Applying sentiment analysis
kilterboard_df["SentimentScore"] = kilterboard_df["Comments"].apply(get_vader_sentiment)

def categorize_sentiment(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

kilterboard_df["SentimentCategory"] = kilterboard_df["SentimentScore"].apply(categorize_sentiment)

print("Sentiment scores and categories generated successfully.")
print(kilterboard_df[["Comments", "SentimentScore", "SentimentCategory", "Numerical_LoggedGrade"]].head())

##ELO RATING INITIALISATION##
def numerical_grade_to_initial_elo(num_grade):
    if pd.isna(num_grade) or num_grade < 0:
        return 1000 #Default Elo

    grade_elo_points = {
        0: 800,    #V0
        1: 950,    #V1
        2: 1100,   #V2
        3: 1250,   #V3
        4: 1400,   #V4
        5: 1550,   #V5
        6: 1700,   #V6
        7: 1850,   #V7
        8: 2000,   #V8
        9: 2150,   #V9
        10: 2300,  #V10
        11: 2450,  #V11
        12: 2600,  #V12
        13: 2750,  #V13
        14: 2900,  #V14
        15: 3050,  #V15
        16: 3200,  #V16
        17: 3350   #V17
    }

    grades_sorted = sorted(grade_elo_points.keys())

    if num_grade <= grades_sorted[0]:
        return grade_elo_points[grades_sorted[0]]
    #if out of grade range elo
    if num_grade >= grades_sorted[-1]:
        return grade_elo_points[grades_sorted[-1]] + (num_grade - grades_sorted[-1]) * 150

    for i in range(len(grades_sorted) - 1):
        g1 = grades_sorted[i]
        g2 = grades_sorted[i+1]
        if g1 <= num_grade <= g2:
            elo1 = grade_elo_points[g1]
            elo2 = grade_elo_points[g2]
            return elo1 + (elo2 - elo1) * (num_grade - g1) / (g2 - g1)
    return 1200

#Intialise default Climber Elo
initial_climber_elo = 1200
unique_climber_ids = kilterboard_df["ClimberID"].unique()
climbers_initial_elo_map = {climber_id: initial_climber_elo for climber_id in unique_climber_ids}

#Generate initial Climb Elo
unique_climb_ids = kilterboard_df["ClimbID"].unique()
climbs_initial_elo_map = {}
for climb_id in unique_climb_ids:
    official_grade_num = kilterboard_df[kilterboard_df["ClimbID"] == climb_id]["Numerical_OfficialGrade"].iloc[0]
    climbs_initial_elo_map[climb_id] = numerical_grade_to_initial_elo(official_grade_num)

# Create DataFrames for current ELOs for climbers and climbs
climbers_df = pd.DataFrame({
    "ClimberID": list(climbers_initial_elo_map.keys()),
    "CurrentELO": list(climbers_initial_elo_map.values())
})

climbs_df = pd.DataFrame({
    "ClimbID": list(climbs_initial_elo_map.keys()),
    "OfficialGrade": [kilterboard_df[kilterboard_df["ClimbID"] == cid]["OfficialGrade"].iloc[0] for cid in list(climbs_initial_elo_map.keys())],
    "InitialELO": list(climbs_initial_elo_map.values()),
    "CurrentELO": list(climbs_initial_elo_map.values())
})

# Define base K-factors
K_FACTOR_CLIMBER_BASE = 40 # Standard K-factor for new/active players
K_FACTOR_CLIMB_BASE = 30   # Lower than climber K-factor

print("Initial ELO ratings generated for climbers and climbs.")
print("Climbers DataFrame head:")
print(climbers_df.head())
print("\nClimbs DataFrame head:")
print(climbs_df.head())
print(f"\nBase K-factor for Climbers: {K_FACTOR_CLIMBER_BASE}")
print(f"Base K-factor for Climbs: {K_FACTOR_CLIMB_BASE}")

##ELO ALGORITHM IMPLEMENTATION WITH SENTIMENT INTEGRATION## 
def calculate_expected_score(R_A, R_B): #Player A VS Climb B
    return 1 / (1 + 10**((R_B - R_A) / 400))

def update_elo_ratings(climber_id, climb_id, outcome,
                       climber_elo_map, climb_elo_map,
                       K_C_base=K_FACTOR_CLIMBER_BASE, K_P_base=K_FACTOR_CLIMB_BASE,
                       sentiment_score=0, perceived_grade_deviation=0):

    R_C = climber_elo_map.get(climber_id, K_FACTOR_CLIMBER_BASE) # Default ELO if not found
    R_P = climb_elo_map.get(climb_id, 1500)

    #Calculate expected scores
    E_C = calculate_expected_score(R_C, R_P)#Expected score for climber to 'win' against climb
    E_P = calculate_expected_score(R_P, R_C) #Expected score for climb to 'win' against climber

    #Calculate actual scores
    if outcome in ["Send", "Flash"]:
        S_C = 1  # Climber wins
        S_P = 0  # Climb loses
    else: # 'Fail'
        S_C = 0  # Climber loses
        S_P = 1  # Climb wins

    #Dynamic K-factor Adjustment
    K_C = K_C_base
    K_P = K_P_base

    # Adjust K-factor based on perceived grade deviation
    
    # If perceived grade is significantly different from current ELO or high sentiment, increase K to adjust faster
    if abs(perceived_grade_deviation) > 1.0:
        K_P *= 1.5
    if abs(sentiment_score) > 0.7:
        K_P *= 1.2 

    #K-factor bounds
    K_C = max(10, min(K_C, 80)) 
    K_P = max(10, min(K_P, 80))

    # Update ratings
    new_R_C = R_C + K_C * (S_C - E_C)
    new_R_P = R_P + K_P * (S_P - E_P)

    return new_R_C, new_R_P

print("ELO update function defined.")

# Initialize current ELO from dataframe
current_climber_elos = climbers_df.set_index("ClimberID")["CurrentELO"].to_dict()
current_climb_elos = climbs_df.set_index("ClimbID")["CurrentELO"].to_dict()

print("Climber and Climb ELO dictionaries initialized for processing.")

#Sort by timestamp to process attempts chronologically
kilterboard_df_sorted = kilterboard_df.sort_values(by="Timestamp").copy()
kilterboard_df_sorted["UpdatedClimberELO"] = np.nan
kilterboard_df_sorted["UpdatedClimbELO"] = np.nan

# Iteratively apply the ELO update function
for index, row in kilterboard_df_sorted.iterrows():
    climber_id = row["ClimberID"]
    climb_id = row["ClimbID"]
    outcome = row["Outcome"]
    sentiment_score = row["SentimentScore"]
    logged_grade_numerical = row["Numerical_LoggedGrade"]
    official_grade_numerical = row["Numerical_OfficialGrade"]

    # Calculate logged grade deviation
    perceived_grade_deviation = logged_grade_numerical - official_grade_numerical if not pd.isna(logged_grade_numerical) and not pd.isna(official_grade_numerical) else 0

    new_C_elo, new_P_elo = update_elo_ratings(climber_id, climb_id, outcome,
                                            current_climber_elos, current_climb_elos,
                                            sentiment_score=sentiment_score,
                                            perceived_grade_deviation=perceived_grade_deviation)

    current_climber_elos[climber_id] = new_C_elo
    current_climb_elos[climb_id] = new_P_elo

    kilterboard_df_sorted.loc[index, "UpdatedClimberELO"] = new_C_elo
    kilterboard_df_sorted.loc[index, "UpdatedClimbELO"] = new_P_elo

# Update the main climbers_df and climbs_df with the final ELOs
climbers_df["CurrentELO"] = climbers_df["ClimberID"].map(current_climber_elos)
climbs_df["CurrentELO"] = climbs_df["ClimbID"].map(current_climb_elos)

print("ELO ratings updated for all attempts in chronological order.")
print("Updated kilterboard_df_sorted head (showing ELO progression):")
print(kilterboard_df_sorted[["ClimberID", "ClimbID", "Outcome", "SentimentScore", "Numerical_LoggedGrade", "UpdatedClimberELO", "UpdatedClimbELO"]].head()) # Changed column name

print("\nFinal Climbers DataFrame head:")
print(climbers_df.head())
print("\nFinal Climbs DataFrame head:")
print(climbs_df.head())

#Convert ELO back to V-grade 
def elo_to_vgrade(elo):
    grade_elo_midpoints = {
        800: "V0", 
        950: "V1", 
        1100: "V2", 
        1250: "V3", 
        1400: "V4",
        1550: "V5", 
        1700: "V6", 
        1850: "V7", 
        2000: "V8", 
        2150: "V9",
        2300: "V10", 
        2450: "V11", 
        2600: "V12", 
        2750: "V13", 
        2900: "V14",
        3050: "V15", 
        3200: "V16", 
        3350: "V17"
    }
    
    #Find closest V-grade match
    sorted_elo_points = sorted(grade_elo_midpoints.items())

    if pd.isna(elo):
        return np.nan
    
    closest_grade = "Unknown"
    min_diff = float("inf")

    for e_val, grade_str in sorted_elo_points:
        diff = abs(elo - e_val)
        if diff < min_diff:
            min_diff = diff
            closest_grade = grade_str
    return closest_grade

climbs_df["Recommended_V_Grade"] = climbs_df["CurrentELO"].apply(elo_to_vgrade)

print("\nClimbs DataFrame with Recommended V-Grade:")
print(climbs_df.head())

##DATA PERSISTENCE##
kilterboard_df_sorted.to_csv("kilterboard_processed_attempts.csv", index=False)
print("kilterboard_processed_attempts.csv saved successfully.")

climbers_df.to_csv("final_climber_elos.csv", index=False)
print("final_climber_elos.csv saved successfully.")

climbs_df.to_csv("final_climb_elos.csv", index=False)
print("final_climb_elos.csv saved successfully.")

##FINAL ANALYSIS AND VISUALISATION##

#Climber ELO Ratings distribution
print("\nDescriptive statistics for Climber ELO Ratings:")
print(climbers_df["CurrentELO"].describe())
plt.figure(figsize=(10, 6))
sns.histplot(climbers_df["CurrentELO"], kde=True)
plt.title("Distribution of Climber ELO Ratings")
plt.xlabel("Climber ELO")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

#Climb ELO Ratings Distribution
print("\nDescriptive statistics for Climb ELO Ratings:")
print(climbs_df["CurrentELO"].describe())
plt.figure(figsize=(10, 6))
sns.histplot(climbs_df["CurrentELO"], kde=True)
plt.title("Distribution of Climb ELO Ratings")
plt.xlabel("Climb ELO")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

#Logged Grade Deviation vs. Sentiment Score
# Calculate Logged_Official_Deviation
kilterboard_df_sorted["Logged_Official_Deviation"] = kilterboard_df_sorted["Numerical_LoggedGrade"] - kilterboard_df_sorted["Numerical_OfficialGrade"]

print("\nDescriptive statistics for Logged_Official_Deviation:")
print(kilterboard_df_sorted["Logged_Official_Deviation"].describe())
print("\nDescriptive statistics for SentimentScore:")
print(kilterboard_df_sorted["SentimentScore"].describe())

plt.figure(figsize=(10, 6))
sns.scatterplot(data=kilterboard_df_sorted, x="Logged_Official_Deviation", y="SentimentScore", alpha=0.6)
plt.title("Sentiment Score vs. Logged Grade Deviation")
plt.xlabel("Logged Grade Deviation (Logged Grade - Official Grade)")
plt.ylabel("Sentiment Score (VADER Compound)")
plt.axvline(0, color="red", linestyle="--", linewidth=0.8, label="Zero Deviation")
plt.axhline(0, color="green", linestyle="--", linewidth=0.8, label="Neutral Sentiment")
plt.legend()
plt.tight_layout()
plt.show()

#Official Grade vs. Recommended ELO Grade
comparison_df = kilterboard_df_sorted.merge(climbs_df[["ClimbID", "Recommended_V_Grade"]], on="ClimbID", how="left")

plt.figure(figsize=(12, 7))
sns.boxplot(data=comparison_df, x="OfficialGrade", y="Numerical_OfficialGrade", hue="Recommended_V_Grade", palette="viridis", dodge=True)
plt.title("Official Grade vs. ELO-Recommended V-Grade")
plt.xlabel("Official Grade")
plt.ylabel("Numerical Grade Value")
plt.legend(title="Recommended V-Grade", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()