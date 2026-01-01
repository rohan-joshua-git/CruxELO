# CruxELO
Using Natural Language Sentiment Analysis and ELO Algorithms to reccomend accurate Rock Climbing Grades

# Project Overview/Introduction
This project develops an ELO-like rating system for climbing, specifically addressing the subjective nature of traditional climbing grades (e.g., V-scale for bouldering). The goal is to provide a more objective, data-driven assessment of both climber skill and climb difficulty by adapting the ELO rating algorithm, which was designed for 2 player games like chess. A unique aspect of this system is the integration of user-generated feedback, including perceived grades and free-text comments, processed through sentiment analysis, to dynamically influence rating adjustments. This approach aims to create a responsive and community-driven grading system that reflects the collective experience of climbers on specific problems, offering a richer and more accurate representation of difficulty.

# Key Features
1. Dynamic ELO Ratings: Continuously updates ratings for both climbers ('players') and climbs ('problems') based on attempt outcomes (send/fail).
2. Sentiment Analysis Integration: Utilizes Natural Language Processing (NLP) to extract sentiment from user comments, providing qualitative feedback on climbs.
3. Perceived Grade Data: Captures climbers' subjective assessment of a climb's difficulty, allowing for direct comparison against official grades and the system's ELO-derived grade.
4. Dynamic K-Factor Adjustment: K-factors (which determine the magnitude of rating changes) are dynamically adjusted based on strong sentiment and significant perceived grade deviations, accelerating the rating convergence for contentious or misgraded problems.
5. Chronological Processing: Processes logbook entries in chronological order to simulate a real-time evolving rating system.
6. V-Grade Recommendation: Converts the final ELO rating of climbs back into a human-readable V-grade equivalent for easy interpretation.
7. Data Persistence: Stores all processed data, including historical attempts, final climber ELOs, and final climb ELOs, in CSV format.

# Assumptions Made
1. System board-like Data: Assumed data from a Systemboards, where climbers log attempts, perceived grades, and comments.
2. Official Grades: Each climb has an 'OfficialGrade' provided, serving as an initial benchmark.
3. Sentiment Expressed in Comments: User comments contain discernible sentiment relevant to the climb's difficulty or quality.
4. K-Factor Calibration: Initial K-factor values and their adjustment multipliers are reasonable starting points, subject to calibration with real-world data.
5. ELO to V-Grade Mapping: A linear-like mapping exists between ELO points and V-grades for conversion purposes.

# Limitations 
1. Synthetic Data: The current implementation primarily uses synthetic data due to access limitations, which may not fully capture the complexity and nuances of real climbing logbook entries.
2. Simple K-Factor Adjustment: The dynamic K-factor logic is relatively basic and could be refined (using Glicko-2's rating deviation).
3. VADER's Generalization: VADER sentiment analysis, while effective, might not fully capture climbing-specific jargon or subtle emotional cues without fine-tuning on a domain-specific corpus.
5. Batch Processing: While chronological, the system processes all historical data in a batch, rather than real-time incremental updates.

# Key Findings/Insights from Analysis
1. ELO Differentiation: The ELO system successfully differentiated climber skills and climb difficulties, with climber ELOs clustering around the initial value and climb ELOs exhibiting a broader distribution reflecting diverse difficulties.
2. Subjectivity Quantified: Perceived grade deviations revealed significant individual variability, confirming the subjective nature of climbing grades and the value of a dynamic rating system.
Sentiment's Role: Sentiment analysis showed an overall positive bias in comments, but strong sentiments (positive or negative) were effective triggers for accelerating ELO adjustments for climbs, indicating areas of high contention or consensus.
3. Complex Correlation: The scatter plot of sentiment vs. perceived grade deviation highlighted a complex, non-linear relationship, suggesting that a climb perceived as 'hard' (positive deviation) might still evoke positive sentiment if it's rewarding, and a 'soft' climb (negative deviation) could still evoke negative sentiment if poorly set. This justifies using both metrics for robust ELO adjustment.
4. V-Grade Recommendations: The system successfully generated ELO-derived V-grade recommendations, offering an alternative, data-driven perspective on a climb's difficulty.
