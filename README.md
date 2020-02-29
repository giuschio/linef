## Training a neural net to score line regression hypotheses in the presence of noise
1. generate sinthetic data (examples) with some given percentage of outliers
2. train the net using DSAC
3. the purpose of the net is to assign a score to each hypothesis. The higher the score, the better the hypothesis is

during testing we use RANSAC to select the hypothesis with the highest score, as prediscted by the net
