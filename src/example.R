source('./client.R')
df <- read.csv("../datasets/sample_submission.csv")
status <- submit_predictions('./config.ini', df)
print(status)
