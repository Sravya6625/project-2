# project-2
⚡ Residential Electricity Consumption Forecasting using Deep Learning

This project focuses on accurately forecasting residential electricity consumption by leveraging advanced deep learning models. With the rise in population and increased usage of household appliances, predicting electricity usage is essential for maintaining a balanced electrical load.

🔍 Overview

We developed and evaluated several neural network architectures using the Electricity Consumption dataset from the UCI repository, which includes data from three household sub-meters. The models implemented include:

CNN (Convolutional Neural Networks)

BiLSTM (Bidirectional Long Short-Term Memory)

BIGRU (Bidirectional Gated Recurrent Units)

SA (Self-Attention Mechanism)

CNN-BiLSTM-SA and CNN-BIGRU-SA hybrid models


A feature selection technique using Maximal Information Coefficient (MIC) was applied to enhance model efficiency by identifying the most relevant features.

📈 Results

CNN-BiLSTM-SA achieved an R² score of 0.97

CNN-BIGRU-SA outperformed with an R² score of 0.9781

Both models significantly outperformed traditional methods such as Linear Regression and Support Vector Machines in terms of RMSE and MAE


✅ Highlights

Real-time electricity consumption forecasting using deep learning

Advanced hybrid models incorporating attention mechanisms

Enhanced accuracy through feature selection with MIC


📂 Dataset

UCI Machine Learning Repository: Individual household electric power consumption



