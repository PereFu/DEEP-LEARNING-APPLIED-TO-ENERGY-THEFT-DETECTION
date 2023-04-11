# Union of wide and deep learning applied to energy theft detection
Implementation of a model based on the union of a wide and a deep convolutional neural network to detect energy theft based on the a daily time series consumption. 

![image](https://user-images.githubusercontent.com/60358958/231254791-7be52589-24c2-49e4-a539-1c349f2ec2fc.png)

This is a independent implementation based of a self-interpretation of the paper "Wide and Deep Convolutional Neural Networks for Electricity-Theft Detection to Secure Smart Grids" (https://ieeexplore.ieee.org/document/8233155/).
Data from: https://github.com/henryRDlab/ElectricityTheftDetection.

To run the code, you must follow this secuence:
1. data_preparation --> Sort dates, split data into labels and time series, one hot encode, outliers and NaNs treatment. Returns the data .cvs used in the wide_cnn notebook.
2. wide_cnn --> Training and performance metrics of the model.

The code has been run in Colab Notebooks, you must change the data URL to execute it.

Data --> 42.372 users, 1035 days, daily energy consumption.
Code --> Python, Pytorch, Colab Notebook.
