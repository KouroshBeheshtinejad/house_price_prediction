# House Price in Tehran Prediction with Neural Network 

This project is a simple yet effective neural network model to predict house prices based on various features. It’s implemented in Python using TensorFlow/Keras.

---

## Project Structure

 All the code is in **one file** (`house_price_prediction.py`), which includes:

- Loading and preprocessing data
- Building the neural network model
- Training and validation
- Visualizing training history (Loss & MAE)
- Comparing predicted vs actual prices

## Features

- Dense neural network with **Dropout layers** to reduce overfitting
- **Mean Absolute Error (MAE)** and **MSE** metrics
- Training & validation loss plotted over epochs
- Predicted vs actual price scatter plot

## How It Works

1. **Data preprocessing**: Normalizes input features using standard scaling
2. **Model building**: Sequential model with 3 Dense layers (128 → 64 → 1)
3. **Training**: 100 epochs with validation set, monitoring MAE & Loss
4. **Evaluation**: Prints test MAE and visualizes predictions


## Model Performance

- Test MAE: 0.1936
- Network: Dense(128) → Dropout → Dense(64) → Dropout → Dense(1)

## Dataset

- Include a sample CSV or a link to the full dataset

## Visualization

- Training history plots: MAE & Loss over epochs
- Scatter plot: Predicted vs Actual prices

## Requirements

- Python >= 3.8
- TensorFlow
- Matplotlib
- Pandas / Numpy
- Install required packages:
`pip install tensorflow matplotlib pandas numpy`

## How to Run
```bash
`python house_price_prediction.py`
```

After running, you’ll see:
- MAE & Loss over epochs
- Test MAE 
- Scatter plot of predicted vs actual prices

## Notes

- All code is in one file for simplicity.
- Feel free to modify the neural network layers, optimizer, or number of epochs to improve performance.
- This project is suitable for portfolio/Github showcase
