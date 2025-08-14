import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping


# 1. Load & preprocess data
df = pd.read_csv('housePrice.csv').dropna()
df['Area'] = pd.to_numeric(df['Area'], errors='coerce')
df = df.dropna(subset=['Area'])
df.drop(columns=['Address'], inplace=True)

# Encode categorical binary features
for col in ['Parking', 'Warehouse', 'Elevator']:
    df[col] = df[col].astype(int)

# Feature engineering
df['price_per_sqm'] = df['Price(USD)'] / df['Area']
df['area_per_room'] = df.apply(lambda r: r['Area'] if r['Room']==0 else r['Area']/r['Room'], axis=1)

# Target & features
X = df.drop(columns=['Price(USD)'])
y = np.log1p(df['Price(USD)'])  # log-transform

# Train/Val/Test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Scale features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# 2. Build the model
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    Dropout(0.2),
    Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    Dropout(0.2),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 3. Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100, batch_size=32,
    callbacks=[early_stop]
)

# 4. Evaluate
test_loss, test_mae = model.evaluate(X_test, y_test)
print("Test MAE:", test_mae)

# Predict & inverse log
y_pred = np.expm1(model.predict(X_test))
y_actual = np.expm1(y_test)

# 5. Visualizations
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend(); plt.title('Loss')
plt.savefig('loss.png')

plt.subplot(1,2,2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.legend(); plt.title('MAE')
plt.savefig('mae.png')
plt.show()

plt.figure(figsize=(6,6))
plt.scatter(y_actual, y_pred, alpha=0.6)
plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--')
plt.xlabel('Actual Price'); plt.ylabel('Predicted Price'); plt.title('Prediction vs Actual')
plt.savefig('pred_vs_actual.png')
plt.show()