import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

inputs = np.array([1, 6, 30, 7, 70, 43, 503, 201, 1005, 99], dtype=float)
outputs = np.array([0.024, 0.1524, 0.762, 0.1778, 1.778,
                   1.0922, 12.776, 5.1054, 25.527, 2.514], dtype=float)

layout1 = tf.keras.layers.Dense(units=1, input_shape=[1])

model = tf.keras.Sequential(layout1)
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.05),
    loss="mean_squared_error")

print("Training the Network.")

training = model.fit(inputs, outputs, epochs=100, verbose=False)

model.save("neronal-network.h5")
model.save_weights("weights.h5")

plt.xlabel("Training stages")
plt.ylabel("Errors")
plt.plot(training.history["loss"])
plt.show()

print("Training finalized.")

i = input("Enter your value in inches: ")
i = float(i)

prediction = model.predict([i])
print(f"Value in metters: {prediction}.")
