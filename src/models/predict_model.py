import train_model
import numpy as np

model = train_model.get_unet()

x_train, y_train = train_model.get_data()

y_output = model.predict(x_train, verbose=1)

y_output.save('y_masks')