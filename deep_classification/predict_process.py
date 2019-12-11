import os
import tensorflow as tf
from tensorflow import keras
# Recreate the exact same model purely from the file
new_model = keras.models.load_model('path_to_my_model.h5')

import numpy as np

# Check that the state is preserved
a = [63,145,233,150,2.3,3,0]
new_predictions = new_model.predict(a)
# np.testing.assert_allclose(predictions, new_predictions, rtol=1e-6, atol=1e-6)