import tensorflow as tf
import tensorflow_hub as hub

# Load the pre-trained model
loaded_model_6 = tf.keras.models.load_model("model_6.h5",
                                            custom_objects={"KerasLayer": hub.KerasLayer})

# Define the tweet
tweet = "Millions of people died in the car crash"

# Get the prediction probability and round it to get the predicted class
pred_prob = tf.squeeze(loaded_model_6.predict([tweet]))
pred = tf.round(pred_prob)

# Print prediction and probability
print(f"Pred: {int(pred)}, Prob: {pred_prob}")
print(f"Text:\n{tweet}\n")

# Define thresholds for disaster prediction
threshold_disaster = 0.5  # Adjust this threshold as needed

# Check if the prediction is close to 1 (disaster) or 0 (not disaster)
if pred_prob >= threshold_disaster:
    print("The text is close to a disaster.")
else:
    print("The text is not close to a disaster.")
