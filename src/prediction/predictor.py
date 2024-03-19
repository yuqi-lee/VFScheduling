class Predictor:
    def __init__(self):
        self.models = {}  # Dictionary to store different models

    def add_model(self, model_name, model_instance):
        """
        Adds a model instance to the predictor.

        Args:
            model_name (str): Name of the model.
            model_instance (Model): Instance of the model to be added.
        """
        self.models[model_name] = model_instance

    def predict(self, model_name, input_sequence):
        """
        Predicts the output sequence using the specified model.

        Args:
            model_name (str): Name of the model to use for prediction.
            input_sequence (list or numpy array): Input sequence of floats with length equal to the model's input_length.

        Returns:
            output_sequence (list): Predicted output sequence of floats with length equal to the model's output_length.
        """
        if model_name not in self.models:
            raise ValueError("Model '{}' not found in the predictor.".format(model_name))
        
        model_instance = self.models[model_name]
        return model_instance.predict(input_sequence)