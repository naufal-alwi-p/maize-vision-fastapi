from model.ml_output import InferenceResult

class InferenceResponse(InferenceResult):
    class_names: list[str]
    binary_class_names: list[str]
