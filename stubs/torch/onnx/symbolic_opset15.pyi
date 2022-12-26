from torch.onnx import symbolic_helper as symbolic_helper


class Prim:
    domain: str
    @staticmethod
    def unchecked_cast(g, self): ...
