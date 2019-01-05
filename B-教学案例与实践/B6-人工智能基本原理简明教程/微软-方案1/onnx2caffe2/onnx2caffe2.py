import onnx
import numpy as np
model = onnx.load("model.onnx")
onnx.checker.check_model(model)
import onnx_caffe2.backend
prepared_backend = onnx_caffe2.backend.prepare(model)
from onnx_caffe2.backend import Caffe2Backend as c2
init_net, predict_net = c2.onnx_graph_to_caffe2_net(model.graph, device="CPU")
with open("squeeze_init_net.pb", "wb") as f:
    f.write(init_net.SerializeToString())
with open("squeeze_predict_net.pb", "wb") as f:
    f.write(predict_net.SerializeToString())
