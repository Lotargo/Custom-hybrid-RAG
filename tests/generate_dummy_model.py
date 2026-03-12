import onnx
from onnx import helper
from onnx import TensorProto
import struct

# Create a very simple ONNX model: y = x (identity)
# Let's define the input and output shapes as [batch_size, seq_len] with type int64
X = helper.make_tensor_value_info('input_ids', TensorProto.INT64, [1, 'seq_len'])
Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 'seq_len'])

# Create a Cast node to convert INT64 to FLOAT
node_def = helper.make_node(
    'Cast', # node name
    ['input_ids'], # inputs
    ['output'], # outputs
    to=getattr(TensorProto, 'FLOAT') # attributes
)

# Create the graph
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [X],
    [Y]
)

# Create the model
model_def = helper.make_model(graph_def, producer_name='sonatadb-test')

# Save it to bytes
with open('dummy_model.onnx', 'wb') as f:
    f.write(model_def.SerializeToString())

print("Model saved to dummy_model.onnx")
