import sys

sys.path.append("python/")
import needle as ndl

v1 = ndl.Tensor([0], dtype="float32")
v2 = ndl.exp(v1)
v3 = ndl.add_scalar(v2, 1)
v4 = v2 * v3
print(v4.inputs[0].inputs[0] is v1)
print(v4.op)
print(v4.inputs[0].op)
print(v3.op.__dict__)
print(type(v4.cached_data))


def print_node(node):
    print("id = %d" % id(node))
    print("inputs = ", [id(i) for i in node.inputs])
    print("op = %s" % type(node.op))
    print("data = %s" % node.cached_data)


print_node(v4)
