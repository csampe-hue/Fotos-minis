"""Convierte realesr-general-x4v3.pth (Real-ESRGAN, SRVGGNetCompact) a ONNX, sin PyTorch.

Uso: python herramientas/convertir_realesrgan.py realesr-general-x4v3.pth salida.onnx herramientas
Requiere el paquete «onnx» (sólo para convertir; la aplicación no lo necesita).
"""
import sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
sys.path.insert(0, sys.argv[3])
from leer_pth import cargar

sd = cargar(sys.argv[1])['params']
convs = sorted({int(k.split('.')[1]) for k in sd if k.endswith('.bias') and sd[k.replace('bias', 'weight')].ndim == 4})
nodos, inits = [], []
def ini(nombre, arr):
    inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=np.float32), nombre)); return nombre
actual = 'x'
for i in convs:
    w, b = sd[f'body.{i}.weight'], sd[f'body.{i}.bias']
    salida = f'c{i}'
    nodos.append(helper.make_node('Conv', [actual, ini(f'w{i}', w), ini(f'b{i}', b)], [salida],
                                  kernel_shape=[3, 3], pads=[1, 1, 1, 1]))
    actual = salida
    if f'body.{i + 1}.weight' in sd and sd[f'body.{i + 1}.weight'].ndim == 1:      # PReLU
        nodos.append(helper.make_node('PRelu', [actual, ini(f'p{i}', sd[f'body.{i + 1}.weight'].reshape(-1, 1, 1))], [f'a{i}']))
        actual = f'a{i}'
escala = int(round(np.sqrt(sd[f'body.{convs[-1]}.weight'].shape[0] / 3)))
nodos.append(helper.make_node('DepthToSpace', [actual], ['ps'], blocksize=escala, mode='CRD'))
nodos.append(helper.make_node('Resize', ['x', '', ini('escalas', np.array([1, 1, escala, escala], np.float32))], ['base'],
                              mode='nearest', coordinate_transformation_mode='asymmetric', nearest_mode='floor'))
nodos.append(helper.make_node('Add', ['ps', 'base'], ['y']))
g = helper.make_graph(nodos, 'realesr_general_x4v3',
                      [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3, 'h', 'w'])],
                      [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 3, 'H', 'W'])], inits)
m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 17)], producer_name='fotos-minis')
m.ir_version = 8
onnx.checker.check_model(m)
onnx.save(m, sys.argv[2])
print('escala', escala, 'convs', len(convs))
