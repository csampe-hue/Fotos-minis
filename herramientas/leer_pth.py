"""Lee los pesos de un .pth de PyTorch sin PyTorch (sólo tensores float32)."""
import pickle, zipfile, collections, sys
import numpy as np

def cargar(ruta):
    z = zipfile.ZipFile(ruta)
    raiz = z.namelist()[0].split('/')[0]
    class Storage:
        def __init__(self, clave, dtype): self.clave, self.dtype = clave, dtype
    tipos = {'FloatStorage': np.float32, 'HalfStorage': np.float16, 'LongStorage': np.int64}
    def rebuild(storage, offset, size, stride, *args):
        buf = np.frombuffer(z.read(f'{raiz}/data/{storage.clave}'), dtype=storage.dtype)
        if not size: return buf[offset:offset+1].reshape(())
        st = [s * buf.itemsize for s in stride]
        return np.lib.stride_tricks.as_strided(buf[offset:], shape=size, strides=st).copy()
    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if name == '_rebuild_tensor_v2': return rebuild
            if name in tipos: return name
            if name == 'OrderedDict': return collections.OrderedDict
            raise pickle.UnpicklingError(f'{mod}.{name}')
        def persistent_load(self, pid):
            _, tipo, clave, loc, n = pid
            return Storage(clave, tipos[tipo])
    return U(__import__('io').BytesIO(z.read(f'{raiz}/data.pkl'))).load()

if __name__ == '__main__':
    d = cargar(sys.argv[1])
    print(type(d), list(d.keys())[:5] if isinstance(d, dict) else '')
    sd = d.get('params', d)
    for k, v in list(sd.items())[:6] + list(sd.items())[-4:]: print(k, v.shape, v.dtype)
    print(len(sd), 'tensores')
