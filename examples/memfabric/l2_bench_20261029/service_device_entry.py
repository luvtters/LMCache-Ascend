"""Select the service transport device before constructing the stock L2 adapter."""
import ctypes,json,sys
import torch
import torch_npu
from lmcache.cli.main import main

logical=int(sys.argv[1])
torch.npu.set_device(logical)
acl=ctypes.CDLL('libascendcl.so')
acl.aclrtGetDevice.argtypes=[ctypes.POINTER(ctypes.c_int)]
acl.aclrtGetDevice.restype=ctypes.c_int
current=ctypes.c_int(-1)
assert acl.aclrtGetDevice(ctypes.byref(current))==0 and current.value==logical
acl.aclrtGetPhyDevIdByLogicDevId.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_int)]
acl.aclrtGetPhyDevIdByLogicDevId.restype=ctypes.c_int
physical=ctypes.c_int(-1)
assert acl.aclrtGetPhyDevIdByLogicDevId(logical,ctypes.byref(physical))==0
assert logical==2 and physical.value==12
print('SERVICE_TRANSPORT_DEVICE '+json.dumps({'logical_device':logical,'physical_device':physical.value,
      'torch_current_device':torch.npu.current_device()}),flush=True)
main(sys.argv[2:])
