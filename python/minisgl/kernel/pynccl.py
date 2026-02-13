from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal

from minisgl.env import ENV

from .utils import load_aot

if TYPE_CHECKING:
    from abc import abstractmethod

    import torch
    from tvm_ffi import Module

    class PyNCCLCommunicator:
        @abstractmethod
        def all_reduce(self, input: torch.Tensor, op: Literal["sum"]) -> None: ...
        @abstractmethod
        def all_gather(self, output: torch.Tensor, input: torch.Tensor) -> None: ...
        @abstractmethod
        def get_buffer(self) -> int: ...

else:
    PyNCCLCommunicator = Any


@lru_cache(maxsize=None)
def _load_nccl_module() -> Module:
    return load_aot("pynccl", cuda_files=["pynccl.cu"], extra_ldflags=["-lnccl"])


@lru_cache(maxsize=None)
def _get_pynccl_wrapper_cls():
    import tvm_ffi

    @tvm_ffi.register_object("minisgl.NCCLWrapper")
    class PyNCCLImpl(tvm_ffi.Object):
        def __init__(self, *args):
            self.__ffi_init__(*args)

    return PyNCCLImpl


def init_pynccl(
    *,
    tp_rank: int,
    tp_size: int,
    tp_cpu_group: torch.distributed.ProcessGroup,
    max_size_bytes: int = 0,
) -> PyNCCLCommunicator:
    """

Rank 0                          Rank 1, 2, 3...
┌─────────────────────┐         ┌──────────────────┐
│ create_nccl_uid()   │         │ Wait for UID     │
│ → generates UID_123 │         │                  │
└────────┬────────────┘         └──────────────────┘
         │
         │ broadcast_object_list via Gloo
         │
         ├────────────────────→ UID_123
         │
    All ranks receive same UID
         │
         ├─→ PyNCCLImpl(0, 4, ..., UID_123)
         ├─→ PyNCCLImpl(1, 4, ..., UID_123)
         ├─→ PyNCCLImpl(2, 4, ..., UID_123)
         └─→ PyNCCLImpl(3, 4, ..., UID_123)
         
    ↓ 建立 GPU 间通信通道（NCCL）
    """
    
    import torch

    max_size_bytes = min(max_size_bytes, ENV.PYNCCL_MAX_BUFFER_SIZE.value)

    module = _load_nccl_module()     # 加载编译的 pynccl.cu
    cls = _get_pynccl_wrapper_cls()  # 获取 PyNCCLImpl 类

    if tp_rank == 0:
        # NCCL 需要一个共享的唯一标识符来识别这个通信组
        # NCCL 还未初始化，无法用 GPU 通信
        # 必须用 CPU 端的 Gloo 来传递初始化信息
        id_list = [module.create_nccl_uid()]           
        torch.distributed.broadcast_object_list(
            id_list,
            src=0,                # ← "我是 Rank 0，我负责发送"
            group=tp_cpu_group,   # 通过 CPU 端 Gloo 广播
        )
        # 此时阻塞，等待所有 rank 收到
    else:
        id_list = [None]  # 接收方初始化为 [None] 作占位符
        torch.distributed.broadcast_object_list(
            id_list,
            src=0,                # ← "Rank 0 是发送方，我等待接收"
            group=tp_cpu_group,   # 通过 CPU 端 Gloo 接受广播内容
        )
        # 阻塞到收到数据为止

    nccl_id = id_list[0]
    assert not nccl_id is None, f"Failed to get NCCL unique ID on {tp_rank = }"

    # bypass type checking for the FFI object
    #  所有 rank 都拿到相同的 UID_ABC123
    # - 使用 nccl_id 进行 ncclGetUniqueId 反向操作
    # - GPU 间建立高速互联通道（NVLink/PCIe）
    # - 现在可以执行 all_reduce, all_gather 等操作
    return cls(tp_rank, tp_size, max_size_bytes, nccl_id)  # type: ignore
