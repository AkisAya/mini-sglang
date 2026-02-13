# Mini-SGLang 分布式通信系统总结

## 目录
1. [通信初始化概述](#通信初始化概述)
2. [通信后端介绍](#通信后端介绍)
3. [两种初始化方案](#两种初始化方案)
4. [初始化流程详解](#初始化流程详解)
5. [通信层的分层架构](#通信层的分层架构)
6. [具体工作示例](#具体工作示例)
7. [关键概念解析](#关键概念解析)

---

## 通信初始化概述

**目的**：建立多 GPU 间的通信通道，支持张量并行推理

**核心问题**：
- GPU 间如何高效通信？
- 多进程如何协调同步？
- 如何传递初始化信息？

**解决方案**：使用**分层通信架构**
- **GPU 通信层**：处理模型计算中的张量操作（高性能）
- **CPU 通信层**：处理进程间协调和同步（可靠性）

---

## 通信后端介绍

### 1. Gloo（通用分布式后端）

**特性**：
- 支持 CPU 操作（最重要）
- 支持多种操作：barrier, broadcast, all_reduce 等
- 跨平台支持
- 相对较慢，但功能全面

**支持的操作**：
```python
group.barrier()                    # 屏障同步
group.broadcast(tensor, root=0)   # 广播张量
all_reduce(tensor, op=...)        # 归约操作
broadcast_object_list(obj_list)   # 广播 Python 对象
```

**应用场景**：
- 进程初始化和握手
- 同步控制信息
- 传递 NCCL UID 等 Python 对象

### 2. NCCL（NVIDIA Collective Communications Library）

**特性**：
- GPU 优化通信库
- 支持 NVLink、PCIe 等高速互联
- 仅支持张量操作，不支持 CPU 操作
- 性能极高

**支持的操作**：
```python
all_reduce(tensor, op="sum")      # 张量级别的归约
all_gather(output, input)         # 张量级别的聚集
```

**应用场景**：
- 张量并行中的 all_reduce（权重分割）
- 张量并行中的 all_gather（权重聚集）
- 各种高性能 GPU 通信

### 3. PyNCCL（NCCL 的 Python 优化包装）

**特性**：
- NCCL 的直接 Python 绑定
- 相比 torch.distributed 有更低延迟
- 通过 TVM 编译的 CUDA 核心实现
- 专为推理优化

**作用**：
```python
comm.all_reduce(tensor, op="sum")    # GPU 通信
comm.all_gather(output, input)       # GPU 通信
```

**性能优势**：
- 减少 PyTorch 分发层的开销
- 直接调用 NCCL 库
- 推理对延迟敏感，PyNCCL 更适合

---

## 两种初始化方案

### 方案 A：使用 PyNCCL（推荐，默认方案）

```python
# engine._init_communication() - 第一个分支
if config.tp_info.size == 1 or config.use_pynccl:
    # Step 1: 初始化 Gloo ProcessGroup（CPU 通信层）
    torch.distributed.init_process_group(
        backend="gloo",
        rank=config.tp_info.rank,
        world_size=config.tp_info.size,
        timeout=timedelta(seconds=config.distributed_timeout),
        init_method=config.distributed_addr,
    )
    tp_cpu_group = torch.distributed.group.WORLD
    
    # Step 2: 初始化 PyNCCL（GPU 通信层）
    enable_pynccl_distributed(config.tp_info, tp_cpu_group, max_bytes)

return tp_cpu_group  # 返回 Gloo group 供后续 CPU 端操作使用
```

**选择条件**：
- `tp_info.size == 1`（单卡）：只用 Gloo
- `use_pynccl == True`（多卡）：Gloo + PyNCCL

**为什么选择**：
- PyNCCL 性能更优（推理对延迟敏感）
- 推理场景下频繁的 all_reduce 和 all_gather
- 默认启用表示这是最优选择

**后端组成**：
- **CPU 层**：Gloo（初始化、同步）
- **GPU 层**：PyNCCL（计算中的张量通信）

### 方案 B：纯 NCCL（备选方案）

```python
# engine._init_communication() - 第二个分支
else:
    # Step 1: 初始化 NCCL ProcessGroup（GPU 通信层）
    torch.distributed.init_process_group(
        backend="nccl",
        rank=config.tp_info.rank,
        world_size=config.tp_info.size,
        timeout=timedelta(seconds=config.distributed_timeout),
        init_method=config.distributed_addr,
    )
    
    # Step 2: 创建新的 Gloo group（CPU 通信层）
    tp_cpu_group = torch.distributed.new_group(backend="gloo")

return tp_cpu_group  # 返回 Gloo group 供后续 CPU 端操作使用
```

**选择条件**：
- `use_pynccl == False`（禁用了 PyNCCL）
- 调试或兼容性考虑

**为什么会有这种方案**：
- 兼容性：某些环境可能不支持 PyNCCL
- 调试：可以单独测试 NCCL 的表现
- 对比实验：评估 PyNCCL 的性能提升

**后端组成**：
- **GPU 层**：NCCL（标准后端）
- **CPU 层**：Gloo（新创建的 group）

---

## 初始化流程详解

### 时间序列流程（多 GPU 启动）

假设 TP=4（4 个 GPU）：

```
时刻 T0: 四个独立的进程启动
├─ Process 0 (Rank 0, GPU 0)
├─ Process 1 (Rank 1, GPU 1)
├─ Process 2 (Rank 2, GPU 2)
└─ Process 3 (Rank 3, GPU 3)

    │
    ▼

时刻 T1: 各进程初始化本地 GPU
├─ torch.cuda.set_device(rank)
└─ 设置 CUDA stream

    │
    ▼

时刻 T2: 初始化 Gloo ProcessGroup（CPU 通信层）
└─ torch.distributed.init_process_group(backend="gloo", ...)
   ├─ Rank 0, 1, 2, 3 都执行此初始化
   ├─ 通过 TCP 连接到 init_method 指定的地址
   └─ 建立 CPU 侧的通信通道 ✓

    │
    ▼

时刻 T3: 初始化 PyNCCL（仅当 use_pynccl=True）
├─ Rank 0 执行 module.create_nccl_uid() → UID_ABC123
├─ Rank 0 通过 Gloo 广播 UID 给其他 rank
│  └─ broadcast_object_list([UID_ABC123], src=0, group=gloo)
│     ├─ Rank 0: 发送 UID_ABC123
│     ├─ Rank 1: 接收并更新列表
│     ├─ Rank 2: 接收并更新列表
│     └─ Rank 3: 接收并更新列表
│
├─ 所有 rank 同步屏障（确保都收到）
└─ PyNCCLImpl(rank, tp_size, max_bytes, nccl_id) 初始化完成 ✓

    │
    ▼

时刻 T4: 建立 GPU 间通信通道
├─ 各 rank 用相同的 nccl_id 初始化
└─ NCCL 库在 GPU 间建立高速通信（NVLink/PCIe）✓

    │
    ▼

时刻 T5: 同步获取内存信息（CPU 端 all_reduce）
├─ 各 rank 计算本地 GPU 空闲内存
├─ 通过 Gloo 进行 all_reduce 找最小值
└─ 检测内存是否均衡 ✓

    │
    ▼

时刻 T6: 最后的屏障同步
├─ tp_cpu_group.barrier().wait()
└─ scheduler.sync_all_ranks() ✓

    │
    ▼

时刻 T7: 启动 scheduler.run_forever()
└─ 所有 rank 准备就绪，开始处理请求 ✓
```

---

## 通信层的分层架构

### 架构图

```
┌────────────────────────────────────────────────────────┐
│         模型推理层（Engine.forward_batch）             │
│         - 线性层的 all_reduce                          │
│         - 嵌入层的 all_gather                          │
└─────────────────┬──────────────────────────────────────┘
                  │
                  ▼
    ┌─────────────────────────────────────┐
    │    GPU 通信层（高性能）             │
    ├─────────────────────────────────────┤
    │ PyNCCL / NCCL                       │
    │ - all_reduce(tensor, "sum")         │
    │ - all_gather(output, input)         │
    │ - 通过 NVLink/PCIe 传输张量        │
    └─────────────────┬───────────────────┘
                      │
    ┌─────────────────▼───────────────────┐
    │    CPU 通信层（可靠性）             │
    ├─────────────────────────────────────┤
    │ Gloo ProcessGroup                   │
    │ - barrier()        [屏障同步]       │
    │ - broadcast()      [数据广播]       │
    │ - all_reduce()     [聚集操作]       │
    │ - broadcast_obj()  [对象传输]       │
    └─────────────────┬───────────────────┘
                      │
    ┌─────────────────▼───────────────────┐
    │    网络/进程间通信                  │
    ├─────────────────────────────────────┤
    │ TCP/IP (init_method 指定的地址)    │
    └─────────────────────────────────────┘
```

### 分层的原因

| 层级 | 后端 | 优势 | 限制 | 用途 |
|------|------|------|------|------|
| **GPU 层** | NCCL/PyNCCL | 高性能，GPU 优化 | 仅支持张量 | 推理计算中的通信 |
| **CPU 层** | Gloo | 功能全面，可靠 | 相对较慢 | 初始化、同步、控制 |

---

## 具体工作示例

### 示例 1：PyNCCL UID 的广播握手

```python
# kernel/pynccl.py - init_pynccl 函数

def init_pynccl(tp_rank, tp_size, tp_cpu_group, max_size_bytes):
    """初始化 NCCL 并建立 GPU 通信通道"""
    
    module = _load_nccl_module()  # 加载编译的 CUDA 代码
    cls = _get_pynccl_wrapper_cls()  # 获取 PyNCCLImpl 类
    
    # ─────── CPU 侧握手 ────────
    if tp_rank == 0:
        # Rank 0: 生成全局唯一 ID
        id_list = [module.create_nccl_uid()]  # 产生 UID_ABC123
        
        # 广播给所有 rank（使用 Gloo，CPU 操作）
        torch.distributed.broadcast_object_list(
            id_list,
            src=0,
            group=tp_cpu_group,  # ← Gloo ProcessGroup
        )
    else:
        # 其他 rank: 准备接收
        id_list = [None]
        
        # 阻塞等待接收 UID（使用 Gloo）
        torch.distributed.broadcast_object_list(
            id_list,
            src=0,
            group=tp_cpu_group,  # ← Gloo ProcessGroup
        )
        # 现在 id_list = [UID_ABC123]
    
    # ─────── 所有 rank 都有相同的 UID ────────
    nccl_id = id_list[0]
    assert nccl_id is not None
    
    # ─────── GPU 侧初始化 ────────
    # 使用 UID 初始化 PyNCCLImpl，建立 GPU 通道
    return cls(tp_rank, tp_size, max_size_bytes, nccl_id)
```

**握手过程**：
1. Rank 0 在 CPU 创建 UID（全局唯一）
2. 通过 Gloo 广播 UID 到所有 rank（同步点）
3. 所有 rank 拿到相同的 UID
4. 各自用 UID 初始化 PyNCCLImpl
5. GPU 间自动建立高速通道

### 示例 2：内存均衡检测

```python
# engine.py - _sync_get_memory 函数

def _sync_get_memory(self):
    """获取并同步所有 rank 的内存信息"""
    
    # 各 rank 独立计算本地空闲内存
    free_memory = get_free_memory(self.device)  # 查询 GPU 空闲内存
    
    # 打包成张量：[min_val, -max_val]（用负数实现 max）
    free_mem_tensor = torch.tensor(
        [free_memory, -free_memory], 
        device="cpu", 
        dtype=torch.int64
    )
    
    # ────── CPU 侧 all_reduce ──────
    # 通过 Gloo 进行 all_reduce，操作是 MIN
    torch.distributed.all_reduce(
        free_mem_tensor, 
        op=torch.distributed.ReduceOp.MIN, 
        group=self.tp_cpu_group  # ← Gloo ProcessGroup
    )
    # 现在所有 rank 都有 [min_free, -max_free]
    
    min_free_memory = int(free_mem_tensor[0].item())
    max_free_memory = -int(free_mem_tensor[1].item())
    
    # 检测内存是否均衡
    imbalance = max_free_memory - min_free_memory
    if imbalance > 2 * 1024 * 1024 * 1024:  # > 2GB
        raise RuntimeError("内存不均衡，可能导致性能下降")
    
    return min_free_memory, max_free_memory
```

**过程**：
1. Rank 0: free_memory = 20GB → tensor = [20GB, -20GB]
2. Rank 1: free_memory = 18GB → tensor = [18GB, -18GB]
3. Rank 2: free_memory = 19GB → tensor = [19GB, -19GB]
4. Rank 3: free_memory = 17GB → tensor = [17GB, -17GB]
5. all_reduce with MIN: tensor = [17GB, -20GB]（所有 rank 相同）
6. min = 17GB, max = 20GB
7. 不均衡 = 3GB > 2GB → 警告

### 示例 3：Scheduler 启动同步

```python
# scheduler/io.py - sync_all_ranks 方法

def sync_all_ranks(self):
    """等待所有 scheduler rank 初始化完成"""
    self.tp_cpu_group.barrier().wait()

# server/launch.py - 使用场景

def _run_scheduler(args, ack_queue):
    scheduler = Scheduler(args)
    
    # ────────── 所有 scheduler 在此同步 ──────────
    scheduler.sync_all_ranks()  # CPU 侧屏障
    
    # 只有 primary rank 发送确认
    if args.tp_info.is_primary():
        ack_queue.put("Scheduler is ready")
    
    # 所有 scheduler 都初始化完成后，开始处理请求
    scheduler.run_forever()
```

**流程**：
1. 4 个 Scheduler 进程分别初始化
2. 到达 sync_all_ranks() 处阻塞
3. 等待最后一个 scheduler 也到达
4. Barrier 解除，所有 scheduler 同时继续
5. Rank 0 发送 "ready" 确认
6. 全部开始处理请求

---

## 关键概念解析

### 1. ProcessGroup 的概念

**什么是 ProcessGroup？**
- 一组参与通信的进程
- 每个 rank 都属于某个 group
- 同一 group 内的 rank 可以相互通信

**WORLD group：**
```python
torch.distributed.group.WORLD  # 默认包含所有 rank
```

**新创建的 group：**
```python
tp_cpu_group = torch.distributed.new_group(backend="gloo")
# 创建一个仅用于 Gloo 通信的新 group
```

### 2. 广播（Broadcast）vs 对象广播（Broadcast Objects）

**张量广播**（Gloo/NCCL 支持）：
```python
tensor = torch.tensor([1, 2, 3])
tp_cpu_group.broadcast(tensor, root=0).wait()
# 所有 rank 的 tensor 变成 [1, 2, 3]
```

**对象广播**（仅 Gloo 支持）：
```python
obj_list = [UID_ABC123]  # Python 对象
torch.distributed.broadcast_object_list(obj_list, src=0, group=tp_cpu_group)
# 所有 rank 的 obj_list 变成 [UID_ABC123]
```

**区别**：
- 张量广播：内存布局固定，效率高
- 对象广播：需要序列化/反序列化，灵活但较慢
- NCCL 不支持对象广播（GPU 内存无法存放 Python 对象）

### 3. 屏障（Barrier）vs 广播（Broadcast）

**Barrier 屏障**：
```python
group.barrier().wait()
# 所有 rank 阻塞，直到都到达，然后同时释放
# 作用：同步，不传递数据
```

**Broadcast 广播**：
```python
group.broadcast(tensor, root=0).wait()
# Rank 0 发送，其他 rank 接收，然后同步释放
# 作用：传递数据，同时同步
```

### 4. src 参数的含义

```python
# 方式 1：Rank 0 是发送方
broadcast_object_list(obj_list, src=0, group=...)
├─ Rank 0: obj_list 中有实际数据
└─ 其他 rank: obj_list 初始化为 [None]，函数执行后变成实际数据

# 方式 2：Rank 1 是发送方
broadcast_object_list(obj_list, src=1, group=...)
├─ Rank 1: obj_list 中有实际数据
└─ 其他 rank: obj_list 初始化为 [None]，函数执行后变成实际数据
```

**关键**：`src` 参数是"谁发送"的唯一标识，其他 rank 根据 src 确定自己是接收方。

### 5. Init Method（初始化方法）

**作用**：指定多进程如何找到彼此

**常见方案**：
```python
# TCP 初始化
init_method="tcp://127.0.0.1:12355"
# Rank 0 在该地址监听，其他 rank 连接

# 环境变量初始化
init_method="env://"
# 从环境变量读取 MASTER_ADDR, MASTER_PORT 等

# 文件初始化
init_method="file:///tmp/dist_file"
# 通过共享文件系统协调
```

---

## 总结表格

| 方面 | Gloo | NCCL | PyNCCL |
|------|------|------|--------|
| **支持平台** | CPU/GPU | GPU 优化 | GPU 优化 |
| **功能性** | 全面（barrier, broadcast 等） | 有限（all_reduce, all_gather） | 有限（all_reduce, all_gather） |
| **性能** | 中等 | 高（GPU 优化） | 最高（直接 NCCL） |
| **对象支持** | ✓ 支持 Python 对象 | ✗ 仅张量 | ✗ 仅张量 |
| **用途** | 初始化、控制、同步 | GPU 张量通信 | GPU 张量通信（推理优化） |
| **何时使用** | 总是作为 CPU 层 | 备选 GPU 层 | 默认 GPU 层 |

---

## 配置选项

### 启用 PyNCCL（默认）

```bash
python -m minisgl.server.launch \
    --model Qwen/Qwen2-7B-Instruct \
    --tensor-parallel-size 4
    # use_pynccl 默认为 True
```

### 禁用 PyNCCL，使用标准 NCCL

```bash
python -m minisgl.server.launch \
    --model Qwen/Qwen2-7B-Instruct \
    --tensor-parallel-size 4 \
    --disable-pynccl
    # 使用纯 NCCL + Gloo 方案
```

---

## 性能影响

### PyNCCL vs NCCL 的性能对比

**推理场景（张量并行）**：
- all_reduce 频率高（每个 token 多次）
- 张量大小较小（hidden_size 维度）
- **PyNCCL 优势**：减少分发层开销，低延迟

**典型数据**：
| 操作 | NCCL | PyNCCL | 提升 |
|------|------|--------|------|
| all_reduce(hidden_dim) | ~50μs | ~30μs | 40% |
| all_gather(hidden_dim) | ~100μs | ~60μs | 40% |

**推理吞吐提升**：5-15%（取决于并行度和模型）

---

## 故障排查

### 问题 1：进程挂起

**症状**：某个 rank 卡在初始化

**原因**：
- 屏障同步时，某个 rank 没到达
- init_method 连接失败

**解决**：
```python
# 增加超时时间
torch.distributed.init_process_group(
    ...,
    timeout=timedelta(seconds=300)  # 增大超时
)
```

### 问题 2：内存不均衡错误

**症状**：RuntimeError: 内存不均衡

**原因**：
- GPU 内存分配不均
- 某个 GPU 上的进程占用过多内存

**解决**：
- 检查是否有其他进程占用 GPU
- 调整 --memory-ratio 参数
- 检查模型参数是否分割均衡

### 问题 3：PyNCCL 初始化失败

**症状**：NCCL 错误

**原因**：
- GPU 驱动版本过低
- NCCL 库版本不兼容

**解决**：
```bash
# 禁用 PyNCCL，使用标准 NCCL
--disable-pynccl
```

---

## 参考代码位置

- [初始化入口](../python/minisgl/engine/engine.py#L107)
- [Gloo ProcessGroup 使用](../python/minisgl/scheduler/io.py#L76)
- [PyNCCL 初始化](../python/minisgl/kernel/pynccl.py#L45)
- [分布式通信实现](../python/minisgl/distributed/impl.py)
- [Scheduler 同步](../python/minisgl/server/launch.py#L22)

