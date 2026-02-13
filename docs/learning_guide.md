# Mini-SGLang 详细学习指南

> 本指南将带你系统地学习 Mini-SGLang 的设计与实现，从整体架构到核心细节。  
> 适合希望深入理解 LLM 推理系统的开发者和研究人员。

---

## 📖 如何使用本指南

- **按顺序学习**：每个阶段都是基于前面的理解
- **动手实践**：阅读代码时建议在 IDE 中打开文件，配合调试器跟踪
- **做笔记**：记录关键概念和疑问点
- **运行测试**：可以用 `--shell` 模式运行系统，观察数据流

---

## 🎯 学习路径总览

```
第一阶段：整体架构理解
    └─> 第二阶段：核心数据结构
        └─> 第三阶段：启动流程
            └─> 第四阶段：调度器（核心）
                └─> 第五阶段：KV Cache 管理
                    └─> 第六阶段：模型实现
                        └─> 第七阶段：Attention Backend
                            └─> 第八阶段：推理引擎
                                └─> 第九阶段：分布式系统
                                    └─> 第十阶段：服务与应用
```

---

## 第一阶段：理解整体架构 🏗️

### 目标
建立对系统的宏观认识，了解各组件如何协同工作。

---

### 1.1 项目概述 - README.md

**文件路径**: `README.md`

**学习重点**:
- Mini-SGLang 的定位：轻量级、高性能 LLM 推理框架
- 核心特性：约 5000 行 Python 代码，但性能接近 SGLang
- 关键优化技术：
  - **Radix Cache**: 前缀共享，避免重复计算
  - **Chunked Prefill**: 将长 prompt 分块处理，降低显存峰值
  - **Overlap Scheduling**: CPU 调度与 GPU 计算并行
  - **Tensor Parallelism**: 多 GPU 并行推理
  - **CUDA Graph**: 减少 CPU 开销

**关键问题**:
- 为什么需要这些优化？
- Radix Cache 如何工作？
- 多 GPU 之间如何通信？

**建议**:
先快速浏览，了解项目能做什么，不必深究技术细节。

---

### 1.2 系统架构 - docs/structures.md

**文件路径**: `docs/structures.md`

**学习重点**:

#### 1.2.1 进程架构
Mini-SGLang 是一个多进程分布式系统：

```
用户请求
   ↓
[API Server]  ← FastAPI，提供 OpenAI 兼容 API
   ↓
[Tokenizer]   ← 文本 → Token
   ↓
[Scheduler 0] ← 调度器（TP Rank 0，主节点）
   ↓ (broadcast)
[Scheduler 1, 2, ..., N-1] ← 其他 GPU 的调度器
   ↓
[Engine]      ← 每个 Scheduler 管理一个 Engine（模型推理）
   ↓
[Detokenizer] ← Token → 文本
   ↓
[API Server]  → 返回给用户
```

#### 1.2.2 通信机制
- **ZeroMQ (ZMQ)**: 进程间控制消息（轻量级）
- **NCCL/Gloo**: GPU 间张量通信（高性能）

#### 1.2.3 模块职责
- `minisgl.core`: 核心数据结构（Req, Batch, Context）
- `minisgl.scheduler`: 调度逻辑（Prefill + Decode）
- `minisgl.engine`: 推理引擎（模型执行）
- `minisgl.kvcache`: KV Cache 管理
- `minisgl.attention`: Attention 计算后端
- `minisgl.models`: LLM 模型实现
- `minisgl.server`: API 服务器
- `minisgl.distributed`: 分布式通信

**关键问题**:
- 为什么要分离 Tokenizer 和 Detokenizer？
- Scheduler 和 Engine 的区别是什么？
- 为什么 Rank 0 是主节点？

**动手实践**:
```bash
# 启动一个简单的服务，观察进程
python -m minisgl --model Qwen/Qwen3-0.6B --shell

# 另一个终端查看进程
ps aux | grep minisgl
```

---

### 1.3 功能特性 - docs/features.md

**文件路径**: `docs/features.md`

**学习重点**:
- **Chunked Prefill**: 长文本分块处理，控制显存
- **Attention Backend**: FlashAttention (Prefill) + FlashInfer (Decode)
- **CUDA Graph**: Decode 阶段捕获计算图，减少启动开销
- **Radix Cache**: 共享前缀，提升吞吐量
- **Overlap Scheduling**: CPU 和 GPU 流水线并行

**思考**:
- 为什么 Prefill 和 Decode 用不同的 Attention Backend？
- CUDA Graph 为什么只能用于 Decode？

---

### 1.4 分布式通信 - notes/distributed_communication_summary.md

**文件路径**: `notes/distributed_communication_summary.md`

**学习重点**:
- **Gloo**: CPU 通信，用于进程初始化和同步
- **NCCL**: GPU 通信，用于张量并行（All-Reduce、All-Gather）
- 初始化流程：Master 端口 → Gloo 握手 → NCCL UID 广播 → NCCL 初始化

**关键概念**:
```python
# Gloo：同步信息
dist.barrier()  # 等待所有进程

# NCCL：张量操作
all_reduce(tensor)   # 求和归约
all_gather(tensors)  # 收集所有 GPU 的张量
```

**思考**:
- 为什么需要两套通信机制？
- 如何保证多进程启动顺序？

---

## 第二阶段：核心数据结构 📊

### 目标
理解系统中数据如何表示和流转。

---

### 2.1 核心数据类 - python/minisgl/core.py

**文件路径**: `python/minisgl/core.py`

**学习重点**:

#### 2.1.1 Req（请求）
```python
@dataclass
class Req:
    input_ids: torch.Tensor    # 输入 token（CPU）
    table_idx: int             # Page Table 的行索引
    cached_len: int            # 已缓存的 token 数量
    output_len: int            # 需要生成的 token 数量
    uid: int                   # 全局唯一 ID
    sampling_params: SamplingParams  # 采样参数
    cache_handle: BaseCacheHandle    # 缓存句柄
```

**关键属性**:
- `device_len`: 当前已处理的 token 数（包括输入和已生成）
- `max_device_len`: 最大长度 = input_len + output_len
- `cached_len`: Radix Cache 命中的长度
- `extend_len`: 本次需要处理的长度 = device_len - cached_len

**生命周期**:
```
新请求（cached_len = 0）
   ↓ Prefill（处理 extend_len 个 token）
   ↓ cached_len = device_len
   ↓ Decode（每次生成 1 个 token）
   ↓ device_len += 1
   ↓ 重复 Decode，直到 device_len == max_device_len
完成
```

#### 2.1.2 Batch（批次）
```python
@dataclass
class Batch:
    reqs: List[Req]            # 批次中的请求
    phase: Literal["prefill", "decode"]  # 阶段
    input_ids: torch.Tensor    # 拼接的输入 token（GPU）
    positions: torch.Tensor    # 每个 token 的位置
    out_loc: torch.Tensor      # 输出位置（用于写回 KV Cache）
    attn_metadata: BaseAttnMetadata  # Attention 后端的元数据
```

**Prefill vs Decode**:
| 属性 | Prefill | Decode |
|------|---------|--------|
| 每个请求处理的 token 数 | 多个（extend_len） | 1 个 |
| input_ids 形状 | `[sum(extend_len)]` | `[batch_size]` |
| Attention 类型 | Causal Attention | Decoder Attention |

#### 2.1.3 Context（全局上下文）
```python
@dataclass
class Context:
    device: torch.device
    dtype: torch.dtype
    phase: Literal["prefill", "decode"]
    attn_backend: BaseAttnBackend
    attn_metadata: BaseAttnMetadata
    moe_backend: BaseMoeBackend | None
```

**作用**:
- 全局单例，避免函数参数传递
- 在 Forward 前设置，Layer 内部读取

#### 2.1.4 SamplingParams（采样参数）
```python
@dataclass
class SamplingParams:
    temperature: float = 0.0   # 温度（0 = 贪婪采样）
    top_k: int = -1            # Top-K 采样
    top_p: float = 1.0         # Top-P 采样
    ignore_eos: bool = False   # 是否忽略 EOS
    max_tokens: int = 1024     # 最大生成长度
```

**思考**:
- 为什么 Req 在 CPU，Batch 在 GPU？
- Context 如何实现全局访问？

**动手实践**:
```python
# 在 shell 中查看 Req 的变化
# 设置断点在 scheduler.py 的 step() 方法
```

---

### 2.2 消息定义 - python/minisgl/message/

**文件路径**: 
- `python/minisgl/message/frontend.py`（用户请求相关）
- `python/minisgl/message/backend.py`（内部调度相关）

**学习重点**:

#### 2.2.1 Frontend 消息（API Server ↔ Tokenizer）
```python
class GenerateReqMsg:
    """用户的生成请求"""
    text: str | List[int]      # 输入文本或 token
    sampling_params: SamplingParams
    rid: str                   # 请求 ID

class GenerateRespMsg:
    """生成的响应"""
    text: str                  # 生成的文本
    rid: str
    finish_reason: FinishReason
```

#### 2.2.2 Backend 消息（Tokenizer ↔ Scheduler ↔ Detokenizer）
```python
class AddReqMsg:
    """添加新请求到调度器"""
    input_ids: torch.Tensor
    sampling_params: SamplingParams
    uid: int

class BatchBackendMsg:
    """批量处理的请求和 token"""
    uids: List[int]
    next_tokens: torch.Tensor

class ExitMsg:
    """退出信号"""
    pass
```

**通信流程**:
```
用户 → GenerateReqMsg → API Server → Tokenizer
    → AddReqMsg → Scheduler → Engine
    → BatchBackendMsg → Detokenizer
    → GenerateRespMsg → API Server → 用户
```

**思考**:
- 为什么需要两层消息系统？
- 如何支持流式输出？

---

## 第三阶段：入口与启动流程 🚀

### 目标
理解系统如何启动，各进程如何初始化。

---

### 3.1 程序入口 - python/minisgl/__main__.py

**文件路径**: `python/minisgl/__main__.py`

**学习重点**:
```python
from .server import launch_server

assert __name__ == "__main__"

launch_server()
```

非常简洁，直接调用 `launch_server()`。

---

### 3.2 启动流程 - python/minisgl/server/launch.py

**文件路径**: `python/minisgl/server/launch.py`

**学习重点**:

#### 3.2.1 启动步骤
```python
def launch_server(run_shell: bool = False):
    # 1. 解析命令行参数
    server_args, run_shell = parse_args(sys.argv[1:], run_shell)
    
    # 2. 设置 multiprocessing 启动方式为 "spawn"
    mp.set_start_method("spawn", force=True)
    
    # 3. 启动 Scheduler 子进程（每个 GPU 一个）
    for i in range(world_size):
        new_args = replace(server_args, tp_info=DistributedInfo(i, world_size))
        mp.Process(target=_run_scheduler, args=(new_args, ack_queue)).start()
    
    # 4. 启动 Tokenizer 子进程
    mp.Process(target=tokenize_worker, kwargs={...}).start()
    
    # 5. 等待所有子进程就绪
    for _ in range(world_size + num_tokenizers):
        ack_queue.get()
    
    # 6. 启动 API Server（主进程）或 Shell 模式
    if run_shell:
        run_shell_mode(...)
    else:
        run_api_server(...)
```

#### 3.2.2 为什么用 "spawn"？
- **fork**: 复制父进程的内存（包括 CUDA 状态），可能导致冲突
- **spawn**: 完全独立的进程，避免 CUDA 初始化问题

#### 3.2.3 子进程启动顺序
1. **Scheduler**: 初始化模型、CUDA、NCCL
2. **Tokenizer**: 加载 tokenizer，等待请求
3. **API Server**: 提供 HTTP 接口

**关键代码** - `_run_scheduler`:
```python
def _run_scheduler(args: ServerArgs, ack_queue: mp.Queue[str]):
    scheduler = Scheduler(args)
    scheduler.sync_all_ranks()  # 同步所有 TP Rank
    
    if args.tp_info.is_primary():
        ack_queue.put("Scheduler is ready")  # 通知主进程
    
    scheduler.run_forever()  # 进入事件循环
```

**思考**:
- 为什么 Scheduler 要先启动？
- 如何保证多 GPU 同步？

**动手实践**:
```bash
# 单 GPU 启动
python -m minisgl --model Qwen/Qwen3-0.6B --shell

# 多 GPU 启动（2 卡）
python -m minisgl --model Qwen/Qwen3-0.6B --tp 2
```

---

### 3.3 命令行参数 - python/minisgl/server/args.py

**文件路径**: `python/minisgl/server/args.py`

**学习重点**:

#### 3.3.1 关键参数
```python
@dataclass
class ServerArgs:
    # 模型相关
    model_path: str              # HuggingFace 模型路径
    dtype: str = "auto"          # 数据类型（bfloat16/float16）
    
    # 调度相关
    max_running_req: int = 2048  # 最大并发请求数
    max_extend_tokens: int = 8192  # Prefill 预算
    max_prefill_length: int = 8192  # Chunked Prefill 块大小
    
    # 缓存相关
    cache_type: str = "radix"    # radix 或 naive
    mem_fraction: float = 0.88   # 显存使用比例
    
    # 分布式
    tp_info: DistributedInfo     # TP 信息
    
    # Attention Backend
    attention_backend: str = "fa,fi"  # Prefill, Decode
    
    # CUDA Graph
    cuda_graph_max_bs: int = 128  # 最大批次大小
```

#### 3.3.2 参数影响
- `max_extend_tokens`: 影响 Prefill 吞吐量
- `max_prefill_length`: 影响显存峰值
- `mem_fraction`: 显存利用率

**思考**:
- 如何调优这些参数？
- 不同场景（长文本/短对话）如何配置？

---

## 第四阶段：Scheduler（调度器 - 核心）⚡

### 目标
理解系统的核心调度逻辑，Prefill 和 Decode 如何协同。

---

### 4.1 调度器配置 - python/minisgl/scheduler/config.py

**文件路径**: `python/minisgl/scheduler/config.py`

**学习重点**:
```python
@dataclass
class SchedulerConfig:
    model_path: str
    model_config: ModelConfig
    dtype: torch.dtype
    
    # 资源限制
    max_running_req: int          # 最大并发请求
    max_extend_tokens: int        # Prefill 预算
    max_prefill_length: int       # 单次 Prefill 最大长度
    
    # 缓存策略
    cache_type: Literal["naive", "radix"]
    mem_fraction: float
    
    # 分布式
    tp_info: DistributedInfo
    
    # Backend
    attention_backend: AttentionBackend
    moe_backend: str | None
```

**作用**:
将命令行参数转换为 Scheduler 和 Engine 的配置。

---

### 4.2 IO 管理 - python/minisgl/scheduler/io.py

**文件路径**: `python/minisgl/scheduler/io.py`

**学习重点**:

#### 4.2.1 SchedulerIOMixin
```python
class SchedulerIOMixin:
    def __init__(self, config: SchedulerConfig, tp_cpu_group):
        self.tp_info = config.tp_info
        self.tp_cpu_group = tp_cpu_group
        
        if self.tp_info.is_primary():
            # Rank 0 接收 Tokenizer 消息
            self.ctx_tokenizer = zmq.Context()
            self.sock_tokenizer = self.ctx_tokenizer.socket(zmq.PULL)
            ...
            
            # Rank 0 发送 Detokenizer 消息
            self.sock_detokenizer = ...
```

#### 4.2.2 消息处理
```python
def recv_requests(self) -> List[AddReqMsg]:
    """从 Tokenizer 接收新请求"""
    if self.tp_info.is_primary():
        msgs = []
        while self.sock_tokenizer.poll(0):  # 非阻塞
            msgs.append(AddReqMsg.deserialize(...))
        return msgs
    return []

def send_to_detokenizer(self, msg: BatchBackendMsg):
    """发送 token 到 Detokenizer"""
    if self.tp_info.is_primary():
        self.sock_detokenizer.send_pyobj(msg)
```

#### 4.2.3 多 Rank 同步
```python
def sync_new_requests(self, reqs: List[AddReqMsg]) -> List[AddReqMsg]:
    """Rank 0 广播新请求到其他 Rank"""
    obj_list = [reqs if self.tp_info.is_primary() else None]
    torch.distributed.broadcast_object_list(
        obj_list, src=0, group=self.tp_cpu_group
    )
    return obj_list[0]
```

**思考**:
- 为什么只有 Rank 0 负责 IO？
- 如何保证所有 Rank 看到相同的请求？

---

### 4.3 Prefill 管理 - python/minisgl/scheduler/prefill.py

**文件路径**: `python/minisgl/scheduler/prefill.py`

**学习重点**:

#### 4.3.1 Chunked Prefill
```python
@dataclass
class ChunkedReq:
    """分块的请求"""
    req: Req
    begin_loc: int   # 本次处理的起始位置
    extend_len: int  # 本次处理的长度
```

**为什么需要分块？**
- 长文本一次性 Prefill 会占用大量显存
- 分块后可以和 Decode 交替执行，提升响应速度

#### 4.3.2 PrefillManager
```python
class PrefillManager:
    def schedule_prefill(
        self, 
        budget: int,  # 可用的 token 预算
        new_reqs: List[Req]
    ) -> Batch | None:
        # 1. 尝试调度新请求
        for req in new_reqs:
            if budget >= req.extend_len:
                chunked = ChunkedReq(req, 0, req.extend_len)
                budget -= req.extend_len
                scheduled.append(chunked)
        
        # 2. 尝试继续未完成的请求（Chunked）
        for chunked in self.waiting_chunked:
            remain = chunked.req.extend_len - chunked.begin_loc
            if budget >= remain:
                chunked.extend_len = remain
                budget -= remain
                scheduled.append(chunked)
        
        # 3. 构建 Batch
        return self._build_batch(scheduled)
```

**调度策略**:
- 优先处理新请求（降低首 token 延迟）
- 再处理分块的请求（公平性）

**思考**:
- 如果预算不足，如何处理长请求？
- Chunked Prefill 对 Radix Cache 有什么影响？

---

### 4.4 Decode 管理 - python/minisgl/scheduler/decode.py

**文件路径**: `python/minisgl/scheduler/decode.py`

**学习重点**:

#### 4.4.1 DecodeManager
```python
class DecodeManager:
    def __init__(self):
        self.running_reqs: List[Req] = []  # 正在 Decode 的请求
    
    def schedule_decode(self) -> Batch | None:
        """调度所有 Decode 请求"""
        if not self.running_reqs:
            return None
        
        # 构建 Decode Batch
        input_ids = [req.input_ids[req.device_len - 1] for req in self.running_reqs]
        positions = [req.device_len - 1 for req in self.running_reqs]
        
        return Batch(
            reqs=self.running_reqs,
            phase="decode",
            input_ids=torch.tensor(input_ids),
            positions=torch.tensor(positions),
            ...
        )
```

**Decode 特点**:
- 每个请求只处理 1 个 token（最后生成的）
- 批次中的所有请求并行处理
- 使用 CUDA Graph 优化

**思考**:
- Decode 为什么更容易批处理？
- CUDA Graph 如何捕获？

---

### 4.5 调度器主逻辑 - python/minisgl/scheduler/scheduler.py

**文件路径**: `python/minisgl/scheduler/scheduler.py`

**学习重点**:

#### 4.5.1 Scheduler 初始化
```python
class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        # 1. 初始化 Engine
        self.engine = Engine(config)
        
        # 2. 初始化 IO（ZMQ Socket）
        super().__init__(config, self.engine.tp_cpu_group)
        
        # 3. 创建独立的 CUDA Stream（Overlap Scheduling）
        self.stream = torch.cuda.Stream()
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        
        # 4. 初始化各个 Manager
        self.table_manager = TableManager(...)
        self.cache_manager = CacheManager(...)
        self.decode_manager = DecodeManager()
        self.prefill_manager = PrefillManager(...)
```

#### 4.5.2 主循环 - Overlap Scheduling
```python
def run_forever(self):
    last_data = None
    ongoing_data = None
    
    while True:
        # 1. 处理上一次 Forward 的结果（在 self.stream）
        if last_data is not None:
            self._process_last_data(last_data, ongoing_data)
        
        # 2. 接收新请求（Rank 0）
        new_reqs = self.recv_requests()
        new_reqs = self.sync_new_requests(new_reqs)
        
        # 3. 调度 Prefill 或 Decode
        batch = self.prefill_manager.schedule_prefill(...)
        if batch is None:
            batch = self.decode_manager.schedule_decode()
        
        if batch is None:
            continue
        
        # 4. 等待 Engine 完成（上一次的 Forward）
        if ongoing_data is not None:
            self.engine.stream.synchronize()
        
        # 5. 提交新的 Forward（在 engine.stream）
        with self.engine_stream_ctx:
            forward_output = self.engine.forward(batch)
        
        # 6. 更新 last_data 和 ongoing_data
        last_data = ongoing_data
        ongoing_data = (batch, forward_output)
```

#### 4.5.3 Overlap Scheduling 图解
```
时间线:
    ┌──────────────┬──────────────┬──────────────┐
    │   Cycle N    │  Cycle N+1   │  Cycle N+2   │
GPU │              │              │              │
    │ Forward N    │ Forward N+1  │ Forward N+2  │
    │              │              │              │
CPU │              │              │              │
    │   Process    │   Process    │   Process    │
    │   Result N-1 │   Result N   │   Result N+1 │
    └──────────────┴──────────────┴──────────────┘
         ↑ 并行 ↑       ↑ 并行 ↑
```

**关键优化**:
- CPU 处理结果（更新缓存、发送 token）和 GPU Forward 并行
- 使用两个 CUDA Stream 实现流水线

**思考**:
- 为什么需要两个 Stream？
- `last_data` 和 `ongoing_data` 的作用？

---

## 第五阶段：KV Cache 管理 💾

### 目标
理解 KV Cache 的组织方式和 Radix Cache 的优化原理。

---

### 5.1 Cache 接口 - python/minisgl/kvcache/base.py

**文件路径**: `python/minisgl/kvcache/base.py`

**学习重点**:

#### 5.1.1 BaseCacheHandle
```python
class BaseCacheHandle:
    """单个请求的 Cache 句柄"""
    def alloc_token(self) -> None:
        """分配一个 token 的缓存"""
        pass
    
    def free_cache(self) -> None:
        """释放缓存"""
        pass
    
    @property
    def page_indices(self) -> torch.Tensor:
        """返回该请求在 Page Table 中的页索引"""
        pass
```

#### 5.1.2 BaseCacheManager
```python
class BaseCacheManager:
    def alloc(self, key: torch.Tensor) -> tuple[BaseCacheHandle, int]:
        """
        分配缓存
        返回：(句柄, 命中长度)
        """
        pass
    
    def free(self, handle: BaseCacheHandle) -> None:
        """释放缓存"""
        pass
    
    @property
    def available_size(self) -> int:
        """可用的 token 数量"""
        pass
```

**思考**:
- 为什么需要 Handle？
- 命中长度是什么？

---

### 5.2 简单缓存 - python/minisgl/kvcache/naive_manager.py

**文件路径**: `python/minisgl/kvcache/naive_manager.py`

**学习重点**:

#### 5.2.1 NaiveCacheManager
```python
class NaiveCacheManager(BaseCacheManager):
    """简单的缓存管理器，不共享前缀"""
    def alloc(self, key: torch.Tensor) -> tuple[BaseCacheHandle, int]:
        needed = len(key)
        if self.available_size < needed:
            raise ValueError("Out of memory")
        
        pages = self._alloc_pages(needed)
        handle = NaiveCacheHandle(pages)
        return handle, 0  # 无命中
```

**特点**:
- 每个请求独立分配
- 无前缀共享

---

### 5.3 Radix Cache - python/minisgl/kvcache/radix_manager.py

**文件路径**: `python/minisgl/kvcache/radix_manager.py`

**学习重点**:

#### 5.3.1 RadixTreeNode
```python
class RadixTreeNode:
    """Radix Tree 的节点"""
    def __init__(self):
        self.children: Dict[int, RadixTreeNode] = {}  # token → 子节点
        self._parent: RadixTreeNode | None = None
        self.ref_count: int = 0  # 引用计数
        self._key: torch.Tensor  # 该节点存储的 token 序列
        self._value: torch.Tensor  # 对应的 page 索引
```

**数据结构示例**:
```
输入序列：[1, 2, 3], [1, 2, 4], [1, 5]

Radix Tree:
       Root
        │
      [1] (page 0)
       ├─ [2] (page 1)
       │   ├─ [3] (page 2)
       │   └─ [4] (page 3)
       └─ [5] (page 4)
```

#### 5.3.2 RadixCacheManager
```python
class RadixCacheManager(BaseCacheManager):
    def alloc(self, key: torch.Tensor) -> tuple[RadixCacheHandle, int]:
        # 1. 在 Radix Tree 中查找最长前缀
        node, match_len = self._match_prefix(self.root, key)
        
        # 2. 如果完全匹配，直接返回
        if match_len == len(key):
            node.ref_count += 1
            return RadixCacheHandle(node), match_len
        
        # 3. 分配新的页
        remain_key = key[match_len:]
        new_pages = self._alloc_pages(len(remain_key))
        new_node = RadixTreeNode()
        new_node.set_key_value(remain_key, new_pages)
        new_node.set_parent(node)
        
        return RadixCacheHandle(new_node), match_len
```

**关键优化**:
- **前缀共享**: 相同前缀只存储一次
- **引用计数**: 自动管理缓存生命周期
- **LRU 淘汰**: 显存不足时淘汰最久未使用的节点

**场景示例**:
```python
# 多轮对话
req1: "你好，我是小明"  → cache: [你, 好, ，, 我, 是, 小, 明]
req2: "你好，我是小红"  → 复用 [你, 好, ，, 我, 是, 小]，新增 [红]
req3: "你好，今天天气"  → 复用 [你, 好, ，]，新增 [今, 天, 天, 气]
```

**思考**:
- Radix Cache 如何处理分支？
- 如何避免缓存碎片？

**动手实践**:
```python
# 运行 benchmark，对比 naive 和 radix
python benchmark/offline/bench.py --cache naive
python benchmark/offline/bench.py --cache radix
```

---

### 5.4 MHA KV Cache - python/minisgl/kvcache/mha_pool.py

**文件路径**: `python/minisgl/kvcache/mha_pool.py`

**学习重点**:

#### 5.4.1 MHAKVCache
```python
class MHAKVCache:
    """Multi-Head Attention 的 KV Cache Pool"""
    def __init__(self, model_config: ModelConfig, num_pages: int, ...):
        self.k_cache = torch.zeros(
            (num_layers, num_pages, page_size, num_kv_heads, head_dim),
            dtype=dtype, device=device
        )
        self.v_cache = torch.zeros(...)
```

**存储布局**:
```
k_cache[layer][page][token][head][dim]
       └─┬─┘ └─┬─┘ └─┬──┘ └─┬─┘ └┬┘
         │      │      │      │     │
     Layer  Page  Token  Head  Dim
```

**Page Table 映射**:
```
page_table[req_idx][token_idx] = page_id
k_cache[layer][page_id] = K 值
```

**思考**:
- 为什么用 Page Table 而不是直接索引？
- Page 大小如何选择？

---

## 第六阶段：模型实现 🤖

### 目标
理解 LLM 模型的实现，特别是如何支持 Tensor Parallelism。

---

### 6.1 Layer 基类 - python/minisgl/layers/base.py

**文件路径**: `python/minisgl/layers/base.py`

**学习重点**:

#### 6.1.1 BaseTPLayer
```python
class BaseTPLayer(nn.Module):
    """支持 Tensor Parallelism 的 Layer 基类"""
    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """从 HuggingFace 权重中加载并切分"""
        pass
```

**TP 切分策略**:
- **Column Parallel**: 沿输出维度切分（如 q_proj, k_proj）
- **Row Parallel**: 沿输入维度切分（如 o_proj）

**示例**:
```python
# Column Parallel (Linear: [hidden, hidden * 4])
# GPU 0: [hidden, hidden * 2]
# GPU 1: [hidden, hidden * 2]

# Row Parallel (Linear: [hidden * 4, hidden])
# GPU 0: [hidden * 2, hidden]
# GPU 1: [hidden * 2, hidden]
# 输出需要 all_reduce
```

---

### 6.2 基础层实现

#### 6.2.1 Linear - python/minisgl/layers/linear.py

**文件路径**: `python/minisgl/layers/linear.py`

**学习重点**:
```python
class ColumnParallelLinear(BaseTPLayer):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 每个 GPU 只处理部分列
        return F.linear(x, self.weight, self.bias)

class RowParallelLinear(BaseTPLayer):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        # 需要 all_reduce 合并结果
        if self.tp_info.size > 1:
            all_reduce(out)
        return out
```

#### 6.2.2 Norm - python/minisgl/layers/norm.py

**文件路径**: `python/minisgl/layers/norm.py`

**学习重点**:
```python
class RMSNorm(BaseTPLayer):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMSNorm 不需要切分
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight
```

#### 6.2.3 Embedding - python/minisgl/layers/embedding.py

**文件路径**: `python/minisgl/layers/embedding.py`

**学习重点**:
```python
class VocabParallelEmbedding(BaseTPLayer):
    """词表并行 Embedding"""
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # 每个 GPU 只负责部分词表
        # 需要 all_reduce
        return F.embedding(input_ids, self.weight)
```

---

### 6.3 Attention Layer - python/minisgl/layers/attention.py

**文件路径**: `python/minisgl/layers/attention.py`

**学习重点**:

#### 6.3.1 AttentionLayer
```python
class AttentionLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        self.q_proj = ColumnParallelLinear(...)
        self.k_proj = ColumnParallelLinear(...)
        self.v_proj = ColumnParallelLinear(...)
        self.o_proj = RowParallelLinear(...)
        self.rotary = RotaryEmbedding(...)
    
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # 1. QKV 投影
        q = self.q_proj(hidden)
        k = self.k_proj(hidden)
        v = self.v_proj(hidden)
        
        # 2. RoPE
        q, k = self.rotary(q, k)
        
        # 3. Attention 计算（调用 Backend）
        ctx = get_global_ctx()
        attn_out = ctx.attn_backend(q, k, v, ctx.attn_metadata)
        
        # 4. 输出投影
        out = self.o_proj(attn_out)
        return out
```

**思考**:
- QKV 为什么用 Column Parallel？
- O_proj 为什么需要 all_reduce？

---

### 6.4 完整模型 - python/minisgl/models/llama.py

**文件路径**: `python/minisgl/models/llama.py`

**学习重点**:

#### 6.4.1 LlamaModel
```python
class LlamaModel(BaseModel):
    def __init__(self, config: ModelConfig):
        self.embed = VocabParallelEmbedding(...)
        self.layers = nn.ModuleList([
            LlamaLayer(config) for _ in range(config.num_layers)
        ])
        self.norm = RMSNorm(...)
        self.lm_head = ColumnParallelLinear(...)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # 1. Embedding
        hidden = self.embed(input_ids)
        
        # 2. Transformer Layers
        for layer in self.layers:
            hidden = layer(hidden)
        
        # 3. Norm + LM Head
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)
        
        return logits
```

#### 6.4.2 LlamaLayer
```python
class LlamaLayer(nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # 1. Attention
        residual = hidden
        hidden = self.attn_norm(hidden)
        hidden = self.attn(hidden)
        hidden = residual + hidden
        
        # 2. FFN
        residual = hidden
        hidden = self.ffn_norm(hidden)
        hidden = self.ffn(hidden)
        hidden = residual + hidden
        
        return hidden
```

**思考**:
- 为什么 Norm 在 Attention 之前？（Pre-Norm vs Post-Norm）
- Residual 连接对 TP 有什么影响？

---

### 6.5 权重加载 - python/minisgl/models/weight.py

**文件路径**: `python/minisgl/models/weight.py`

**学习重点**:

#### 6.5.1 load_hf_weight
```python
def load_hf_weight(model_path: str, tp_rank: int, tp_size: int) -> Dict:
    """加载 HuggingFace 权重并切分"""
    # 1. 加载权重
    state_dict = {}
    for shard_file in glob(f"{model_path}/*.safetensors"):
        with safe_open(shard_file, framework="pt") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    
    # 2. 转换为 Mini-SGLang 的格式
    converted = {}
    for key, tensor in state_dict.items():
        # 根据 Layer 类型切分
        if "q_proj" in key or "k_proj" in key:
            # Column Parallel
            tensor = shard_column(tensor, tp_rank, tp_size)
        elif "o_proj" in key:
            # Row Parallel
            tensor = shard_row(tensor, tp_rank, tp_size)
        converted[key] = tensor
    
    return converted
```

**思考**:
- 如何处理 GQA（Grouped Query Attention）？
- 如何处理 MoE 模型？

---

## 第七阶段：Attention Backend（性能关键）⚡

### 目标
理解高性能 Attention 实现，FlashAttention 和 FlashInfer 的差异。

---

### 7.1 Attention 接口 - python/minisgl/attention/base.py

**文件路径**: `python/minisgl/attention/base.py`

**学习重点**:

#### 7.1.1 BaseAttnBackend
```python
class BaseAttnBackend:
    def init_prefill_metadata(self, batch: Batch) -> BaseAttnMetadata:
        """初始化 Prefill 阶段的元数据"""
        pass
    
    def init_decode_metadata(self, batch: Batch) -> BaseAttnMetadata:
        """初始化 Decode 阶段的元数据"""
        pass
    
    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        metadata: BaseAttnMetadata,
    ) -> torch.Tensor:
        """执行 Attention 计算"""
        pass
```

#### 7.1.2 BaseAttnMetadata
```python
class BaseAttnMetadata:
    """Attention 元数据（每个 Backend 实现不同）"""
    pass
```

**为什么需要 Metadata？**
- Attention 需要知道每个请求的长度、位置
- 不同 Backend 需要不同的数据格式
- 在 Scheduler 中准备好，避免 Forward 时计算

---

### 7.2 FlashAttention - python/minisgl/attention/fa.py

**文件路径**: `python/minisgl/attention/fa.py`

**学习重点**:

#### 7.2.1 FAAttnBackend
```python
class FAAttnBackend(BaseAttnBackend):
    def __call__(self, q, k, v, metadata: FAAttnMetadata):
        from flash_attn import flash_attn_varlen_func
        
        return flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=metadata.cu_seqlens_q,
            cu_seqlens_k=metadata.cu_seqlens_k,
            max_seqlen_q=metadata.max_seqlen_q,
            max_seqlen_k=metadata.max_seqlen_k,
            causal=True,
        )
```

#### 7.2.2 FAAttnMetadata
```python
@dataclass
class FAAttnMetadata:
    cu_seqlens_q: torch.Tensor  # Cumulative sequence lengths for query
    cu_seqlens_k: torch.Tensor  # Cumulative sequence lengths for key
    max_seqlen_q: int           # Maximum query sequence length
    max_seqlen_k: int           # Maximum key sequence length
```

**示例**:
```python
# Batch: [req1: 3 tokens, req2: 5 tokens, req3: 2 tokens]
cu_seqlens_q = [0, 3, 8, 10]  # Cumulative sum
max_seqlen_q = 5
```

**FlashAttention 特点**:
- 使用 HBM → SRAM 的分块策略
- 降低显存访问次数
- 适合 Prefill（长序列）

---

### 7.3 FlashInfer - python/minisgl/attention/fi.py

**文件路径**: `python/minisgl/attention/fi.py`

**学习重点**:

#### 7.3.1 FIAttnBackend
```python
class FIAttnBackend(BaseAttnBackend):
    def __init__(self, ...):
        # 创建 Wrapper（提前准备元数据）
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(...)
        self.decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(...)
    
    def __call__(self, q, k, v, metadata: FIAttnMetadata):
        if metadata.is_prefill:
            return self.prefill_wrapper.forward(q, self.kv_cache, ...)
        else:
            return self.decode_wrapper.forward(q, self.kv_cache, ...)
```

**FlashInfer 特点**:
- 专门优化 Paged KV Cache
- Decode 阶段性能更好（通过 Wrapper 预处理）
- 支持 CUDA Graph

**Prefill vs Decode**:
| 特性 | Prefill | Decode |
|------|---------|--------|
| Query 长度 | 长 | 1 |
| KV Cache 长度 | 短 | 长 |
| 计算特点 | Compute-bound | Memory-bound |
| 最佳 Backend | FlashAttention | FlashInfer |

**思考**:
- 为什么 Decode 是 Memory-bound？
- 如何在运行时切换 Backend？

---

### 7.4 Attention Utils - python/minisgl/attention/utils.py

**文件路径**: `python/minisgl/attention/utils.py`

**学习重点**:

#### 7.4.1 create_attention_backend
```python
def create_attention_backend(
    backend_str: str,  # "fa" or "fi" or "fa,fi"
    model_config: ModelConfig,
    kv_cache: MHAKVCache,
    page_table: torch.Tensor,
) -> tuple[BaseAttnBackend, BaseAttnBackend]:
    """创建 Prefill 和 Decode 的 Backend"""
    prefill_name, decode_name = backend_str.split(",")
    
    prefill_backend = _create_backend(prefill_name, ...)
    decode_backend = _create_backend(decode_name, ...)
    
    return prefill_backend, decode_backend
```

**常见组合**:
- `fa,fi`: Prefill 用 FA，Decode 用 FI（推荐）
- `fa,fa`: 全部用 FA
- `fi,fi`: 全部用 FI

---

## 第八阶段：推理引擎 Engine 🚀

### 目标
理解 Engine 如何整合模型、Attention、CUDA Graph，执行实际推理。

---

### 8.1 Engine 配置 - python/minisgl/engine/config.py

**文件路径**: `python/minisgl/engine/config.py`

**学习重点**:
```python
@dataclass
class EngineConfig:
    model_path: str
    model_config: ModelConfig
    dtype: torch.dtype
    
    # 资源配置
    max_running_req: int
    max_seq_len: int
    mem_fraction: float
    
    # Attention Backend
    attention_backend: tuple[str, str]  # (prefill, decode)
    
    # MoE Backend
    moe_backend: str | None
    
    # CUDA Graph
    cuda_graph_max_bs: int
    
    # 分布式
    tp_info: DistributedInfo
```

---

### 8.2 采样逻辑 - python/minisgl/engine/sample.py

**文件路径**: `python/minisgl/engine/sample.py`

**学习重点**:

#### 8.2.1 Sampler
```python
class Sampler:
    def __call__(
        self,
        logits: torch.Tensor,  # [batch_size, vocab_size]
        sampling_params: List[SamplingParams],
    ) -> torch.Tensor:
        """采样下一个 token"""
        # 1. 提取每个请求的 logits
        # 2. 根据 temperature, top_k, top_p 采样
        # 3. 返回 next_tokens [batch_size]
        
        for i, params in enumerate(sampling_params):
            if params.is_greedy:
                next_token = logits[i].argmax()
            else:
                probs = self._apply_temperature(logits[i], params.temperature)
                probs = self._apply_top_k(probs, params.top_k)
                probs = self._apply_top_p(probs, params.top_p)
                next_token = torch.multinomial(probs, 1)
            
            next_tokens.append(next_token)
        
        return torch.tensor(next_tokens)
```

#### 8.2.2 采样策略
```python
def _apply_temperature(logits, temperature):
    """温度采样：temperature 越大，分布越平滑"""
    return torch.softmax(logits / temperature, dim=-1)

def _apply_top_k(probs, k):
    """Top-K 采样：只保留概率最高的 k 个 token"""
    top_k_probs, top_k_indices = torch.topk(probs, k)
    return top_k_probs / top_k_probs.sum()

def _apply_top_p(probs, p):
    """Top-P (Nucleus) 采样：保留累积概率 > p 的 token"""
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    mask = cumsum <= p
    return sorted_probs[mask]
```

**思考**:
- temperature = 0 和 temperature = 1 的区别？
- Top-K 和 Top-P 哪个更好？

---

### 8.3 CUDA Graph - python/minisgl/engine/graph.py

**文件路径**: `python/minisgl/engine/graph.py`

**学习重点**:

#### 8.3.1 GraphRunner
```python
class GraphRunner:
    def __init__(self, engine, max_batch_sizes: List[int]):
        self.graphs: Dict[int, torch.cuda.CUDAGraph] = {}
        self.input_buffers: Dict[int, torch.Tensor] = {}
        self.output_buffers: Dict[int, torch.Tensor] = {}
        
        # 预先捕获不同 batch size 的 CUDA Graph
        for bs in max_batch_sizes:
            self._capture_graph(engine, bs)
    
    def _capture_graph(self, engine, batch_size):
        """捕获 CUDA Graph"""
        # 1. 准备输入（固定形状）
        input_ids = torch.zeros(batch_size, dtype=torch.long, device=engine.device)
        positions = torch.arange(batch_size, device=engine.device)
        
        # 2. Warmup（JIT 编译）
        for _ in range(3):
            _ = engine.model(input_ids, positions)
        
        # 3. 捕获
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = engine.model(input_ids, positions)
        
        # 4. 保存
        self.graphs[batch_size] = graph
        self.input_buffers[batch_size] = input_ids
        self.output_buffers[batch_size] = output
    
    def forward(self, input_ids, positions, batch_size):
        """重放 CUDA Graph"""
        # 更新输入 buffer
        self.input_buffers[batch_size].copy_(input_ids)
        
        # 重放
        self.graphs[batch_size].replay()
        
        # 返回输出
        return self.output_buffers[batch_size]
```

**CUDA Graph 优势**:
- 减少 CPU → GPU 的命令提交开销
- Decode 阶段提升 10-20% 吞吐量

**限制**:
- 输入形状必须固定
- 只能用于 Decode（Prefill 形状不固定）

**思考**:
- 如果 batch_size 不在预设列表中怎么办？
- 如何选择捕获哪些 batch_size？

---

### 8.4 Engine 主逻辑 - python/minisgl/engine/engine.py

**文件路径**: `python/minisgl/engine/engine.py`

**学习重点**:

#### 8.4.1 Engine 初始化
```python
class Engine:
    def __init__(self, config: EngineConfig):
        # 1. 设置设备和 TP 信息
        self.device = torch.device(f"cuda:{config.tp_info.rank}")
        torch.cuda.set_device(self.device)
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)
        
        # 2. 初始化分布式通信
        self.tp_cpu_group = self._init_communication(config)
        
        # 3. 加载模型
        self.model = create_model(config.model_config)
        self.model.load_state_dict(self._load_weight_state_dict(config))
        
        # 4. 分配 KV Cache
        self.num_pages = self._determine_num_pages(config)
        self.kv_cache = create_kvcache(
            model_config=config.model_config,
            num_pages=self.num_pages + 1,  # +1 for dummy page
            device=self.device,
            dtype=self.dtype,
        )
        
        # 5. 创建 Page Table
        self.page_table = create_page_table(
            (config.max_running_req + 1, self.max_seq_len),
            device=self.device,
        )
        
        # 6. 初始化 Attention Backend
        self.attn_backend = create_attention_backend(
            config.attention_backend,
            config.model_config,
            self.kv_cache,
            self.page_table,
        )
        
        # 7. 初始化 Sampler 和 CUDA Graph
        self.sampler = Sampler()
        self.graph_runner = GraphRunner(self, config.cuda_graph_max_bs)
```

#### 8.4.2 Forward
```python
def forward(self, batch: Batch) -> ForwardOutput:
    """执行一次 Forward"""
    # 1. 设置全局 Context
    set_global_ctx(Context(
        device=self.device,
        dtype=self.dtype,
        phase=batch.phase,
        attn_backend=self.attn_backend,
        attn_metadata=batch.attn_metadata,
        moe_backend=self.moe_backend,
    ))
    
    # 2. 模型推理
    if batch.phase == "decode" and self._can_use_cuda_graph(batch):
        logits = self.graph_runner.forward(batch.input_ids, batch.positions, len(batch.reqs))
    else:
        logits = self.model(batch.input_ids, batch.positions)
    
    # 3. 采样
    next_tokens_gpu = self.sampler(
        logits[batch.out_loc],  # 只对输出位置采样
        [req.sampling_params for req in batch.reqs],
    )
    
    # 4. 异步拷贝到 CPU
    next_tokens_cpu = torch.empty_like(next_tokens_gpu, device="cpu", pin_memory=True)
    next_tokens_cpu.copy_(next_tokens_gpu, non_blocking=True)
    copy_done_event = torch.cuda.Event()
    copy_done_event.record()
    
    return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)
```

**关键优化**:
- 使用 CUDA Graph 优化 Decode
- 异步拷贝 token 到 CPU（Overlap）
- Context 避免参数传递

**思考**:
- 为什么要拷贝到 CPU？
- copy_done_event 的作用？

---

## 第九阶段：分布式系统 🌐

### 目标
理解 Tensor Parallelism 的实现，多 GPU 如何协同。

---

### 9.1 分布式接口 - python/minisgl/distributed/impl.py

**文件路径**: `python/minisgl/distributed/impl.py`

**学习重点**:

#### 9.1.1 集合通信操作
```python
def all_reduce(tensor: torch.Tensor, op=dist.ReduceOp.SUM):
    """All-Reduce：所有 GPU 求和，结果广播到所有 GPU"""
    if get_tp_info().size > 1:
        dist.all_reduce(tensor, op=op, group=get_tp_group())

def all_gather(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """All-Gather：收集所有 GPU 的 tensor，拼接"""
    if get_tp_info().size == 1:
        return tensor
    
    world_size = get_tp_info().size
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor, group=get_tp_group())
    return torch.cat(gathered, dim=dim)
```

#### 9.1.2 使用场景
```python
# Row Parallel Linear
class RowParallelLinear(BaseTPLayer):
    def forward(self, x):
        out = F.linear(x, self.weight)  # 每个 GPU 计算部分
        all_reduce(out)  # 合并结果
        return out

# Vocab Parallel Embedding
class VocabParallelEmbedding(BaseTPLayer):
    def forward(self, input_ids):
        # GPU 0: vocab [0, vocab_size/2]
        # GPU 1: vocab [vocab_size/2, vocab_size]
        local_ids = input_ids - self.vocab_start
        mask = (local_ids >= 0) & (local_ids < self.vocab_size_per_rank)
        out = F.embedding(local_ids, self.weight)
        out = out * mask.unsqueeze(-1)  # 屏蔽不属于本 GPU 的 ID
        all_reduce(out)  # 合并
        return out
```

**思考**:
- All-Reduce 和 All-Gather 的区别？
- 如何最小化通信开销？

---

### 9.2 分布式信息 - python/minisgl/distributed/info.py

**文件路径**: `python/minisgl/distributed/info.py`

**学习重点**:
```python
@dataclass
class DistributedInfo:
    rank: int        # 当前进程的 Rank
    size: int        # 总进程数（World Size）
    
    def is_primary(self) -> bool:
        """是否是主 Rank（Rank 0）"""
        return self.rank == 0
    
    def shard_dim(self, dim: int, axis: int = 0) -> tuple[int, int]:
        """计算切分后的维度"""
        assert dim % self.size == 0
        per_rank = dim // self.size
        start = self.rank * per_rank
        return start, start + per_rank
```

**应用**:
```python
# 切分 Linear 权重（Column Parallel）
start, end = tp_info.shard_dim(weight.shape[0])
self.weight = weight[start:end].contiguous()
```

---

### 9.3 张量并行切分 - notes/tensor_parallelism_sharding.md

**文件路径**: `notes/tensor_parallelism_sharding.md`

**学习重点**:

#### 9.3.1 Transformer Layer 的切分
```
Input (Hidden State)
   ↓
┌──────────────────────────────┐
│   Attention                  │
│                              │
│  Q_proj (Column Parallel)    │  GPU 0: heads [0, n/2]
│  K_proj (Column Parallel)    │  GPU 1: heads [n/2, n]
│  V_proj (Column Parallel)    │
│                              │
│  Attention Compute (Local)   │
│                              │
│  O_proj (Row Parallel)       │  需要 all_reduce
└──────────────────────────────┘
   ↓ (Residual)
┌──────────────────────────────┐
│   FFN                        │
│                              │
│  Gate & Up (Column Parallel) │  GPU 0: [0, hidden/2]
│  Activation (Local)          │  GPU 1: [hidden/2, hidden]
│  Down (Row Parallel)         │  需要 all_reduce
└──────────────────────────────┘
   ↓ (Residual)
Output
```

#### 9.3.2 通信次数
每个 Transformer Layer:
- Attention: 1 次 all_reduce（O_proj）
- FFN: 1 次 all_reduce（Down_proj）
- **总计**: 2 次通信 / Layer

**优化思路**:
- 通信与计算 Overlap
- 使用 NCCL 的高性能实现

---

## 第十阶段：服务与应用 🌍

### 目标
理解如何对外提供服务，以及如何使用 Mini-SGLang。

---

### 10.1 API Server - python/minisgl/server/api_server.py

**文件路径**: `python/minisgl/server/api_server.py`

**学习重点**:

#### 10.1.1 FastAPI 服务
```python
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI 兼容的 Chat Completion API"""
    # 1. 提取用户消息
    text = extract_text_from_messages(request.messages)
    
    # 2. 构建采样参数
    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
    )
    
    # 3. 发送到 Tokenizer
    req_msg = GenerateReqMsg(text, sampling_params, rid)
    zmq_socket.send_pyobj(req_msg)
    
    # 4. 流式返回结果
    async def generate():
        async for resp_msg in receive_responses(rid):
            yield format_openai_response(resp_msg)
    
    if request.stream:
        return StreamingResponse(generate())
    else:
        full_response = await collect_full_response(rid)
        return full_response
```

#### 10.1.2 支持的 Endpoint
- `/v1/chat/completions`: Chat 接口
- `/v1/completions`: Completion 接口
- `/health`: 健康检查
- `/models`: 模型列表

**客户端示例**:
```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[
        {"role": "user", "content": "你好"}
    ],
    stream=True,
)

for chunk in response:
    print(chunk.choices[0].delta.content, end="")
```

---

### 10.2 Tokenizer Worker - python/minisgl/tokenizer/

**文件路径**: `python/minisgl/tokenizer/tokenize.py`, `detokenize.py`

**学习重点**:

#### 10.2.1 Tokenize Worker
```python
def tokenize_worker(tokenizer_path, addr, backend_addr, ...):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # 接收来自 API Server 的请求
    sock_frontend = zmq_socket(zmq.PULL, addr)
    
    # 发送到 Scheduler
    sock_backend = zmq_socket(zmq.PUSH, backend_addr)
    
    while True:
        # 1. 接收请求
        req = GenerateReqMsg.deserialize(sock_frontend.recv())
        
        # 2. Tokenize
        input_ids = tokenizer.encode(req.text)
        
        # 3. 发送到 Scheduler
        add_req = AddReqMsg(
            input_ids=torch.tensor(input_ids),
            sampling_params=req.sampling_params,
            uid=req.uid,
        )
        sock_backend.send_pyobj(add_req)
```

#### 10.2.2 Detokenize Worker
```python
def detokenize_worker(tokenizer_path, backend_addr, frontend_addr):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # 接收来自 Scheduler 的 token
    sock_backend = zmq_socket(zmq.PULL, backend_addr)
    
    # 发送到 API Server
    sock_frontend = zmq_socket(zmq.PUSH, frontend_addr)
    
    while True:
        # 1. 接收 token
        batch_msg = BatchBackendMsg.deserialize(sock_backend.recv())
        
        # 2. Detokenize
        for uid, token in zip(batch_msg.uids, batch_msg.next_tokens):
            text = tokenizer.decode([token])
            
            # 3. 发送到 API Server
            resp = GenerateRespMsg(text, rid, finish_reason)
            sock_frontend.send_pyobj(resp)
```

**思考**:
- 为什么要分离 Tokenizer 和 Detokenizer？
- 如何处理流式输出？

---

### 10.3 Python API - python/minisgl/llm/llm.py

**文件路径**: `python/minisgl/llm/llm.py`

**学习重点**:

#### 10.3.1 LLM 类
```python
class LLM:
    """Python API 接口"""
    def __init__(self, model: str, **kwargs):
        # 启动后台服务
        self.server_process = launch_server_subprocess(model, **kwargs)
        
        # 创建 OpenAI 客户端
        self.client = openai.OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="EMPTY",
        )
    
    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        """批量生成"""
        results = []
        for prompt in prompts:
            response = self.client.completions.create(
                model=self.model,
                prompt=prompt,
                **kwargs,
            )
            results.append(response.choices[0].text)
        return results
    
    def chat(self, messages: List[Dict], **kwargs) -> str:
        """Chat 接口"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message.content
```

**使用示例**:
```python
from minisgl import LLM

llm = LLM(model="Qwen/Qwen3-0.6B")

# 1. 简单生成
outputs = llm.generate(["你好", "介绍一下 Python"])

# 2. Chat
response = llm.chat([
    {"role": "user", "content": "你好"}
])
```

---

### 10.4 Benchmark - benchmark/

**文件路径**: `benchmark/offline/bench.py`, `benchmark/online/bench_simple.py`

**学习重点**:

#### 10.4.1 离线 Benchmark
```python
# benchmark/offline/bench.py
def run_offline_bench(model, num_prompts, input_len, output_len):
    """测试离线吞吐量"""
    from minisgl import LLM
    
    llm = LLM(model=model)
    
    # 生成测试数据
    prompts = ["dummy " * input_len] * num_prompts
    
    # 测试
    start = time.time()
    outputs = llm.generate(prompts, max_tokens=output_len)
    duration = time.time() - start
    
    # 计算吞吐量
    total_tokens = num_prompts * output_len
    throughput = total_tokens / duration
    print(f"Throughput: {throughput:.2f} tokens/s")
```

#### 10.4.2 在线 Benchmark
```python
# benchmark/online/bench_simple.py
async def run_online_bench(model, qps, duration):
    """测试在线延迟"""
    import aiohttp
    
    async def send_request():
        start = time.time()
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:8000/v1/chat/completions",
                json={"messages": [...], "stream": False},
            ) as resp:
                result = await resp.json()
        latency = time.time() - start
        return latency
    
    # 以固定 QPS 发送请求
    latencies = []
    for _ in range(int(qps * duration)):
        latency = await send_request()
        latencies.append(latency)
        await asyncio.sleep(1.0 / qps)
    
    # 统计
    print(f"P50: {np.percentile(latencies, 50):.3f}s")
    print(f"P99: {np.percentile(latencies, 99):.3f}s")
```

**运行**:
```bash
# 离线吞吐量测试
python benchmark/offline/bench.py --model Qwen/Qwen3-0.6B

# 在线延迟测试
python benchmark/online/bench_simple.py --qps 10
```

---

## 📝 附录：学习资源

### 相关论文
1. **FlashAttention**: [Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
2. **SGLang**: [Efficient LLM Serving with Radix Attention](https://arxiv.org/abs/2312.07104)
3. **Sarathi-Serve**: [Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills](https://arxiv.org/abs/2403.02310)
4. **PagedAttention**: [Efficient Memory Management for LLM Serving](https://arxiv.org/abs/2309.06180)

### 相关项目
- [SGLang](https://github.com/sgl-project/sglang): 原始项目
- [vLLM](https://github.com/vllm-project/vllm): 高性能 LLM 推理框架
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer): 高性能 Attention 库
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM): NVIDIA 官方推理框架

### 调试技巧
```bash
# 1. 使用 shell 模式观察系统行为
python -m minisgl --model Qwen/Qwen3-0.6B --shell

# 2. 设置日志级别
export MINISGL_LOG_LEVEL=DEBUG

# 3. 使用 PyTorch Profiler
python -m minisgl --model ... --profile

# 4. 查看显存使用
watch -n 1 nvidia-smi
```

---

## 🎓 学习检查清单

完成每个阶段后，确保你能回答以下问题：

### 第一阶段
- [ ] Mini-SGLang 的核心组件有哪些？
- [ ] Radix Cache 的工作原理？
- [ ] 为什么需要 Overlap Scheduling？

### 第二阶段
- [ ] Req 和 Batch 的区别？
- [ ] cached_len 和 device_len 的含义？
- [ ] Context 如何实现全局访问？

### 第三阶段
- [ ] 系统启动顺序是什么？
- [ ] 为什么用 "spawn" 而不是 "fork"？
- [ ] Scheduler 如何初始化？

### 第四阶段
- [ ] Prefill 和 Decode 如何调度？
- [ ] Chunked Prefill 的优势？
- [ ] Overlap Scheduling 如何工作？

### 第五阶段
- [ ] Radix Tree 的数据结构？
- [ ] 如何处理前缀共享？
- [ ] LRU 淘汰策略的实现？

### 第六阶段
- [ ] Column Parallel 和 Row Parallel 的区别？
- [ ] 哪些层需要 all_reduce？
- [ ] 如何加载和切分权重？

### 第七阶段
- [ ] FlashAttention 和 FlashInfer 的区别？
- [ ] 为什么 Decode 用 FlashInfer 更好？
- [ ] Attention Metadata 的作用？

### 第八阶段
- [ ] Engine 的初始化流程？
- [ ] CUDA Graph 如何工作？
- [ ] Forward 的优化技巧？

### 第九阶段
- [ ] All-Reduce 和 All-Gather 的区别？
- [ ] 如何最小化通信开销？
- [ ] Transformer Layer 的切分方式？

### 第十阶段
- [ ] 如何提供 OpenAI 兼容 API？
- [ ] Tokenizer 和 Detokenizer 的职责？
- [ ] 如何进行性能测试？

---

## 🚀 进阶方向

完成学习后，可以尝试：

1. **实现新功能**
   - 支持新的模型架构（如 MoE）
   - 实现 Speculative Decoding
   - 添加 Multi-LoRA 支持

2. **性能优化**
   - 优化 Radix Cache 的查找速度
   - 实现更激进的 Overlap Scheduling
   - 优化通信开销

3. **系统扩展**
   - 支持 Pipeline Parallelism
   - 实现动态批处理
   - 添加监控和可视化

4. **研究方向**
   - 研究不同调度策略的影响
   - 分析 Radix Cache 的命中率
   - 探索新的 Attention 优化方法

---

**祝你学习愉快！如有疑问，欢迎提 Issue 讨论。** 🎉
