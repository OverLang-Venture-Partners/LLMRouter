# LLMRouter 插件系统 - 完整使用指南

## 📋 概述

LLMRouter 现在支持**插件系统**，允许用户添加自定义 router 而**无需修改原有代码**。

### 核心优势

✅ **零侵入** - 不修改核心代码
✅ **自动发现** - 插件自动被发现和加载
✅ **统一接口** - 与内置 router 使用方式完全一致
✅ **灵活扩展** - 支持训练和推理

---

## 🚀 快速开始

### 1. 创建自定义 Router

```bash
# 创建目录
mkdir -p custom_routers/my_router

# 创建 router 文件
cat > custom_routers/my_router/router.py << 'EOF'
from llmrouter.models.meta_router import MetaRouter
import torch.nn as nn

class MyRouter(MetaRouter):
    """我的自定义路由器"""

    def __init__(self, yaml_path: str):
        # 创建模型（如果不需要可以用 nn.Identity()）
        model = nn.Identity()
        super().__init__(model=model, yaml_path=yaml_path)

        # 初始化
        self.llm_names = list(self.llm_data.keys())
        print(f"✅ MyRouter 加载了 {len(self.llm_names)} 个 LLM")

    def route_single(self, query_input: dict) -> dict:
        """路由单个查询"""
        # 你的路由逻辑
        query = query_input.get('query', '')

        # 示例：简单选择第一个模型
        selected = self.llm_names[0]

        return {
            "query": query,
            "model_name": selected,
            "predicted_llm": selected,
        }

    def route_batch(self, batch: list) -> list:
        """批量路由"""
        return [self.route_single(q) for q in batch]
EOF

# 创建配置文件
cat > custom_routers/my_router/config.yaml << 'EOF'
data_path:
  llm_data: 'data/example_data/llm_candidates/default_llm.json'

hparam:
  # 你的超参数

api_endpoint: 'https://integrate.api.nvidia.com/v1'
EOF
```

### 2. 使用自定义 Router

```bash
# 推理
llmrouter infer --router my_router \
  --config custom_routers/my_router/config.yaml \
  --query "What is AI?"

# 查看所有可用的 router（包括自定义）
llmrouter list-routers

# 仅路由（不调用 API）
llmrouter infer --router my_router \
  --config custom_routers/my_router/config.yaml \
  --query "Test query" \
  --route-only
```

---

## 🎯 核心组件

### 1. 插件系统核心 (`llmrouter/plugin_system.py`)

**功能：**
- 自动发现自定义 router
- 验证 router 实现
- 注册到系统中

**发现位置：**
1. `./custom_routers/` (项目目录)
2. `~/.llmrouter/plugins/` (用户目录)
3. `$LLMROUTER_PLUGINS` 环境变量

### 2. CLI 集成

**修改的文件：**
- `llmrouter/cli/router_inference.py` - 推理时加载插件
- `llmrouter/cli/router_train.py` - 训练时加载插件

**集成方式：**
```python
# 自动发现并注册插件
from llmrouter.plugin_system import discover_and_register_plugins

plugin_registry = discover_and_register_plugins(verbose=False)

# 注册到 ROUTER_REGISTRY
for router_name, router_class in plugin_registry.discovered_routers.items():
    ROUTER_REGISTRY[router_name] = router_class
```

### 3. 目录结构

```
LLMRouter/
├── llmrouter/
│   ├── plugin_system.py          # ⭐ 新增：插件系统核心
│   ├── cli/
│   │   ├── router_inference.py   # 🔧 修改：集成插件加载
│   │   └── router_train.py       # 🔧 修改：集成插件加载
│   └── models/
│       └── meta_router.py        # 基类
├── custom_routers/               # ⭐ 新增：自定义 router 目录
│   ├── __init__.py
│   ├── README.md                 # 使用说明
│   ├── randomrouter/             # 示例 1: 随机路由
│   │   ├── __init__.py
│   │   ├── router.py
│   │   ├── trainer.py
│   │   └── config.yaml
│   └── thresholdrouter/          # 示例 2: 基于难度路由
│       ├── __init__.py
│       ├── router.py
│       ├── trainer.py
│       └── config.yaml
└── docs/
    └── CUSTOM_ROUTERS.md         # ⭐ 新增：详细文档
```

---

## 📚 示例 Router

### 示例 1: RandomRouter（简单基线）

**位置：** `custom_routers/randomrouter/`

**功能：** 随机选择一个 LLM

**特点：**
- ✅ 最简单的实现
- ✅ 不需要训练
- ✅ 适合作为基线对比

**使用：**
```bash
llmrouter infer --router randomrouter \
  --config custom_routers/randomrouter/config.yaml \
  --query "Hello world" \
  --route-only
```

### 示例 2: ThresholdRouter（可训练）

**位置：** `custom_routers/thresholdrouter/`

**功能：** 基于查询难度估计进行路由
- 简单查询 → 小模型（便宜）
- 困难查询 → 大模型（能力强）

**特点：**
- ✅ 完整的训练流程
- ✅ 神经网络难度估计器
- ✅ 可配置阈值
- ✅ 支持自定义模型选择

**训练：**
```bash
llmrouter train --router thresholdrouter \
  --config custom_routers/thresholdrouter/config.yaml
```

**推理：**
```bash
llmrouter infer --router thresholdrouter \
  --config custom_routers/thresholdrouter/config.yaml \
  --query "Explain quantum entanglement"
```

---

## 🔧 实现要求

### 必须实现的方法

```python
class YourRouter(MetaRouter):
    def __init__(self, yaml_path: str):
        """初始化路由器"""
        model = ...  # 你的模型
        super().__init__(model=model, yaml_path=yaml_path)

    def route_single(self, query_input: dict) -> dict:
        """
        路由单个查询

        Args:
            query_input: {'query': '查询文本', ...}

        Returns:
            {'model_name': '选中的模型', ...}
        """
        pass

    def route_batch(self, batch: list) -> list:
        """
        批量路由

        Args:
            batch: [query_input1, query_input2, ...]

        Returns:
            [result1, result2, ...]
        """
        pass
```

### 可选：添加训练支持

```python
# trainer.py
from llmrouter.models.base_trainer import BaseTrainer

class YourRouterTrainer(BaseTrainer):
    def __init__(self, router, config: dict, device: str = "cpu"):
        super().__init__(router, config, device)
        # 初始化优化器等

    def train(self) -> None:
        """训练逻辑"""
        # 你的训练循环
        pass
```

---

## 💡 设计模式和最佳实践

### 1. 基于规则的路由

```python
def route_single(self, query_input):
    query = query_input['query'].lower()

    # 根据关键词路由
    if 'code' in query or 'program' in query:
        return {"model_name": "code-specialist-model"}

    # 根据长度路由
    elif len(query) < 50:
        return {"model_name": "small-fast-model"}

    else:
        return {"model_name": "large-capable-model"}
```

### 2. 基于成本的路由

```python
def route_single(self, query_input):
    difficulty = self._estimate_difficulty(query_input)

    # 选择能胜任且成本最低的模型
    for model_name, model_info in sorted(
        self.llm_data.items(),
        key=lambda x: x[1]['cost']
    ):
        if model_info['capability'] >= difficulty:
            return {"model_name": model_name}
```

### 3. 集成嵌入（Embedding）

```python
from llmrouter.utils import get_longformer_embedding

def route_single(self, query_input):
    query = query_input['query']

    # 生成嵌入
    embedding = get_longformer_embedding(query)

    # 使用嵌入进行路由
    selected = self._route_by_embedding(embedding)

    return {"model_name": selected}
```

### 4. 缓存优化

```python
class CachedRouter(MetaRouter):
    def __init__(self, yaml_path: str):
        super().__init__(...)
        self.cache = {}

    def route_single(self, query_input):
        query = query_input['query']

        # 检查缓存
        if query in self.cache:
            return self.cache[query]

        # 执行路由
        result = self._do_routing(query_input)

        # 存入缓存
        self.cache[query] = result
        return result
```

---

## 🐛 调试和故障排除

### 启用详细输出

```python
from llmrouter.plugin_system import discover_and_register_plugins

# 启用详细输出
registry = discover_and_register_plugins(
    plugin_dirs=['custom_routers'],
    verbose=True  # 显示发现过程
)

print(f"发现的 router: {registry.get_router_names()}")
```

### 常见问题

**问题 1: Router 未被发现**

```
Error: Unknown router: my_router
```

**解决方案：**
- ✅ 检查目录名与 router 名一致（小写）
- ✅ 确保 router 类名以 `Router` 结尾
- ✅ 验证 `custom_routers/` 目录存在
- ✅ 启用 `verbose=True` 查看详细日志

**问题 2: 导入错误**

```
ModuleNotFoundError: No module named 'xxx'
```

**解决方案：**
- ✅ 安装缺失的依赖
- ✅ 确保 `__init__.py` 文件存在
- ✅ 检查导入路径

**问题 3: 验证失败**

```
Router class validation failed
```

**解决方案：**
- ✅ 实现 `route_single` 和 `route_batch`
- ✅ 继承自 `MetaRouter`
- ✅ 方法签名正确

---

## 📊 完整示例流程

### Step 1: 创建 Router

```bash
mkdir -p custom_routers/smart_router
```

**router.py:**
```python
from llmrouter.models.meta_router import MetaRouter
import torch.nn as nn

class SmartRouter(MetaRouter):
    def __init__(self, yaml_path: str):
        model = nn.Identity()
        super().__init__(model=model, yaml_path=yaml_path)
        self.llm_names = list(self.llm_data.keys())

    def route_single(self, query_input: dict) -> dict:
        query = query_input['query']

        # 智能路由逻辑
        if len(query) > 100:
            selected = self.llm_names[-1]  # 长查询用大模型
        else:
            selected = self.llm_names[0]   # 短查询用小模型

        return {
            "query": query,
            "model_name": selected,
            "predicted_llm": selected,
            "reason": "length-based"
        }

    def route_batch(self, batch: list) -> list:
        return [self.route_single(q) for q in batch]
```

### Step 2: 创建配置

**config.yaml:**
```yaml
data_path:
  llm_data: 'data/example_data/llm_candidates/default_llm.json'

hparam:
  threshold_length: 100

api_endpoint: 'https://integrate.api.nvidia.com/v1'
```

### Step 3: 测试

```bash
# 测试路由决策
llmrouter infer --router smart_router \
  --config custom_routers/smart_router/config.yaml \
  --query "Short query" \
  --route-only

# 实际调用 LLM
llmrouter infer --router smart_router \
  --config custom_routers/smart_router/config.yaml \
  --query "This is a much longer query that should trigger routing to a more capable model..."
```

---

## 🎓 进阶主题

### 多轮路由

```python
class MultiRoundRouter(MetaRouter):
    def answer_query(self, query: str, return_intermediate: bool = False):
        # 分解查询
        sub_queries = self._decompose(query)

        # 每个子查询独立路由
        results = []
        for sq in sub_queries:
            routing = self.route_single({'query': sq})
            # 调用 API、聚合结果等
            results.append(routing)

        return self._aggregate(results)
```

### 共享工具函数

```python
# custom_routers/shared_utils.py
def preprocess_query(query):
    """查询预处理"""
    return query.strip().lower()

def compute_difficulty(query):
    """难度估计"""
    # 基于长度、复杂度等
    return len(query) / 100
```

### 环境变量配置

```bash
# 添加额外的插件目录
export LLMROUTER_PLUGINS="/path/to/plugins1:/path/to/plugins2"

# 使用
llmrouter infer --router my_custom_router ...
```

---

## 📖 相关文档

- **详细教程**: [docs/CUSTOM_ROUTERS.md](docs/CUSTOM_ROUTERS.md)
- **示例代码**: [custom_routers/README.md](custom_routers/README.md)
- **API 文档**: [llmrouter/plugin_system.py](llmrouter/plugin_system.py)

---

## 🤝 贡献和分享

如果你创建了有用的 router，欢迎：

1. 提交 Pull Request 添加到示例
2. 发布为独立 Python 包
3. 在社区分享经验

---

## ✅ 总结

**创建的核心文件：**
1. `llmrouter/plugin_system.py` - 插件系统核心
2. `llmrouter/cli/router_inference.py` - 集成到推理 CLI
3. `llmrouter/cli/router_train.py` - 集成到训练 CLI
4. `custom_routers/` - 自定义 router 目录
5. `docs/CUSTOM_ROUTERS.md` - 详细文档

**示例 Router：**
- `randomrouter` - 简单随机路由
- `thresholdrouter` - 基于难度的可训练路由

**使用流程：**
```bash
# 1. 创建 router
mkdir -p custom_routers/my_router
# 编写 router.py

# 2. 创建配置
# 编写 config.yaml

# 3. 使用
llmrouter infer --router my_router --config ... --query "..."
```

**优势：**
- ✅ 零侵入式扩展
- ✅ 自动发现和加载
- ✅ 与内置 router 使用方式完全一致
- ✅ 支持训练和推理

现在用户可以自由添加自定义 router，而无需修改任何原有代码！🎉
