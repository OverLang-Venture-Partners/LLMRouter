# Custom Router 插件系统 - 实现总结

## 🎯 实现目标

让用户可以**添加自定义 router 而不修改原有代码结构**。

## ✅ 已完成的工作

### 1. 核心插件系统

**新增文件：** `llmrouter/plugin_system.py`

**功能：**
- 🔍 自动发现自定义 router
- ✅ 验证 router 实现
- 📦 注册到系统中
- 🔧 支持多种发现策略

**关键类：**
```python
class PluginRegistry:
    - discover_plugins(plugin_dir, verbose)  # 发现插件
    - _load_router_from_directory()          # 加载 router
    - _validate_router_class()               # 验证接口
    - register_to_dict()                     # 注册到字典
```

### 2. CLI 集成

**修改文件：**
- `llmrouter/cli/router_inference.py` (推理)
- `llmrouter/cli/router_train.py` (训练)

**修改内容：** 添加插件发现和注册代码段

```python
# ============================================================================
# Plugin System Integration
# ============================================================================
from llmrouter.plugin_system import discover_and_register_plugins

plugin_registry = discover_and_register_plugins(verbose=False)

for router_name, router_class in plugin_registry.discovered_routers.items():
    ROUTER_REGISTRY[router_name] = router_class
# ============================================================================
```

### 3. 示例 Router

#### RandomRouter（简单示例）
- 📁 `custom_routers/randomrouter/`
- 功能：随机选择 LLM
- 用途：基线对比

#### ThresholdRouter（高级示例）
- 📁 `custom_routers/thresholdrouter/`
- 功能：基于难度估计路由
- 特点：包含完整训练流程

### 4. 完整文档

- 📖 `docs/CUSTOM_ROUTERS.md` - 详细教程
- 📖 `custom_routers/README.md` - 快速开始
- 📖 `PLUGIN_SYSTEM_GUIDE.md` - 完整指南

---

## 📂 完整文件结构

```
LLMRouter/
│
├── llmrouter/
│   ├── plugin_system.py              ⭐ NEW - 插件系统核心
│   ├── cli/
│   │   ├── router_inference.py       🔧 MODIFIED - 集成插件
│   │   └── router_train.py           🔧 MODIFIED - 集成插件
│   └── models/
│       └── meta_router.py            原有基类
│
├── custom_routers/                   ⭐ NEW - 自定义 router 目录
│   ├── __init__.py
│   ├── README.md                     ⭐ NEW - 使用说明
│   │
│   ├── randomrouter/                 ⭐ NEW - 示例 1
│   │   ├── __init__.py
│   │   ├── router.py                 随机路由实现
│   │   ├── trainer.py                训练器（no-op）
│   │   └── config.yaml               配置示例
│   │
│   └── thresholdrouter/              ⭐ NEW - 示例 2
│       ├── __init__.py
│       ├── router.py                 难度估计路由
│       ├── trainer.py                完整训练器
│       └── config.yaml               (可选)
│
├── docs/
│   └── CUSTOM_ROUTERS.md             ⭐ NEW - 详细文档
│
├── PLUGIN_SYSTEM_GUIDE.md            ⭐ NEW - 完整指南
└── test_plugin_system.py             ⭐ NEW - 测试脚本
```

---

## 🔑 核心设计

### 1. 插件发现机制

**自动搜索路径：**
```
1. ./custom_routers/          (项目目录，推荐)
2. ~/.llmrouter/plugins/      (用户目录)
3. $LLMROUTER_PLUGINS         (环境变量)
```

**发现策略：**
- 扫描子目录
- 查找 `router.py` 或 `model.py`
- 寻找以 `Router` 结尾的类
- 可选加载 `trainer.py` 中的 `Trainer` 类

### 2. Router 接口要求

**必须实现：**
```python
class YourRouter(MetaRouter):
    def __init__(self, yaml_path: str):
        super().__init__(model=..., yaml_path=yaml_path)

    def route_single(self, query_input: dict) -> dict:
        # 返回包含 'model_name' 的字典
        pass

    def route_batch(self, batch: list) -> list:
        # 返回结果列表
        pass
```

**可选实现（支持训练）：**
```python
class YourRouterTrainer(BaseTrainer):
    def train(self) -> None:
        # 训练逻辑
        pass
```

### 3. 零侵入集成

**原理：**
- 使用 Python 的动态导入
- 在运行时注册到现有的 `ROUTER_REGISTRY`
- 对原有代码零修改（仅添加集成代码段）

---

## 💻 使用示例

### 创建自定义 Router

```python
# custom_routers/my_router/router.py
from llmrouter.models.meta_router import MetaRouter
import torch.nn as nn

class MyRouter(MetaRouter):
    def __init__(self, yaml_path: str):
        model = nn.Identity()
        super().__init__(model=model, yaml_path=yaml_path)
        self.llm_names = list(self.llm_data.keys())

    def route_single(self, query_input: dict) -> dict:
        # 简单示例：根据查询长度路由
        query = query_input['query']

        if len(query) < 50:
            selected = self.llm_names[0]  # 短查询 -> 小模型
        else:
            selected = self.llm_names[-1]  # 长查询 -> 大模型

        return {
            "query": query,
            "model_name": selected,
            "predicted_llm": selected,
        }

    def route_batch(self, batch: list) -> list:
        return [self.route_single(q) for q in batch]
```

### 使用自定义 Router

```bash
# 推理
llmrouter infer --router my_router \
  --config custom_routers/my_router/config.yaml \
  --query "What is machine learning?"

# 训练（如果有 trainer）
llmrouter train --router my_router \
  --config custom_routers/my_router/config.yaml

# 查看所有可用 router
llmrouter list-routers
```

---

## 🎨 设计模式示例

### 1. 基于规则的路由
```python
def route_single(self, query_input):
    query = query_input['query'].lower()

    if 'code' in query:
        return {"model_name": "code-specialist"}
    elif len(query) < 50:
        return {"model_name": "small-fast-model"}
    else:
        return {"model_name": "large-model"}
```

### 2. 基于嵌入的路由
```python
from llmrouter.utils import get_longformer_embedding

def route_single(self, query_input):
    embedding = get_longformer_embedding(query_input['query'])
    similarity = self._compute_similarity(embedding)
    best_model = max(similarity, key=similarity.get)
    return {"model_name": best_model}
```

### 3. 基于成本优化的路由
```python
def route_single(self, query_input):
    difficulty = self._estimate_difficulty(query_input)

    # 选择能胜任且成本最低的模型
    for model in sorted(self.llm_data.items(), key=lambda x: x[1]['cost']):
        if model[1]['capability'] >= difficulty:
            return {"model_name": model[0]}
```

### 4. 集成路由（Ensemble）
```python
def route_single(self, query_input):
    # 多个子路由器投票
    votes = [r.route_single(query_input) for r in self.sub_routers]

    # 多数投票
    from collections import Counter
    model_votes = Counter(v['model_name'] for v in votes)
    winner = model_votes.most_common(1)[0][0]

    return {"model_name": winner}
```

---

## 🧪 测试方法

### 1. 单元测试
```python
from custom_routers.my_router import MyRouter

router = MyRouter("custom_routers/my_router/config.yaml")
result = router.route_single({"query": "test"})
assert "model_name" in result
```

### 2. 集成测试
```bash
# 仅路由测试
llmrouter infer --router my_router \
  --config config.yaml \
  --query "test" \
  --route-only

# 完整测试（包含 API 调用）
llmrouter infer --router my_router \
  --config config.yaml \
  --query "test" \
  --verbose
```

### 3. 调试模式
```python
from llmrouter.plugin_system import discover_and_register_plugins

registry = discover_and_register_plugins(
    plugin_dirs=['custom_routers'],
    verbose=True  # 显示详细发现过程
)
```

---

## 🌟 关键优势

### 1. 零侵入
- ✅ 不修改核心代码
- ✅ 只添加集成代码段（5-10行）
- ✅ 原有功能完全不受影响

### 2. 自动化
- ✅ 自动发现
- ✅ 自动验证
- ✅ 自动注册

### 3. 灵活性
- ✅ 支持多种发现路径
- ✅ 支持训练和推理
- ✅ 支持复杂 router 实现

### 4. 易用性
- ✅ 与内置 router 使用方式完全一致
- ✅ 丰富的示例和文档
- ✅ 清晰的错误提示

---

## 📊 代码统计

### 新增代码
- `llmrouter/plugin_system.py`: ~400 行
- CLI 集成代码: ~30 行（总共）
- 示例 router: ~600 行
- 文档: ~1000 行

### 修改代码
- `router_inference.py`: +15 行
- `router_train.py`: +15 行

### 总计
- 新增: ~2000 行
- 修改: ~30 行
- 侵入性: **极低**

---

## 🚀 使用流程总结

```bash
# Step 1: 创建 router 目录
mkdir -p custom_routers/awesome_router

# Step 2: 实现 router
cat > custom_routers/awesome_router/router.py << 'EOF'
from llmrouter.models.meta_router import MetaRouter
import torch.nn as nn

class AwesomeRouter(MetaRouter):
    def __init__(self, yaml_path: str):
        super().__init__(model=nn.Identity(), yaml_path=yaml_path)
        self.llm_names = list(self.llm_data.keys())

    def route_single(self, query_input: dict) -> dict:
        # 你的路由逻辑
        return {
            "query": query_input['query'],
            "model_name": self.llm_names[0],
            "predicted_llm": self.llm_names[0],
        }

    def route_batch(self, batch: list) -> list:
        return [self.route_single(q) for q in batch]
EOF

# Step 3: 创建配置
cat > custom_routers/awesome_router/config.yaml << 'EOF'
data_path:
  llm_data: 'data/example_data/llm_candidates/default_llm.json'
api_endpoint: 'https://integrate.api.nvidia.com/v1'
EOF

# Step 4: 使用！
llmrouter infer --router awesome_router \
  --config custom_routers/awesome_router/config.yaml \
  --query "Hello, world!"
```

---

## 📚 文档索引

1. **快速开始**: `custom_routers/README.md`
2. **详细教程**: `docs/CUSTOM_ROUTERS.md`
3. **完整指南**: `PLUGIN_SYSTEM_GUIDE.md`
4. **API 文档**: `llmrouter/plugin_system.py` 内联文档

---

## 🎓 推荐学习路径

1. 📖 阅读 `custom_routers/README.md`
2. 🔍 查看 `RandomRouter` 示例（最简单）
3. 💡 理解 `ThresholdRouter` 示例（可训练）
4. 🛠️ 创建自己的简单 router
5. 📈 逐步增加复杂功能
6. 🚀 分享给社区

---

## ✅ 验证清单

- [x] 插件系统核心实现
- [x] CLI 集成
- [x] 简单示例 router (RandomRouter)
- [x] 高级示例 router (ThresholdRouter)
- [x] 完整文档
- [x] 使用指南
- [x] 测试脚本
- [x] 零侵入性验证

---

## 🎉 总结

通过这个插件系统，用户现在可以：

1. ✅ **轻松扩展** - 创建自定义 router 只需几分钟
2. ✅ **无缝集成** - 使用方式与内置 router 完全一致
3. ✅ **灵活部署** - 支持多种发现路径和配置方式
4. ✅ **快速迭代** - 无需修改核心代码，快速实验新想法

**核心价值：** 让 LLMRouter 成为一个真正可扩展的框架！🚀

---

## 📞 支持

- GitHub Issues: https://github.com/ulab-uiuc/LLMRouter/issues
- 示例代码: `custom_routers/`
- 详细文档: `docs/CUSTOM_ROUTERS.md`
