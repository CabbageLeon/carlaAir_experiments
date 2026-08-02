### 初始设置
#### 获取api key
https://bailian.console.aliyun.com/cn-beijing?tab=home#/home
#### 设置环境变量
在.bashrcl里填入export OPENAI_API_KEY=‘你的apikey’
### 启动单次试验
```
PYTHONPATH=/workspace/CarlaAir python -m carlaAir_experiments.spf_eval.runner landing C0 \
    --policy openfly \
    --seeds 109 --episodes-per-seed 1 --seconds 60 \
    --output runs/test_openfly


```
* landing C0:降落任务的c0阶段
* model:使用的模型名称
* seeds:随机种子号
* episodes-per-seed:每个种子执行的次数
* seconds:每个episode最大进行的时间