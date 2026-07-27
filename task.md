### 任务介绍
1. 本环境基于CarlaAir 仿真环境，需要复现包括see-point-fly在内的一系列论文，控制示例可以查看/home/shuning/CarlaAir/examples下的代码
2. python虚拟环境配置采用uv
2. 实验相关要求参考/home/shuning/CarlaAir/experiment.md 
### 你的任务
1. 使用carlaair仿真环境搭建一个无人机+卡车（目前默认是tesla，可能需要更改模型）环境
2. 严格按照/home/shuning/CarlaAir/experiment.md 的要求编写实验代码
3. 首先针对spf编写实验代码，并留出其他论文（比如AerialVLN）的接口，便于后续接入
4. 实验代码需要
### 关于spf
* spf的全套代码在/home/shuning/CarlaAir/see-point-fly
* api请求地址为https://ws-9uve2kqdj44bdrwr.cn-beijing.maas.aliyuncs.com/compatible-mode/v1（与openai兼容）
* api_key为sk-ws-H.EHYHRXL.gwjW.MEQCIGtv0uNPqmWXRilJ7NZu8XgvBxuFSnBWPMxj8LcZTS3qAiAPo19TbHHxYSL4VH0SXqUCvZLK8VIDciZhxq9yTE80Pw
### 关于最后指标
#### 协同降落指标
| Method | Mode | TSR↑ | LSR↑ | CCR↑ | CG↑ |
|---|---|---|---|---|---|
| AerialVLA | C0 | 0.78±0.04 | 0.13±0.03 | 0.17±0.02 | 0.00 |
| AerialVLA | C1 | 0.76±0.04 | 0.10±0.03 | 0.13±0.02 | −0.03±0.02 |
| AerialVLA | C2 | 0.70±0.04 | 0.06±0.02 | 0.09±0.02 | −0.07±0.02 |
| OpenFly | C0 | 0.81±0.03 | 0.14±0.03 | 0.17±0.02 | 0.00 |
| OpenFly | C1 | 0.79±0.04 | 0.10±0.03 | 0.13±0.02 | −0.04±0.02 |
| OpenFly | C2 | 0.73±0.04 | 0.05±0.02 | 0.07±0.02 | −0.09±0.02 |
| OpenUAV | C0 | 0.74±0.04 | 0.09±0.03 | 0.12±0.02 | 0.00 |
| OpenUAV | C1 | 0.70±0.04 | 0.06±0.02 | 0.09±0.02 | −0.03±0.02 |
| OpenUAV | C2 | 0.64±0.04 | 0.03±0.02 | 0.05±0.02 | −0.06±0.02 |
| SPF | C0 | 0.62±0.04 | 0.06±0.02 | 0.10±0.02 | 0.00 |
| SPF | C1 | 0.60±0.04 | 0.04±0.02 | 0.07±0.02 | −0.02±0.02 |
| SPF | C2 | 0.55±0.04 | 0.02±0.01 | 0.04±0.02 | −0.04±0.02 |
| AerialVLN | C0 | 0.55±0.03 | 0.03±0.02 | 0.05±0.02 | 0.00 |
| AerialVLN | C1 | 0.51±0.03 | 0.02±0.01 | 0.04±0.01 | −0.01±0.01 |
| AerialVLN | C2 | 0.47±0.03 | 0.01±0.01 | 0.02±0.01 | −0.02±0.01 |
| *State-based cooperative reference* | | | | | |
| Rule-Coop-State | Ref. | **0.84±0.03** | **0.42±0.03** | **0.50±0.03** | — |
#### 护送指标
| Method | Mode | RSR↑ | RAT (s)↓ |
|---|---|---|---|
| AerialVLA | C0 | 0.56±0.04 | 4.8±0.3 |
| AerialVLA | C1 | 0.58±0.04 | 4.6±0.3 |
| OpenFly | C0 | 0.59±0.04 | 4.5±0.3 |
| OpenFly | C1 | 0.57±0.04 | 4.7±0.3 |
| OpenUAV | C0 | 0.48±0.04 | 6.2±0.4 |
| OpenUAV | C1 | 0.44±0.04 | 6.7±0.4 |
| SPF | C0 | 0.46±0.04 | 6.5±0.4 |
| SPF | C1 | 0.54±0.04 | 5.4±0.3 |
| AerialVLN | C0 | 0.41±0.03 | 7.2±0.4 |
| AerialVLN | C1 | 0.40±0.03 | 7.4±0.4 |
| *State-based cooperative reference* | | | |
| Rule-Coop-State | Ref. | **0.86±0.03** | **1.8±0.2** |

