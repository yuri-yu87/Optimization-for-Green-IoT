# 优化问题构造与非凸优化项目

本项目提供了关于如何构造目标函数、设置约束条件以及进行非凸优化的全面指南和实现示例。

## 项目文件说明

### 1. 文档资料
- **`optimization_guide.md`** - 优化问题构造与非凸优化方法的详细指南
  - 目标函数的构造方法
  - 约束条件的设置技巧
  - 各种非凸优化算法介绍
  - 实际案例分析（包括保密率最大化问题）
  - 实现技巧和注意事项

### 2. 代码示例
- **`nonconvex_optimization_examples.py`** - Python实现示例
  - 梯度下降法
  - DC规划算法
  - 交替优化
  - 模拟退火
  - 实际应用案例（投资组合优化等）

- **`nonconvex_optimization_matlab.m`** - MATLAB实现示例
  - 各种优化算法的MATLAB实现
  - SDP松弛技术
  - 保密率最大化应用
  - 优化景观可视化

### 3. 项目相关文件
- **`Design_Journal.md`** - ELEC9123绿色物联网优化设计任务记录
- **`ELEC9123_Design_Task_E_T2_2025_OGI.pdf`** - 原始任务说明文档
- **`requirements.txt`** - Python代码所需的依赖包

## 快速开始

### Python环境
```bash
# 安装依赖
pip install -r requirements.txt

# 运行示例
python nonconvex_optimization_examples.py
```

### MATLAB环境
```matlab
% 在MATLAB中运行
run('nonconvex_optimization_matlab.m')
```

## 主要内容概览

### 目标函数构造
- 识别优化目标
- 数学建模技巧
- 多目标优化处理

### 约束条件设置
- 等式约束、不等式约束、箱式约束
- 约束建模技巧（松弛、聚合、规范化）

### 非凸优化方法
- **局部方法**：梯度下降、牛顿法
- **全局方法**：分支定界、凸松弛、SDP松弛
- **启发式算法**：模拟退火、遗传算法、粒子群
- **特殊方法**：交替优化、DC规划

## 应用示例

项目中包含了多个实际应用示例：
1. 非凸二次函数优化
2. DC规划求解稀疏优化问题
3. 投资组合优化（带交易成本）
4. 保密率最大化（波束成形优化）

## 相关资源

- CVX工具箱：http://cvxr.com/cvx/
- CVXPY文档：https://www.cvxpy.org/
- YALMIP工具箱：https://yalmip.github.io/

## 作者
Yuri Yu

---
更新日期：2025年1月 
