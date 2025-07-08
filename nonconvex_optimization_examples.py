"""
非凸优化算法实现示例
包含多种非凸优化方法的Python实现
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from typing import Tuple, Callable, Optional
import cvxpy as cp


class NonconvexOptimizer:
    """非凸优化算法集合"""
    
    def __init__(self, objective_func: Callable, constraints: Optional[list] = None):
        """
        初始化优化器
        
        Args:
            objective_func: 目标函数
            constraints: 约束条件列表
        """
        self.objective = objective_func
        self.constraints = constraints if constraints else []
        self.history = {'objective': [], 'x': []}
    
    def gradient_descent(self, x0: np.ndarray, lr: float = 0.01, 
                        max_iter: int = 1000, tol: float = 1e-6) -> Tuple[np.ndarray, float]:
        """
        梯度下降法
        
        Args:
            x0: 初始点
            lr: 学习率
            max_iter: 最大迭代次数
            tol: 收敛容差
        
        Returns:
            最优解和最优值
        """
        x = x0.copy()
        
        for i in range(max_iter):
            # 计算梯度
            grad = self._numerical_gradient(x)
            
            # 更新
            x_new = x - lr * grad
            
            # 投影到可行域（如果有约束）
            x_new = self._project_to_feasible(x_new)
            
            # 检查收敛
            if np.linalg.norm(x_new - x) < tol:
                break
                
            x = x_new
            self.history['objective'].append(self.objective(x))
            self.history['x'].append(x.copy())
        
        return x, self.objective(x)
    
    def alternating_optimization(self, x0: np.ndarray, y0: np.ndarray,
                                max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        交替优化算法
        
        Args:
            x0, y0: 初始点
            max_iter: 最大迭代次数
        
        Returns:
            最优解(x, y)和最优值
        """
        x, y = x0.copy(), y0.copy()
        
        for i in range(max_iter):
            # 固定y，优化x
            x_opt = self._optimize_x_fixed_y(x, y)
            
            # 固定x，优化y  
            y_opt = self._optimize_y_fixed_x(x_opt, y)
            
            # 检查收敛
            if np.linalg.norm(x_opt - x) + np.linalg.norm(y_opt - y) < 1e-6:
                break
                
            x, y = x_opt, y_opt
            self.history['objective'].append(self.objective(np.concatenate([x, y])))
        
        return x, y, self.objective(np.concatenate([x, y]))
    
    def dc_programming(self, x0: np.ndarray, g_func: Callable, h_func: Callable,
                      max_iter: int = 100) -> Tuple[np.ndarray, float]:
        """
        DC规划算法 (Difference of Convex)
        目标函数形式: f(x) = g(x) - h(x), 其中g和h都是凸函数
        
        Args:
            x0: 初始点
            g_func: 凸函数g
            h_func: 凸函数h
            max_iter: 最大迭代次数
        
        Returns:
            最优解和最优值
        """
        x = x0.copy()
        
        for i in range(max_iter):
            # 计算h在x_k处的次梯度
            h_grad = self._numerical_gradient(h_func, x)
            
            # 求解凸优化子问题
            # min g(x) - h(x_k) - <h_grad, x - x_k>
            def subproblem(y):
                return g_func(y) - np.dot(h_grad, y - x)
            
            result = minimize(subproblem, x, method='BFGS')
            x_new = result.x
            
            # 检查收敛
            if np.linalg.norm(x_new - x) < 1e-6:
                break
                
            x = x_new
            self.history['objective'].append(g_func(x) - h_func(x))
        
        return x, g_func(x) - h_func(x)
    
    def simulated_annealing(self, x0: np.ndarray, T0: float = 100.0,
                           cooling_rate: float = 0.95, max_iter: int = 1000) -> Tuple[np.ndarray, float]:
        """
        模拟退火算法
        
        Args:
            x0: 初始点
            T0: 初始温度
            cooling_rate: 冷却速率
            max_iter: 最大迭代次数
        
        Returns:
            最优解和最优值
        """
        x_current = x0.copy()
        x_best = x0.copy()
        f_current = self.objective(x_current)
        f_best = f_current
        T = T0
        
        for i in range(max_iter):
            # 产生邻域解
            x_new = x_current + np.random.randn(*x_current.shape) * T / T0
            
            # 投影到可行域
            x_new = self._project_to_feasible(x_new)
            
            f_new = self.objective(x_new)
            
            # 接受准则
            delta = f_new - f_current
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                x_current = x_new
                f_current = f_new
                
                if f_new < f_best:
                    x_best = x_new.copy()
                    f_best = f_new
            
            # 降温
            T *= cooling_rate
            self.history['objective'].append(f_best)
        
        return x_best, f_best
    
    def _numerical_gradient(self, func: Optional[Callable] = None, x: Optional[np.ndarray] = None, 
                           eps: float = 1e-8) -> np.ndarray:
        """计算数值梯度"""
        if func is None:
            func = self.objective
        if x is None:
            raise ValueError("x must be provided")
            
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            grad[i] = (func(x_plus) - func(x_minus)) / (2 * eps)
        return grad
    
    def _project_to_feasible(self, x: np.ndarray) -> np.ndarray:
        """投影到可行域"""
        # 这里简单实现箱式约束的投影
        # 实际应用中需要根据具体约束实现
        return np.clip(x, -10, 10)
    
    def _optimize_x_fixed_y(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """固定y优化x的子问题"""
        # 这里需要根据具体问题实现
        # 示例：使用scipy.optimize
        def sub_objective(x_var):
            return self.objective(np.concatenate([x_var, y]))
        
        result = minimize(sub_objective, x, method='BFGS')
        return result.x
    
    def _optimize_y_fixed_x(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """固定x优化y的子问题"""
        # 这里需要根据具体问题实现
        def sub_objective(y_var):
            return self.objective(np.concatenate([x, y_var]))
        
        result = minimize(sub_objective, y, method='BFGS')
        return result.x


# 示例1：非凸二次函数优化
def example_nonconvex_quadratic():
    """
    最小化: f(x, y) = x^4 - 2x^2 + y^2 + xy
    这是一个非凸函数，有多个局部最优解
    """
    def objective(x):
        return x[0]**4 - 2*x[0]**2 + x[1]**2 + x[0]*x[1]
    
    # 创建优化器
    optimizer = NonconvexOptimizer(objective)
    
    # 测试不同算法
    x0 = np.array([2.0, 1.0])
    
    print("=== 非凸二次函数优化示例 ===")
    
    # 梯度下降
    x_gd, f_gd = optimizer.gradient_descent(x0, lr=0.01)
    print(f"梯度下降: x = {x_gd}, f(x) = {f_gd:.4f}")
    
    # 模拟退火
    optimizer.history = {'objective': [], 'x': []}
    x_sa, f_sa = optimizer.simulated_annealing(x0)
    print(f"模拟退火: x = {x_sa}, f(x) = {f_sa:.4f}")
    
    # 可视化
    plot_optimization_landscape(objective, optimizer.history)


# 示例2：带约束的非凸优化（使用CVXPY的DC规划）
def example_dc_programming():
    """
    DC规划示例：
    minimize: ||x||_1 - ||x||_2 (非凸)
    subject to: Ax = b
    """
    print("\n=== DC规划示例 ===")
    
    # 问题参数
    n = 10
    m = 5
    np.random.seed(42)
    A = np.random.randn(m, n)
    b = np.random.randn(m)
    
    # DC分解: f(x) = ||x||_1 - ||x||_2
    # g(x) = ||x||_1 (凸)
    # h(x) = ||x||_2 (凸)
    
    def g_func(x):
        return np.sum(np.abs(x))
    
    def h_func(x):
        return np.linalg.norm(x)
    
    # 初始点
    x0 = np.linalg.lstsq(A, b, rcond=None)[0]
    
    # 手动实现DC算法
    x = x0.copy()
    max_iter = 50
    
    for k in range(max_iter):
        # 计算h的次梯度
        if np.linalg.norm(x) > 1e-10:
            h_grad = x / np.linalg.norm(x)
        else:
            h_grad = np.zeros_like(x)
        
        # 求解线性化子问题 (使用CVXPY)
        x_var = cp.Variable(n)
        objective = cp.norm(x_var, 1) - h_grad @ x_var
        constraints = [A @ x_var == b]
        prob = cp.Problem(cp.Minimize(objective), constraints)
        prob.solve()
        
        x_new = x_var.value
        
        # 检查收敛
        if np.linalg.norm(x_new - x) < 1e-6:
            break
            
        x = x_new
    
    print(f"DC算法结果: ||x||_1 - ||x||_2 = {g_func(x) - h_func(x):.4f}")
    print(f"稀疏度: {np.sum(np.abs(x) > 1e-6)} / {n}")


# 示例3：实际应用 - 投资组合优化（风险-收益权衡）
def example_portfolio_optimization():
    """
    投资组合优化（带交易成本的非凸问题）
    maximize: μᵀx - λ·xᵀΣx - c·||x - x0||_0
    subject to: 1ᵀx = 1, x ≥ 0
    
    其中||·||_0是0范数（非凸），表示交易数量
    """
    print("\n=== 投资组合优化示例 ===")
    
    # 问题参数
    n = 10  # 资产数量
    np.random.seed(42)
    
    # 预期收益
    mu = np.random.randn(n) * 0.1 + 0.05
    
    # 协方差矩阵
    A = np.random.randn(n, n)
    Sigma = A.T @ A / n
    
    # 当前持仓
    x0 = np.ones(n) / n
    
    # 风险厌恶系数和交易成本
    lambda_risk = 2.0
    c_trade = 0.01
    
    # 使用L1范数近似L0范数（凸松弛）
    x = cp.Variable(n)
    objective = mu @ x - lambda_risk * cp.quad_form(x, Sigma) - c_trade * cp.norm(x - x0, 1)
    constraints = [cp.sum(x) == 1, x >= 0]
    
    prob = cp.Problem(cp.Maximize(objective), constraints)
    prob.solve()
    
    x_opt = x.value
    
    print(f"优化后组合:")
    print(f"预期收益: {mu @ x_opt:.4f}")
    print(f"风险 (标准差): {np.sqrt(x_opt @ Sigma @ x_opt):.4f}")
    print(f"非零持仓数: {np.sum(x_opt > 1e-4)}")
    print(f"最大持仓: {np.max(x_opt):.4f}")


def plot_optimization_landscape(objective, history):
    """绘制优化过程的轨迹"""
    # 创建网格
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = objective(np.array([X[i, j], Y[i, j]]))
    
    # 绘图
    plt.figure(figsize=(10, 6))
    
    # 等高线图
    plt.subplot(1, 2, 1)
    plt.contour(X, Y, Z, levels=30)
    plt.colorbar(label='Objective Value')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Optimization Landscape')
    
    # 收敛曲线
    plt.subplot(1, 2, 2)
    if history['objective']:
        plt.plot(history['objective'])
        plt.xlabel('Iteration')
        plt.ylabel('Objective Value')
        plt.title('Convergence History')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('optimization_results.png')
    print("结果已保存到 optimization_results.png")


if __name__ == "__main__":
    # 运行示例
    example_nonconvex_quadratic()
    example_dc_programming()
    example_portfolio_optimization()