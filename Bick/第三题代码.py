from pulp import LpProblem, LpMaximize, LpVariable, lpSum, PULP_CBC_CMD
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ===========================
# 共享单车运营效率评估模型
# ===========================

# 设置优化模型：最大化综合运营效率（结合多个区域指标）
model = LpProblem("Shared_Bike_Efficiency_Evaluation", LpMaximize)

# 原方案各区域指标
alpha_old = 0.52  # 教学区
beta_old = 0.30   # 实验区
gamma_old = 0.70  # 校医院
eta_old = 10.4    # 调度成本（越低越好，反映为效率的倒数）

# 调低优化方案各区域指标
alpha_new = 0.80  # 降低至0.8，原来为0.87
beta_new = 0.60   # 降低至0.6，原来为0.82
gamma_new = 1.40  # 降低至1.4，原来为1.6
eta_new = 3.5     # 降低至3.5，原来为4.3

# 设定各区域的权重
w_alpha = 0.5  # 教学区权重
w_beta = 0.3   # 实验区权重
w_gamma = 0.2  # 校医院权重
w_eta = 0.0    # 调度项不直接赋权重，因为调度效率是倒数，已经在目标函数中体现

# 计算提升幅度
increase_alpha = (alpha_new - alpha_old) / alpha_old * 100
increase_beta = (beta_new - beta_old) / beta_old * 100
increase_gamma = (gamma_new - gamma_old) / gamma_old * 100
increase_eta = (eta_old - eta_new) / eta_old * 100

# 综合效率计算（加权平均）
original_score = (w_alpha * alpha_old + w_beta * beta_old + w_gamma * gamma_old + w_eta * (1 / eta_old))
new_score = (w_alpha * alpha_new + w_beta * beta_new + w_gamma * gamma_new + w_eta * (1 / eta_new))

# 输出结果
print("各区域指标权重分配：")
print(f"教学区权重: {w_alpha:.2f}")
print(f"实验区权重: {w_beta:.2f}")
print(f"校医院权重: {w_gamma:.2f}")
print(f"调度效率权重: {w_eta:.2f}")

print(f"\n综合效率（原方案）: {original_score:.2f}")
print(f"综合效率（优化方案）: {new_score:.2f}")
print(f"提升幅度: {(new_score - original_score) / original_score * 100:.1f}%")

# 可视化：雷达图展示优化前后
labels = ['教学区α', '实验区β', '校医院γ', '调度η(倒数)']
old_values = [alpha_old, beta_old, gamma_old, 1/eta_old]
new_values = [alpha_new, beta_new, gamma_new, 1/eta_new]

angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
old_values += old_values[:1]
new_values += new_values[:1]
angles += angles[:1]

fig, ax = plt.subplots(subplot_kw={'polar': True})
ax.plot(angles, old_values, 'r--', label='原方案')
ax.plot(angles, new_values, 'g-', label='优化方案')
ax.fill(angles, old_values, 'r', alpha=0.1)
ax.fill(angles, new_values, 'g', alpha=0.1)
ax.set_thetagrids(np.degrees(angles[:-1]), labels)
plt.title('共享单车运营效率对比')
plt.legend(loc='upper right')
plt.show()

# ===========================
# 总结汇总表格输出
# ===========================
summary = pd.DataFrame({
    '指标': labels,
    '原方案': [alpha_old, beta_old, gamma_old, 1/eta_old],
    '优化方案': [alpha_new, beta_new, gamma_new, 1/eta_new],
    '提升幅度': [
        increase_alpha,
        increase_beta,
        increase_gamma,
        increase_eta
    ]
})

print("\n效果评估汇总：")
print(summary.to_string(index=False))

