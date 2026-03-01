# -*- coding: utf-8 -*-
"""
基于《制冷学报》2025年文献数据的VQRC性能对比
文献来源：高玉平, 温成钰, 石文星, 等. 房间空调器性能动态测量虚拟建筑关键参数研究[J]. 制冷学报, 2025: 1-9.
DOI: 10.12465/issn.0253-4339.20250301002
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ==================== 设置中文字体 ====================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# ==================== 从文献提取的关键参数 ====================
R = 1.33                # 制冷季节选型系数 [文献表1]
C_z = 270               # 有效热容，kJ/K [文献表1，取中间值]
Q_nominal = 3500        # 空调额定容量，W（典型3.5 kW）
T_0 = 15                # 0负荷点温度，℃ [文献Fig.2估算]
T_nc = 35               # 负荷点温度，℃ [文献Fig.2]
tau = C_z / 1000        # 热时间常数，小时（简化为热容/1000）

# ==================== 生成夏季典型日温度曲线 ====================
hours = np.arange(0, 24, 0.5)  # 每半小时一个点，共48点
# 温度模型：最高温出现在15:00（35℃），最低温出现在5:00（25℃）
T_out = 25 + 8 * np.sin(np.pi * (hours - 9) / 12)  # 峰值在15点

# ==================== 计算建筑逐时负荷 ====================
load_factor = (1/R) * (T_out - T_0) / (T_nc - T_0)
load_factor = np.clip(load_factor, 0, 1)
hourly_load = Q_nominal * load_factor

# ==================== 考虑房间热惯性的负荷修正 ====================
load_actual = np.zeros_like(hourly_load)
load_actual[0] = hourly_load[0]
dt = 0.5
for i in range(1, len(hours)):
    alpha = dt / (tau + dt)
    load_actual[i] = (1-alpha) * load_actual[i-1] + alpha * hourly_load[i]

# ==================== COP建模 ====================
# 理论最优COP（无任何损失，仅随负荷变化）
cop_ideal = 4.2 * (0.9 + 0.1 * load_actual / load_actual.max())

# 节流损失系数
loss_eev = 0.04    # 传统EEV节流损失4%（对应论文中的5-10 kJ/kg，取中间值）
loss_vqrc = 0.02   # VQRC节流损失2%（对应1-2 kJ/kg）

# 基于理论最优乘以损失系数得到基准COP
cop_eev_base = cop_ideal * (1 - loss_eev)
cop_vqrc_base = cop_ideal * (1 - loss_vqrc)

# 传统EEV：滞后30分钟 + 3%随机波动
cop_eev = np.roll(cop_eev_base, 1)  # 滞后半小时（1个时间步）
cop_eev = cop_eev * (1 + 0.03 * np.random.randn(len(hours)))  # 3%波动
cop_eev = np.clip(cop_eev, 2.5, 5.0)  # 限制合理范围

# VQRC：无滞后，无波动（或允许极小波动）
cop_vqrc = cop_vqrc_base  # 完全逼近理论最优的98%

# ==================== 绘图 ====================
plt.figure(figsize=(10, 5))
plt.plot(hours, cop_ideal, 'k--', label='理论最优（无损失）', alpha=0.5, linewidth=1)
plt.plot(hours, cop_eev, 'r-', label='传统EEV（滞后+波动）', alpha=0.7, linewidth=1.5)
plt.plot(hours, cop_vqrc, 'b-', label='VQRC（无延迟自适应）', linewidth=2)
plt.xlabel('时间 (h)')
plt.ylabel('COP')
plt.title('基于动态建筑负荷模型的空调系统逐时效率对比\n（建筑参数源自《制冷学报》2025年文献）')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(np.arange(0, 24, 2))
plt.xlim(0, 24)
plt.tight_layout()

# 保存图片（PNG和PDF）
plt.savefig('vqrc_dynamic_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('vqrc_dynamic_comparison.pdf', bbox_inches='tight')
plt.show()

# ==================== 输出量化结果 ====================
print("全天平均COP统计：")
print(f"理论最优平均COP: {np.mean(cop_ideal):.3f}")
print(f"传统EEV平均COP:  {np.mean(cop_eev):.3f}")
print(f"VQRC平均COP:      {np.mean(cop_vqrc):.3f}")
print(f"VQRC相对EEV提升:  {(np.mean(cop_vqrc)-np.mean(cop_eev))/np.mean(cop_eev)*100:.1f}%")

# 保存数据到CSV
df = pd.DataFrame({
    '时刻': hours,
    '温度(℃)': T_out,
    '负荷系数': load_factor,
    '实际负荷(W)': load_actual,
    '理论最优COP': cop_ideal,
    '传统EEV_COP': cop_eev,
    'VQRC_COP': cop_vqrc
})
df.to_csv('vqrc_simulation_data.csv', index=False, encoding='utf-8-sig')
print("\n数据已保存至 vqrc_simulation_data.csv")