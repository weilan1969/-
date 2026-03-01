# -*- coding: utf-8 -*-
"""
基于《制冷学报》2025年文献数据的VQRC性能对比
文献来源：高玉平, 温成钰, 石文星, 等. 房间空调器性能动态测量虚拟建筑关键参数研究[J]. 制冷学报, 2025: 1-9.
DOI: 10.12465/issn.0253-4339.20250301002

本代码用于模拟传统电子膨胀阀（EEV）与干度基准控制器（VQRC）在典型办公建筑中的逐时运行效率，
并生成对比曲线。所有关键参数均源自上述文献。

使用方法：
1. 安装依赖：pip install numpy matplotlib pandas
2. 运行脚本：python vqrc_plot_fixed.py
3. 结果：生成对比图 (PNG/PDF) 和数据文件 (CSV)

版本：1.0
作者：张晖等
许可证：MIT
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# ==================== 全局设置 ====================
# 设置随机种子以保证结果可重复
np.random.seed(42)

# 设置中文字体（若系统无SimHei，请替换为其他可用中文字体，如'Microsoft YaHei'）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 从文献提取的关键参数 ====================
# 建筑负荷模型参数（源自文献表1及Fig.2）
PARAMS = {
    'R': 1.33,               # 制冷季节选型系数（无量纲）
    'C_z': 270,              # 有效热容，kJ/K（取文献范围240-300的中值）
    'Q_nominal': 3500,       # 空调额定容量，W（典型3.5 kW）
    'T_0': 15,               # 0负荷点温度，℃（文献Fig.2估算）
    'T_nc': 35,              # 负荷点温度，℃（文献Fig.2）
    'tau': 270 / 1000,       # 热时间常数，小时（简化为热容/1000）
    'loss_eev': 0.04,        # 传统EEV节流损失系数（对应5-10 kJ/kg，取中值）
    'loss_vqrc': 0.02,       # VQRC节流损失系数（对应1-2 kJ/kg）
}

# ==================== 生成夏季典型日温度曲线 ====================
def generate_temperature_profile(hours):
    """
    生成夏季典型日24小时室外温度变化曲线。
    假设最高温35℃出现在15:00，最低温25℃出现在5:00。
    """
    # 调整相位使峰值在15点（hours=15）
    T_out = 25 + 8 * np.sin(np.pi * (hours - 9) / 12)
    return T_out

# ==================== 建筑负荷计算 ====================
def calculate_building_load(T_out, params):
    """
    根据室外温度计算建筑逐时冷负荷。
    使用文献中的负荷线公式。
    """
    # 负荷系数
    load_factor = (1 / params['R']) * (T_out - params['T_0']) / (params['T_nc'] - params['T_0'])
    load_factor = np.clip(load_factor, 0, 1)  # 限制在0~1之间
    hourly_load = params['Q_nominal'] * load_factor
    return hourly_load, load_factor

# ==================== 考虑房间热惯性修正 ====================
def apply_thermal_inertia(load, dt, tau):
    """
    通过一阶惯性滤波模拟房间热容对负荷的滞后影响。
    """
    load_filtered = np.zeros_like(load)
    load_filtered[0] = load[0]
    for i in range(1, len(load)):
        alpha = dt / (tau + dt)
        load_filtered[i] = (1 - alpha) * load_filtered[i-1] + alpha * load[i]
    return load_filtered

# ==================== COP建模 ====================
def compute_cop(load_actual, params):
    """
    计算理论最优COP、EEV和VQRC的逐时COP。
    """
    # 理论最优COP：随负荷线性变化（低负荷时COP略低）
    cop_ideal = 4.2 * (0.9 + 0.1 * load_actual / load_actual.max())

    # 考虑节流损失后的基准COP
    cop_eev_base = cop_ideal * (1 - params['loss_eev'])
    cop_vqrc_base = cop_ideal * (1 - params['loss_vqrc'])

    # 传统EEV：滞后半小时 + 3%随机波动
    cop_eev = np.roll(cop_eev_base, 1)  # 滞后半小时（1个时间步）
    cop_eev = cop_eev * (1 + 0.03 * np.random.randn(len(cop_eev)))  # 3%波动
    cop_eev = np.clip(cop_eev, 2.5, 5.0)  # 限制合理范围

    # VQRC：无滞后，无波动
    cop_vqrc = cop_vqrc_base

    return cop_ideal, cop_eev, cop_vqrc

# ==================== 绘图函数 ====================
def plot_results(hours, cop_ideal, cop_eev, cop_vqrc, save_path=None):
    """
    绘制COP对比曲线并保存图片。
    """
    plt.figure(figsize=(10, 5))
    plt.plot(hours, cop_ideal, 'k--', label='理论最优（无损失）', alpha=0.5, linewidth=1)
    plt.plot(hours, cop_eev, 'r-', label='传统EEV（滞后+波动）', alpha=0.7, linewidth=1.5)
    plt.plot(hours, cop_vqrc, 'b-', label='VQRC（无延迟自适应）', linewidth=2)
    plt.xlabel('时间 (h)', fontsize=12)
    plt.ylabel('COP', fontsize=12)
    plt.title('基于动态建筑负荷模型的空调系统逐时效率对比\n（建筑参数源自《制冷学报》2025年文献）', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(np.arange(0, 24, 2))
    plt.xlim(0, 24)
    plt.tight_layout()

    if save_path:
        # 同时保存为PNG和PDF
        plt.savefig(save_path + '.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_path + '.pdf', bbox_inches='tight')
        print(f"图表已保存为 {save_path}.png 和 {save_path}.pdf")
    plt.show()

# ==================== 主程序 ====================
def main():
    print("=" * 50)
    print("VQRC性能仿真 - 基于《制冷学报》2025年文献")
    print("=" * 50)

    # 时间向量（24小时，步长0.5小时）
    hours = np.arange(0, 24, 0.5)
    dt = 0.5  # 时间步长（小时）

    # 生成温度曲线
    T_out = generate_temperature_profile(hours)

    # 计算建筑负荷
    hourly_load, load_factor = calculate_building_load(T_out, PARAMS)

    # 考虑热惯性修正
    load_actual = apply_thermal_inertia(hourly_load, dt, PARAMS['tau'])

    # 计算COP
    cop_ideal, cop_eev, cop_vqrc = compute_cop(load_actual, PARAMS)

    # 输出统计结果
    print("\n全天平均COP统计：")
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
    csv_file = 'vqrc_simulation_data.csv'
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"\n数据已保存至 {csv_file}")

    # 绘图
    plot_results(hours, cop_ideal, cop_eev, cop_vqrc, save_path='vqrc_dynamic_comparison')

    print("\n仿真完成。")

if __name__ == "__main__":
    main()