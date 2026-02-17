#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
西汉铜漏仿真程序（96刻制最终版）
论文：《过程与常数：中国古代计量范式的重新发现》
作者：张晖
版本：2.0
日期：2026-02-17

设定：
- 一天分为96刻，每刻 = 900秒
- 汉尺 = 231 mm
- 流量系数 Cd = 0.62
- 重力加速度 g = 9.80665 m/s²

功能：
1. 压入式铜壶（铜壶刻）：千章、兴平、海昏侯、满城、凤栖原、汉丞相府
   - 根据目标时间反求理论孔径 d_eff
   - 正向仿真计算实际时间误差
2. 泄水式铜壶（铜壶漏）：巨野
   - 根据考古孔径计算实际时间，并找出最接近的整数刻数
3. 两用式铜壶：汉错银（压入式0.2刻 + 泄水式0.5刻）
   - 寻找最佳折中孔径，计算两种模式的时间误差
4. 输出所有仿真结果，生成论文所需图表：
   - 扩孔校准过程仿真（图2）
   - 蒙特卡洛模拟误差分布（图3）
   - 水位-时间曲线（图4）
   - 孔径灵敏度对比（图6）
   - 压差灵敏度对比（图7）
   - 考古孔径与理论孔径对比（图8）

所有结果保存至 ./figures/ 目录，并生成数据表格 simulation_results.csv
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import bisect
from datetime import datetime

# ==================== 全局字体配置（确保中文显示） ====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 全局常量 ====================
G = 9.80665               # 重力加速度 (m/s²)
CD = 0.62                 # 流量系数
HAN_CHI = 0.231           # 1汉尺 = 231 mm
SEC_PER_KE = 900          # 每刻900秒（96刻制）

# 常用目标时间（秒）
TARGETS = {
    '1刻': SEC_PER_KE,
    '1/2刻': SEC_PER_KE / 2,
    '2/3刻': SEC_PER_KE * 2 / 3,
    '6刻': SEC_PER_KE * 6,
    '32刻': SEC_PER_KE * 32,
    '0.2刻': SEC_PER_KE * 0.2,
    '0.5刻': SEC_PER_KE * 0.5,
}

# ==================== 铜漏参数（最终确认版） ====================
# 字段说明：
#   name        : 名称
#   mode        : 'press' 压入式 / 'drain' 泄水式 / 'dual' 两用式
#   D           : 壶内径 (m)
#   d_arch      : 考古末端孔径 (m) (None 表示缺失)
#   H           : 箭尺行程 (m) (压入式用)
#   DeltaH0     : 初始压差 (m) (压入式用)
#   H_total     : 总深 (m) (泄水式用)
#   H_dead      : 死水区高度 (m) (泄水式用)
#   target_T    : 目标时间 (s) (可多个，用列表)
#   ke          : 刻数 (可多个)
CU_LOU_LIST = [
    # 千章 (压入式1刻)
    {'name': '千章', 'mode': 'press', 'D': 0.187, 'd_arch': 0.0031,
     'H': HAN_CHI, 'DeltaH0': 2*HAN_CHI, 'target_T': TARGETS['1刻'], 'ke': 1},
    # 兴平 (压入式1刻)
    {'name': '兴平', 'mode': 'press', 'D': 0.106, 'd_arch': 0.0025,
     'H': HAN_CHI, 'DeltaH0': 2*HAN_CHI, 'target_T': TARGETS['1刻'], 'ke': 1},
    # 海昏侯 (压入式1刻)
    {'name': '海昏侯', 'mode': 'press', 'D': 0.185, 'd_arch': 0.0040,
     'H': HAN_CHI, 'DeltaH0': 2*HAN_CHI, 'target_T': TARGETS['1刻'], 'ke': 1},
    # 满城 (压入式半刻)
    {'name': '满城', 'mode': 'press', 'D': 0.086, 'd_arch': None,
     'H': HAN_CHI/2, 'DeltaH0': HAN_CHI, 'target_T': TARGETS['1/2刻'], 'ke': 0.5},
    # 凤栖原 (压入式6刻)
    {'name': '凤栖原', 'mode': 'press', 'D': 0.210, 'd_arch': 0.0012,
     'H': 0.308, 'DeltaH0': 0.616, 'target_T': TARGETS['6刻'], 'ke': 6},
    # 汉丞相府 (压入式2/3刻) —— 新参数
    {'name': '汉丞相府', 'mode': 'press', 'D': 0.130, 'd_arch': None,
     'H': HAN_CHI * 2 / 3, 'DeltaH0': HAN_CHI * 4 / 3, 'target_T': TARGETS['2/3刻'], 'ke': 2/3},
    # 巨野 (泄水式32刻) —— 新数据
    {'name': '巨野', 'mode': 'drain', 'D': 0.470, 'd_arch': 0.0022,
     'H_total': 0.793 - 0.007,   # 总高减底厚
     'H_dead': (0.050 - 0.007),   # 孔中心距底50mm - 底厚7mm
     'target_T': TARGETS['32刻'], 'ke': 32},
    # 汉错银 (两用式) —— 单独处理
    {'name': '汉错银', 'mode': 'dual', 'D': 0.0416, 'd_arch': None,
     'H': HAN_CHI / 3,            # 行程77mm
     'H_total': 0.0855,           # 总深85.5mm
     'target_T_press': TARGETS['0.2刻'], 'ke_press': 0.2,
     'target_T_drain': TARGETS['0.5刻'], 'ke_drain': 0.5},
]

# ==================== 物理公式 ====================
def area_from_diameter(d):
    """计算圆面积 (m²)"""
    return np.pi * (d/2)**2

def time_pressurized(D, d, H, DeltaH0):
    """压入式计时公式 (式1)"""
    A = area_from_diameter(D)
    a = area_from_diameter(d)
    term = 2 * A / (CD * a * np.sqrt(2 * G))
    return term * (np.sqrt(DeltaH0) - np.sqrt(DeltaH0 - H))

def time_drain(D, d, H_eff):
    """泄水式计时公式 (式2)"""
    A = area_from_diameter(D)
    a = area_from_diameter(d)
    return 2 * A / (CD * a * np.sqrt(2 * G)) * np.sqrt(H_eff)

def inverse_diameter_press(target_T, D, H, DeltaH0):
    """二分法反求压入式铜壶的理论孔径 d_eff (m)"""
    def error(d):
        return time_pressurized(D, d, H, DeltaH0) - target_T
    # 搜索范围 [0.1mm, 10mm]
    try:
        d_sol = bisect(error, 0.0001, 0.01, xtol=1e-8)
    except ValueError:
        d_sol = bisect(error, 0.00005, 0.02, xtol=1e-8)
    return d_sol

def inverse_diameter_drain(target_T, D, H_eff):
    """二分法反求泄水式铜壶的理论孔径 d_eff (m)"""
    def error(d):
        return time_drain(D, d, H_eff) - target_T
    try:
        d_sol = bisect(error, 0.0001, 0.01, xtol=1e-8)
    except ValueError:
        d_sol = bisect(error, 0.00005, 0.02, xtol=1e-8)
    return d_sol

# ==================== 正向仿真函数 ====================
def forward_simulation(cu, d_eff):
    """根据铜漏参数和给定孔径计算仿真时间"""
    if cu['mode'] == 'press':
        return time_pressurized(cu['D'], d_eff, cu['H'], cu['DeltaH0'])
    elif cu['mode'] == 'drain':
        H_eff = cu['H_total'] - cu['H_dead']
        return time_drain(cu['D'], d_eff, H_eff)
    else:
        raise ValueError("两用式需单独处理")

# ==================== 蒙特卡洛模拟（千章） ====================
def monte_carlo_qianzhang(n_samples=1_000_000, seed=42):
    """千章铜壶蒙特卡洛模拟 (论文图3)"""
    np.random.seed(seed)
    # 读取千章参数
    D = 0.187
    H = HAN_CHI
    DeltaH0 = 2*HAN_CHI
    d_eff = 0.002423  # 理论孔径，后续可从仿真获得，此处固定

    # 基础时间
    T0 = time_pressurized(D, d_eff, H, DeltaH0)

    # 误差源
    reading_errors = np.random.uniform(-2.16, 2.16, n_samples)   # 读数误差 ±0.5mm → ±2.16s
    flow_errors = np.random.normal(0, 2.6, n_samples)            # 流量波动 标准差2.6s
    errors = reading_errors + flow_errors
    return errors

# ==================== 扩孔校准仿真 (千章) ====================
def drill_calibration_sim(D, H, DeltaH0, d_start=0.0015, step=2e-5, max_d=0.005):
    """
    模拟扩孔过程 (论文图2)
    返回：d_list (mm), T_list (s)
    """
    d_list, T_list = [], []
    d = d_start
    while d <= max_d:
        T = time_pressurized(D, d, H, DeltaH0)
        d_list.append(d * 1000)
        T_list.append(T)
        d += step
    return d_list, T_list

# ==================== 灵敏度分析 ====================
def sensitivity_aperture():
    """孔径灵敏度分析 (论文图6)"""
    D = 0.187
    d0 = 0.002423
    H = HAN_CHI
    DeltaH0 = 2*HAN_CHI
    H_eff = 0.3   # 泄水式典型有效水深

    rel_err = np.linspace(-0.1, 0.1, 100)
    d_rel = d0 * (1 + rel_err)

    # 压入式
    T_press = [time_pressurized(D, d, H, DeltaH0) for d in d_rel]
    T_press0 = time_pressurized(D, d0, H, DeltaH0)
    press_err = (np.array(T_press) - T_press0) / T_press0 * 100

    # 泄水式
    T_drain = [time_drain(D, d, H_eff) for d in d_rel]
    T_drain0 = time_drain(D, d0, H_eff)
    drain_err = (np.array(T_drain) - T_drain0) / T_drain0 * 100

    # 恒压受水型 (近似)
    const_err = -2 * rel_err * 100
    return rel_err * 100, press_err, drain_err, const_err

def sensitivity_pressure():
    """压差灵敏度分析 (论文图7)"""
    D = 0.187
    d0 = 0.002423
    H = HAN_CHI
    DeltaH0 = 2*HAN_CHI
    H_eff = 0.3

    deltaP = np.linspace(-0.02, 0.02, 100)  # 压差波动 ±20 mm

    # 压入式
    T_press = [time_pressurized(D, d0, H, DeltaH0 + dp) for dp in deltaP]
    T_press0 = time_pressurized(D, d0, H, DeltaH0)
    press_err = (np.array(T_press) - T_press0) / T_press0 * 100

    # 泄水式 (有效水深变化)
    T_drain = [time_drain(D, d0, H_eff + dp) for dp in deltaP]
    T_drain0 = time_drain(D, d0, H_eff)
    drain_err = (np.array(T_drain) - T_drain0) / T_drain0 * 100

    # 恒压受水型 (近似)
    const_err = -0.5 * (deltaP / DeltaH0) * 100
    return deltaP * 1000, press_err, drain_err, const_err

# ==================== 水位-时间曲线 ====================
def water_level_curve():
    """千章铜壶水位-时间曲线 (论文图4)"""
    D = 0.187
    d = 0.002423
    H = HAN_CHI
    DeltaH0 = 2*HAN_CHI
    A = area_from_diameter(D)
    a = area_from_diameter(d)
    h = np.linspace(0, H, 500)
    term = 2 * A / (CD * a * np.sqrt(2 * G))
    T = term * (np.sqrt(DeltaH0) - np.sqrt(DeltaH0 - h))
    return h * 1000, T   # 水位 mm, 时间 s

# ==================== 孔径对比图 (图8) ====================
def aperture_comparison():
    """提取有考古孔径数据的铜漏"""
    names = []
    d_arch = []
    d_eff = []
    for cu in CU_LOU_LIST:
        if cu['mode'] in ('press', 'drain') and cu.get('d_arch') is not None:
            names.append(cu['name'])
            d_arch.append(cu['d_arch'] * 1000)
            if cu['mode'] == 'press':
                d_eff_val = inverse_diameter_press(cu['target_T'], cu['D'], cu['H'], cu['DeltaH0']) * 1000
            else:
                H_eff = cu['H_total'] - cu['H_dead']
                d_eff_val = inverse_diameter_drain(cu['target_T'], cu['D'], H_eff) * 1000
            d_eff.append(d_eff_val)
    # 注意：巨野使用新数据，已包含在内
    return names, d_arch, d_eff

# ==================== 汉错银两用式优化 ====================
def han_cuoyin_dual():
    """
    汉错银两用式优化：寻找最佳折中孔径，使得压入式0.2刻与泄水式0.5刻的误差尽可能小
    返回：最佳孔径 d_opt (mm)，两种模式的实际时间和误差
    """
    D = 0.0416
    H = HAN_CHI / 3          # 77 mm
    DeltaH0 = 2 * H
    H_total = 0.0855
    H_dead = 0.0095          # 原始死水区9.5mm，行程77mm，总深85.5mm，死水区8.5mm？此处沿用原值，但可根据需要调整
    H_eff = H_total - H_dead  # 有效水深 = 0.076 m (与行程一致)
    target_press = TARGETS['0.2刻']
    target_drain = TARGETS['0.5刻']

    # 计算单独满足各自目标的孔径
    d_press = inverse_diameter_press(target_press, D, H, DeltaH0) * 1000
    d_drain = inverse_diameter_drain(target_drain, D, H_eff) * 1000

    # 在两者之间扫描最佳折中
    d_min = min(d_press, d_drain) - 0.02
    d_max = max(d_press, d_drain) + 0.02
    d_range = np.linspace(d_min, d_max, 200)
    best_d = None
    best_err = float('inf')
    for d_mm in d_range:
        d_m = d_mm / 1000
        a = area_from_diameter(d_m)
        # 计算两种模式的时间
        T_p = time_pressurized(D, d_m, H, DeltaH0)
        T_d = time_drain(D, d_m, H_eff)
        # 误差平方和
        err = (T_p - target_press)**2 + (T_d - target_drain)**2
        if err < best_err:
            best_err = err
            best_d = d_mm

    # 用最佳孔径计算最终时间
    d_m = best_d / 1000
    a = area_from_diameter(d_m)
    T_p_final = time_pressurized(D, d_m, H, DeltaH0)
    T_d_final = time_drain(D, d_m, H_eff)
    return best_d, T_p_final, T_d_final

# ==================== 主程序 ====================
def main():
    start_time = datetime.now()
    print(f"仿真开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    os.makedirs('figures', exist_ok=True)

    # -------------------- 1. 正向仿真与反求（压入式/泄水式） --------------------
    print("\n正在进行正向仿真与理论孔径反求...")
    results = []
    for cu in CU_LOU_LIST:
        if cu['mode'] == 'dual':
            continue   # 两用式单独处理
        if 'd_eff' not in cu:
            if cu['mode'] == 'press':
                cu['d_eff'] = inverse_diameter_press(cu['target_T'], cu['D'], cu['H'], cu['DeltaH0'])
            else:
                H_eff = cu['H_total'] - cu['H_dead']
                cu['d_eff'] = inverse_diameter_drain(cu['target_T'], cu['D'], H_eff)
        T_sim = forward_simulation(cu, cu['d_eff'])
        error = T_sim - cu['target_T']
        K = T_sim * (cu['d_eff']**2) / (cu['D']**2)
        print(f"{cu['name']}: 目标 {cu['target_T']:.2f}s, 仿真 {T_sim:.2f}s, 误差 {error:.3f}s, d_eff={cu['d_eff']*1000:.3f}mm, K={K:.4f}")
        results.append({
            'name': cu['name'],
            'mode': '压入' if cu['mode']=='press' else '泄水',
            'target_T': cu['target_T'],
            'sim_T': T_sim,
            'error': error,
            'd_eff_mm': cu['d_eff']*1000,
            'K': K,
        })

    # -------------------- 2. 汉错银两用式 --------------------
    d_opt, T_p, T_d = han_cuoyin_dual()
    print("\n【汉错银两用式】")
    print(f"最佳折中孔径: {d_opt:.3f} mm")
    print(f"压入式0.2刻: 目标 {TARGETS['0.2刻']:.0f}s, 仿真 {T_p:.1f}s, 误差 {T_p - TARGETS['0.2刻']:.1f}s")
    print(f"泄水式0.5刻: 目标 {TARGETS['0.5刻']:.0f}s, 仿真 {T_d:.1f}s, 误差 {T_d - TARGETS['0.5刻']:.1f}s")
    results.append({
        'name': '汉错银',
        'mode': '两用',
        'target_T': f"{TARGETS['0.2刻']:.0f}/{TARGETS['0.5刻']:.0f}",
        'sim_T': f"{T_p:.1f}/{T_d:.1f}",
        'error': f"{T_p - TARGETS['0.2刻']:.1f}/{T_d - TARGETS['0.5刻']:.1f}",
        'd_eff_mm': d_opt,
        'K': None,
    })

    # -------------------- 3. 保存结果表格 --------------------
    with open('simulation_results.csv', 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['名称', '类型', '目标时间(s)', '仿真时间(s)', '误差(s)', '理论孔径(mm)', 'K值'])
        for r in results:
            writer.writerow([r['name'], r['mode'], r['target_T'], r['sim_T'], r['error'], r['d_eff_mm'], r['K']])
    print("\n结果已保存至 simulation_results.csv")

    # -------------------- 4. 蒙特卡洛模拟（图3） --------------------
    print("\n运行蒙特卡洛模拟 (100万次)...")
    errors = monte_carlo_qianzhang()
    plt.figure(figsize=(8,5))
    plt.hist(errors, bins=200, density=True, alpha=0.7, color='steelblue')
    plt.xlabel('计时误差 (秒)')
    plt.ylabel('概率密度')
    plt.title('千章铜壶蒙特卡洛模拟 (100万次)')
    plt.grid(alpha=0.3)
    mean_err = np.mean(errors)
    std_err = np.std(errors)
    plt.axvline(mean_err, color='red', linestyle='--', label=f'均值: {mean_err:.2f}s')
    plt.axvline(mean_err - 1.96*std_err, color='orange', linestyle=':', label='95% CI')
    plt.axvline(mean_err + 1.96*std_err, color='orange', linestyle=':')
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/fig3_montecarlo.jpg', dpi=300)
    plt.close()
    print(f"蒙特卡洛结果: 均值={mean_err:.3f}s, 标准差={std_err:.3f}s")

    # -------------------- 5. 水位-时间曲线（图4） --------------------
    print("\n生成水位-时间曲线...")
    h_mm, T_s = water_level_curve()
    plt.figure(figsize=(8,5))
    plt.plot(T_s, h_mm, 'b-', linewidth=2)
    plt.xlabel('时间 (秒)')
    plt.ylabel('水位上升高度 (mm)')
    plt.title('千章铜壶水位-时间曲线')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/fig4_water_level.jpg', dpi=300)
    plt.close()

    # -------------------- 6. 扩孔校准仿真（图2） --------------------
    print("\n模拟扩孔校准过程...")
    D = 0.187
    H = HAN_CHI
    DeltaH0 = 2*HAN_CHI
    d_mm, T_list = drill_calibration_sim(D, H, DeltaH0)
    plt.figure(figsize=(8,5))
    plt.plot(d_mm, T_list, 'g-', linewidth=2)
    plt.axhline(y=SEC_PER_KE, color='red', linestyle='--', label=f'目标时间 {SEC_PER_KE:.0f} s')
    plt.xlabel('孔径 (mm)')
    plt.ylabel('计时时间 (s)')
    plt.title('千章铜漏扩孔校准过程仿真 (步长 0.02 mm)')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/fig2_calibration.jpg', dpi=300)
    plt.close()

    # -------------------- 7. 孔径灵敏度分析（图6） --------------------
    print("\n孔径灵敏度分析...")
    x, press_err, drain_err, const_err = sensitivity_aperture()
    plt.figure(figsize=(8,5))
    plt.plot(x, press_err, 'b-', label='压水型')
    plt.plot(x, drain_err, 'r--', label='泄水型')
    plt.plot(x, const_err, 'k:', label='恒压受水型')
    plt.xlabel('孔径相对误差 (%)')
    plt.ylabel('时间相对误差 (%)')
    plt.title('三种漏刻模式对孔径加工误差的灵敏度')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/fig6_aperture_sensitivity.jpg', dpi=300)
    plt.close()

    # -------------------- 8. 压差灵敏度分析（图7） --------------------
    print("\n压差灵敏度分析...")
    x_p, press_p, drain_p, const_p = sensitivity_pressure()
    plt.figure(figsize=(8,5))
    plt.plot(x_p, press_p, 'b-', label='压水型')
    plt.plot(x_p, drain_p, 'r--', label='泄水型')
    plt.plot(x_p, const_p, 'k:', label='恒压受水型')
    plt.xlabel('压差波动 (mm)')
    plt.ylabel('时间相对误差 (%)')
    plt.title('压差波动对计时精度的影响')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/fig7_pressure_sensitivity.jpg', dpi=300)
    plt.close()

    # -------------------- 9. 孔径对比图（图8） --------------------
    print("\n生成孔径对比图...")
    names, d_arch, d_eff = aperture_comparison()
    x = np.arange(len(names))
    width = 0.35
    plt.figure(figsize=(8,5))
    plt.bar(x - width/2, d_arch, width, label='考古末端孔径', color='lightcoral')
    plt.bar(x + width/2, d_eff, width, label='理论有效孔径', color='steelblue')
    plt.xlabel('铜漏名称')
    plt.ylabel('孔径 (mm)')
    plt.title('五件铜漏考古末端孔径与理论有效孔径对比')
    plt.xticks(x, names)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/fig8_aperture_comparison.jpg', dpi=300)
    plt.close()

    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    print(f"\n仿真结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {elapsed:.2f} 秒")
    print("所有图表已保存至 ./figures/ 目录 (JPG格式)")

if __name__ == "__main__":
    main()