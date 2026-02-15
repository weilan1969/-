#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
西汉铜漏仿真程序（最终确认版）
《过程与常数：中国古代计量范式的重新发现——以西汉铜漏为中心》
作者：张晖
版本：5.0
日期：2026-02-15

参数设置（最终确认）：
- 1刻压入式：ΔH₀ = 462 mm, H = 231 mm
- 半刻压入式：ΔH₀ = 231 mm, H = 115.5 mm
- 泄水式：H_eff = H_total - H_dead
- d_eff 采用反求值（满城1.325，汉丞相府1.992，凤栖原1.163，巨野1.256，汉错银0.90）

所有图表保存至 ./figures/，格式JPG
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import bisect
from datetime import datetime

# ==================== 中文字体配置 ====================
try:
    from matplotlib import rcParams
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'DengXian']
    for font in chinese_fonts:
        try:
            rcParams['font.sans-serif'] = [font] + rcParams['font.sans-serif']
            plt.text(0,0,'测试', fontsize=12)
            plt.close()
            print(f"使用中文字体: {font}")
            break
        except:
            continue
    rcParams['axes.unicode_minus'] = False
except Exception as e:
    print("字体配置失败，图表中文可能无法正常显示:", e)

# ==================== 全局常量 ====================
G = 9.80665          # 重力加速度 (m/s²)
CD = 0.62            # 流量系数
HAN_CHI = 0.231      # 1汉尺 = 231 mm
HALF_CHI = HAN_CHI / 2
TARGET_1KE = 864     # 1刻 = 864 秒

# ==================== 铜漏参数（最终确认版） ====================
CU_LOU_LIST = [
    # 铜壶刻 (压入式) —— 1刻（ΔH₀ = 2L, H = L）
    {'name': '千章', 'mode': 'press', 'D': 0.187, 'd_arch': 0.0031, 'd_eff': 0.002423,
     'H': HAN_CHI, 'DeltaH0': 2*HAN_CHI, 'target_T': TARGET_1KE, 'ke': 1},
    {'name': '兴平', 'mode': 'press', 'D': 0.106, 'd_arch': 0.0025, 'd_eff': 0.001373,
     'H': HAN_CHI, 'DeltaH0': 2*HAN_CHI, 'target_T': TARGET_1KE, 'ke': 1},
    {'name': '海昏侯', 'mode': 'press', 'D': 0.185, 'd_arch': 0.0040, 'd_eff': 0.002397,
     'H': HAN_CHI, 'DeltaH0': 2*HAN_CHI, 'target_T': TARGET_1KE, 'ke': 1},
    # 铜壶刻 (压入式) —— 半刻（ΔH₀ = L, H = L/2）—— 修正 d_eff 使 T = 432 s
    {'name': '满城', 'mode': 'press', 'D': 0.086, 'd_arch': None, 'd_eff': 0.001325,
     'H': HALF_CHI, 'DeltaH0': HAN_CHI, 'target_T': TARGET_1KE/2, 'ke': 0.5},
    {'name': '汉丞相府', 'mode': 'press', 'D': 0.12936, 'd_arch': None, 'd_eff': 0.001992,
     'H': HALF_CHI, 'DeltaH0': HAN_CHI, 'target_T': TARGET_1KE/2, 'ke': 0.5},
    # 铜壶漏 (泄水式) —— 保持原值
    {'name': '凤栖原', 'mode': 'drain', 'D': 0.210, 'd_arch': 0.0012, 'd_eff': 0.001163,
     'H_total': 0.340, 'H_dead': 0.0425, 'target_T': 15*TARGET_1KE, 'ke': 15},
    {'name': '巨野', 'mode': 'drain', 'D': 0.470, 'd_arch': 0.0022, 'd_eff': 0.001256,
     'H_total': 0.797, 'H_dead': 0.0797, 'target_T': 100*TARGET_1KE, 'ke': 100},
    {'name': '汉错银', 'mode': 'drain', 'D': 0.0416, 'd_arch': None, 'd_eff': 0.00090,
     'H_total': 0.0855, 'H_dead': 0.0095, 'target_T': TARGET_1KE/2, 'ke': 0.5},
]

# ==================== 物理公式 ====================
def area_from_diameter(d):
    return np.pi * (d/2)**2

def time_pressurized(D, d, H, DeltaH0):
    A = area_from_diameter(D)
    a = area_from_diameter(d)
    term = 2 * A / (CD * a * np.sqrt(2 * G))
    return term * (np.sqrt(DeltaH0) - np.sqrt(DeltaH0 - H))

def time_drain(D, d, H_eff):
    A = area_from_diameter(D)
    a = area_from_diameter(d)
    return 2 * A / (CD * a * np.sqrt(2 * G)) * np.sqrt(H_eff)

# ==================== 正向仿真 ====================
def forward_simulation(cu):
    if cu['mode'] == 'press':
        return time_pressurized(cu['D'], cu['d_eff'], cu['H'], cu['DeltaH0'])
    else:
        H_eff = cu['H_total'] - cu['H_dead']
        return time_drain(cu['D'], cu['d_eff'], H_eff)

# ==================== 蒙特卡洛模拟 (千章) ====================
def monte_carlo_qianzhang(n_samples=1_000_000, seed=42):
    np.random.seed(seed)
    qz = next(c for c in CU_LOU_LIST if c['name'] == '千章')
    D, d_eff = qz['D'], qz['d_eff']
    H, DeltaH0 = qz['H'], qz['DeltaH0']
    reading_errors = np.random.uniform(-2.16, 2.16, n_samples)
    flow_errors = np.random.normal(0, 2.6, n_samples)
    return reading_errors + flow_errors

# ==================== 扩孔校准仿真 (千章) ====================
def drill_calibration_sim(D, H, DeltaH0, d_start=0.0015, step=2e-5, max_d=0.005):
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
    qz = next(c for c in CU_LOU_LIST if c['name'] == '千章')
    D, d0, H, DeltaH0 = qz['D'], qz['d_eff'], qz['H'], qz['DeltaH0']
    H_eff = 0.3
    rel_err = np.linspace(-0.1, 0.1, 100)
    d_rel = d0 * (1 + rel_err)
    T_press = [time_pressurized(D, d, H, DeltaH0) for d in d_rel]
    T_press0 = time_pressurized(D, d0, H, DeltaH0)
    press_err = (np.array(T_press) - T_press0) / T_press0 * 100
    T_drain = [time_drain(D, d, H_eff) for d in d_rel]
    T_drain0 = time_drain(D, d0, H_eff)
    drain_err = (np.array(T_drain) - T_drain0) / T_drain0 * 100
    const_err = -2 * rel_err * 100
    return rel_err * 100, press_err, drain_err, const_err

def sensitivity_pressure():
    qz = next(c for c in CU_LOU_LIST if c['name'] == '千章')
    D, d0, H, DeltaH0 = qz['D'], qz['d_eff'], qz['H'], qz['DeltaH0']
    H_eff = 0.3
    deltaP = np.linspace(-0.02, 0.02, 100)
    T_press = [time_pressurized(D, d0, H, DeltaH0 + dp) for dp in deltaP]
    T_press0 = time_pressurized(D, d0, H, DeltaH0)
    press_err = (np.array(T_press) - T_press0) / T_press0 * 100
    const_err = -0.5 * (deltaP / DeltaH0) * 100
    T_drain = [time_drain(D, d0, H_eff + dp) for dp in deltaP]
    T_drain0 = time_drain(D, d0, H_eff)
    drain_err = (np.array(T_drain) - T_drain0) / T_drain0 * 100
    return deltaP * 1000, press_err, drain_err, const_err

# ==================== 孔径对比图 ====================
def aperture_comparison():
    names, d_arch, d_eff = [], [], []
    for cu in CU_LOU_LIST:
        if cu['d_arch'] is not None:
            names.append(cu['name'])
            d_arch.append(cu['d_arch'] * 1000)
            d_eff.append(cu['d_eff'] * 1000)
    return names, d_arch, d_eff

# ==================== 水位-时间曲线 ====================
def water_level_curve():
    qz = next(c for c in CU_LOU_LIST if c['name'] == '千章')
    D, d, H, DeltaH0 = qz['D'], qz['d_eff'], qz['H'], qz['DeltaH0']
    A = area_from_diameter(D)
    a = area_from_diameter(d)
    h = np.linspace(0, H, 500)
    term = 2 * A / (CD * a * np.sqrt(2 * G))
    T = term * (np.sqrt(DeltaH0) - np.sqrt(DeltaH0 - h))
    return h * 1000, T

# ==================== 输出表格 ====================
def print_results_table():
    print("\n" + "="*90)
    print("正向仿真结果汇总（最终确认版）")
    print("="*90)
    header = ["铜漏名称", "类型", "目标(s)", "仿真(s)", "误差(s)", "相对误差(%)", "K值", "d_eff(mm)"]
    print(f"{header[0]:<10} {header[1]:<8} {header[2]:<8} {header[3]:<8} {header[4]:<8} {header[5]:<10} {header[6]:<6} {header[7]:<8}")
    print("-"*90)

    rows = []
    for cu in CU_LOU_LIST:
        T_sim = forward_simulation(cu)
        error = T_sim - cu['target_T']
        rel_error = error / cu['target_T'] * 100
        K = T_sim * (cu['d_eff']**2) / (cu['D']**2)
        mode_str = "压入" if cu['mode'] == 'press' else "泄水"
        print(f"{cu['name']:<10} {mode_str:<8} {cu['target_T']:<8.2f} {T_sim:<8.2f} {error:<8.3f} {rel_error:<10.3f} {K:<6.4f} {cu['d_eff']*1000:<8.3f}")
        rows.append([cu['name'], mode_str, cu['target_T'], T_sim, error, rel_error, K, cu['d_eff']*1000])

    with open('simulation_results.csv', 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(["铜漏名称", "类型", "目标时间(s)", "仿真时间(s)", "误差(s)", "相对误差(%)", "K值", "理论孔径(mm)"])
        writer.writerows(rows)
    print("\n结果已保存至 simulation_results.csv")

# ==================== 主程序 ====================
def main():
    start_time = datetime.now()
    print(f"仿真开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    os.makedirs('figures', exist_ok=True)

    # 正向仿真
    print("\n正在执行正向仿真...")
    for cu in CU_LOU_LIST:
        T_sim = forward_simulation(cu)
        print(f"  {cu['name']}: 目标 {cu['target_T']:.2f}s, 仿真 {T_sim:.2f}s, 误差 {T_sim-cu['target_T']:.3f}s")

    # 输出表格
    print_results_table()

    # 蒙特卡洛模拟
    print("\n运行蒙特卡洛模拟 (100万次)...")
    errors = monte_carlo_qianzhang()
    plt.figure(figsize=(8,5))
    plt.hist(errors, bins=200, density=True, alpha=0.7, color='steelblue')
    plt.xlabel('计时误差 (秒)')
    plt.ylabel('概率密度')
    plt.title('千章铜壶蒙特卡洛模拟 (100万次)')
    plt.grid(alpha=0.3)
    mean_err, std_err = np.mean(errors), np.std(errors)
    plt.axvline(mean_err, color='red', linestyle='--', label=f'均值: {mean_err:.2f}s')
    plt.axvline(mean_err - 1.96*std_err, color='orange', linestyle=':', label='95% CI')
    plt.axvline(mean_err + 1.96*std_err, color='orange', linestyle=':')
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/fig2_montecarlo.jpg', dpi=300)
    plt.close()
    print(f"  蒙特卡洛结果: 均值={mean_err:.3f}s, 标准差={std_err:.3f}s")

    # 水位-时间曲线
    print("\n生成水位-时间曲线...")
    h_mm, T_s = water_level_curve()
    plt.figure(figsize=(8,5))
    plt.plot(T_s, h_mm, 'b-', linewidth=2)
    plt.xlabel('时间 (秒)')
    plt.ylabel('水位上升高度 (mm)')
    plt.title('千章铜壶水位-时间曲线')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/fig3_qianzhang_sim.jpg', dpi=300)
    plt.close()

    # 扩孔校准
    print("\n模拟扩孔校准过程...")
    qz = next(c for c in CU_LOU_LIST if c['name'] == '千章')
    d_mm, T_list = drill_calibration_sim(qz['D'], qz['H'], qz['DeltaH0'])
    plt.figure(figsize=(8,5))
    plt.plot(d_mm, T_list, 'g-', linewidth=2)
    plt.axhline(y=864, color='red', linestyle='--', label='目标时间 864 s')
    plt.xlabel('孔径 (mm)')
    plt.ylabel('计时时间 (s)')
    plt.title('千章铜漏扩孔校准过程仿真 (步长 0.02 mm)')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/fig5_calibration.jpg', dpi=300)
    plt.close()

    # 孔径灵敏度
    print("\n孔径灵敏度分析...")
    x, press, drain, const = sensitivity_aperture()
    plt.figure(figsize=(8,5))
    plt.plot(x, press, 'b-', label='定压注水型')
    plt.plot(x, drain, 'r--', label='泄水型')
    plt.plot(x, const, 'k:', label='恒压受水型')
    plt.xlabel('孔径相对误差 (%)')
    plt.ylabel('时间相对误差 (%)')
    plt.title('三种漏刻模式对孔径加工误差的灵敏度')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/fig_sensitivity.jpg', dpi=300)
    plt.close()

    # 压差灵敏度
    print("\n压差灵敏度分析...")
    x_p, press_p, drain_p, const_p = sensitivity_pressure()
    plt.figure(figsize=(8,5))
    plt.plot(x_p, press_p, 'b-', label='定压注水型')
    plt.plot(x_p, drain_p, 'r--', label='泄水型')
    plt.plot(x_p, const_p, 'k:', label='恒压受水型')
    plt.xlabel('压差波动 (mm)')
    plt.ylabel('时间相对误差 (%)')
    plt.title('压差波动对计时精度的影响')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/fig_pressure.jpg', dpi=300)
    plt.close()

    # 孔径对比
    print("\n生成孔径对比图...")
    names, d_arch, d_eff = aperture_comparison()
    x = np.arange(len(names))
    width = 0.35
    plt.figure(figsize=(8,5))
    plt.bar(x - width/2, d_arch, width, label='考古末端孔径', color='lightcoral')
    plt.bar(x + width/2, d_eff, width, label='理论有效孔径', color='steelblue')
    plt.xlabel('铜漏名称')
    plt.ylabel('孔径 (mm)')
    plt.title('四件铜漏考古末端孔径与理论有效孔径对比')
    plt.xticks(x, names)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/aperture_comparison.jpg', dpi=300)
    plt.close()

    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    print(f"\n仿真结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {elapsed:.2f} 秒")
    print("所有图表已保存至 ./figures/ 目录 (JPG格式)")

if __name__ == "__main__":
    main()