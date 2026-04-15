# -*- coding: utf-8 -*-
"""
条件14候補5銀河の2段階パラメータ抽出
"""
import csv
import numpy as np
from pathlib import Path

CSV_PATH = Path(r'D:\ドキュメント\エントロピー\新膜宇宙論\これまでの軌跡\パイソン\phase1\sparc_results.csv')
TARGET = ['NGC1003','NGC2403','NGC2903','NGC6015','UGC00128']

with open(CSV_PATH, newline='', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    print(f"\n{'galaxy':<12} {'rs1':>6} {'rs2':>6} "
          f"{'fe':>5} {'rs2/rs1':>8} {'Ud':>5} {'vflat':>7} {'chi2_2s':>8}")
    print('-'*60)
    rs1_list, rs2_list, fe_list, ratio_list = [], [], [], []
    for row in reader:
        if row['galaxy'] in TARGET:
            rs1 = float(row['rs1'])
            rs2 = float(row['rs2'])
            fe  = float(row['fe'])
            ud  = float(row['ud'])
            vf  = float(row['vflat'])
            c2  = float(row['chi2_2s'])
            ratio = rs2/rs1 if rs1 > 0 else float('nan')
            print(f"{row['galaxy']:<12} {rs1:>6.2f} {rs2:>6.2f} "
                  f"{fe:>5.2f} {ratio:>8.2f} {ud:>5.2f} {vf:>7.1f} {c2:>8.3f}")
            rs1_list.append(rs1)
            rs2_list.append(rs2)
            fe_list.append(fe)
            ratio_list.append(ratio)
    print('-'*60)
    print(f"{'median':<12} "
          f"{np.median(rs1_list):>6.2f} "
          f"{np.median(rs2_list):>6.2f} "
          f"{np.median(fe_list):>5.2f} "
          f"{np.median(ratio_list):>8.2f}")
    print(f"{'mean':<12} "
          f"{np.mean(rs1_list):>6.2f} "
          f"{np.mean(rs2_list):>6.2f} "
          f"{np.mean(fe_list):>5.2f} "
          f"{np.mean(ratio_list):>8.2f}")
