# reaction.py
import sys
import os
import numpy as np
import pandas as pd

class Dyn:
    @staticmethod
    def empty(lst):
        return np.hstack(lst) if lst else np.empty((0, 0))
    @staticmethod
    def spectrum(wave, damp, dt, Tp=6, dtp=0.02):
        # 转换成numpy数组，改成float格式
        arr = np.asarray(wave, dtype=float)
        # 统一成二维向量
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        Tt = np.arange(dtp, Tp + dtp, dtp).reshape(-1, 1)
        _, cols = arr.shape

        ARS = []
        VRS = []
        DRS = []
        PVRS = []
        PARS = []

        for i in range(cols):
            arri = arr[:, i]
            A, V, D, PA, PV = Dyn.wave2res(arri, damp, dt, Tp, dtp)
            ARS.append(A.reshape(-1, 1))
            VRS.append(V.reshape(-1, 1))
            DRS.append(D.reshape(-1, 1))
            PVRS.append(PV.reshape(-1, 1))
            PARS.append(PA.reshape(-1, 1))

        lsts = [ARS, VRS, DRS, PVRS,PARS]
        ARS, VRS, DRS, PVRS,PARS = [Dyn.empty(lst) for lst in lsts]

        return Tt, ARS, VRS, DRS, PVRS,PARS
   

    @staticmethod
    def wave2res(arr, damp, dt, Tp, dtp):
        arr = np.asarray(arr, dtype=float).ravel()
        l = len(arr)
        D = np.zeros(l, dtype=float)
        V = np.zeros(l, dtype=float)
        A = np.zeros(l, dtype=float)
       
        # 从第一个周期点，不要0，到第最后一个周期点的值
        Tr = np.arange(dtp, Tp + dtp, dtp)
        MaxD = np.zeros(len(Tr), dtype=float)
        MaxV = np.zeros(len(Tr), dtype=float)
        MaxA = np.zeros(len(Tr), dtype=float)
        for t,T in enumerate(Tr):
            WN = 2.0 * np.pi / T
            # 防止阻尼系数damp大于1，sqrt为负代码出问题，实际结构中的阻尼系数很小
            temp = max(1-damp**2,1e-12)
            sq_temp = np.sqrt(temp)
            WD = WN * sq_temp
            et = np.exp(-damp * WN * dt)
            sint = np.sin(WD * dt)
            cost = np.cos(WD * dt)

            M = np.zeros((2, 2), dtype=float)
            M[0, 0] = et * (sint * damp / sq_temp + cost)
            M[0, 1] = et * sint / WD
            M[1, 0] = -WN * et * sint / sq_temp
            M[1, 1] = et * (-sint * damp / sq_temp + cost)

            df = (2.0 * damp**2 - 1.0) / (WN**2 * dt)
            d3t = damp / (WN**3 * dt)

            N = np.zeros((2, 2), dtype=float)
            N[0, 0] = (
                et
                * (
                    (df + damp / WN) * sint / WD
                    + (2.0 * d3t + 1.0 / WN**2) * cost
                )
                - 2.0 * d3t
            )
            N[0, 1] = (
                -et * (df * sint / WD + 2.0 * d3t * cost)
                - 1.0 / WN**2
                + 2.0 * d3t
            )
            N[1, 0] = (
                et
                * (
                    (df + damp / WN)
                    * (cost - damp / np.sqrt(temp) * sint)
                    - (2.0 * d3t + 1.0 / WN**2)
                    * (WD * sint + damp * WN * cost)
                )
                + 1.0 / (WN**2 * dt)
            )
            N[1, 1] = (
                et
                * (
                    1.0 / (WN**2 * dt) * cost
                    + sint * damp / (WN * WD * dt)
                )
                - 1.0 / (WN**2 * dt)
            )

            for j in range(l - 1):
                D[j + 1] = (
                    M[0, 0] * D[j]
                    + M[0, 1] * V[j]
                    + N[0, 0] * arr[j]
                    + N[0, 1] * arr[j + 1]
                )
                V[j + 1] = (
                    M[1, 0] * D[j]
                    + M[1, 1] * V[j]
                    + N[1, 0] * arr[j]
                    + N[1, 1] * arr[j + 1]
                )
                A[j + 1] = (
                    -2.0 * damp * WN * V[j + 1]
                    - WN**2 * D[j + 1]
                )
              
            MaxD[t] = np.max(np.abs(D))
            MaxV[t] = np.max(np.abs(V))
            MaxA[t] = np.max(np.abs(A))
                    
            D.fill(0.0)
            V.fill(0.0)
            A.fill(0.0)


        wn = 2.0 * np.pi / Tr
        
        MaxPV = MaxD * wn
        MaxPA = MaxPV * wn
        return MaxA, MaxV, MaxD, MaxPA, MaxPV

def loadfile(file_path: str) -> np.ndarray:
    """
    读取地震波文件。
    支持：
    - 单列 txt/csv
    - 多列 txt/csv
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        data = pd.read_csv(file_path, header=None).values
    else:
        data = np.loadtxt(file_path)

    data = np.asarray(data, dtype=float)
    return data


def save_csv(output_path: str, T, ARS, VRS, DRS, PVRS,PARS):
    data = pd.DataFrame({
        "T": T[:, 0],
        "ARS": ARS[:, 0],
        "VRS": VRS[:, 0],
        "DRS": DRS[:, 0],
        "PVRS": PVRS[:, 0],
        "PARS":PARS[:, 0]
    })
    data.to_csv(output_path, index=False, encoding="utf-8-sig")

def main():
    if len(sys.argv) < 7:
        print("参数不够：")
        print("python reaction.py <wave_file> <damp> <dt> <Tp> <dtp> <output_csv>")
        sys.exit(1)

    wave_file = sys.argv[1]
    damp = float(sys.argv[2])
    dt = float(sys.argv[3])
    Tp = float(sys.argv[4])
    dtp = float(sys.argv[5])
    output_csv = sys.argv[6]
    wave = loadfile(wave_file)
    T, ARS, VRS, DRS, PVRS,PARS = Dyn.spectrum(wave, damp, dt, Tp, dtp)
    save_csv(output_csv, T, ARS, VRS, DRS, PVRS,PARS)
    print(f"计算完成，文件已保存在: {output_csv}")
if __name__ == "__main__":
    main()