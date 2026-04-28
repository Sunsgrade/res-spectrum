# reaction_core.py
import os
import numpy as np
import pandas as pd

class Dyn:
    @staticmethod
    def empty(lst):
        return np.hstack(lst) if lst else np.empty((0, 0))
    
    @staticmethod
    def spectrum(wave, damp, dt, Tp=6, dtp=0.02):
        arr = np.asarray(wave, dtype=float)

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

        lsts = [ARS, VRS, DRS, PVRS, PARS]
        ARS, VRS, DRS, PVRS, PARS = [Dyn.empty(lst) for lst in lsts]

        return Tt, ARS, VRS, DRS, PVRS, PARS

    @staticmethod
    def wave2res(arr, damp, dt, Tp, dtp):
        arr = np.asarray(arr, dtype=float).ravel()
        l = len(arr)

        D = np.zeros(l, dtype=float)
        V = np.zeros(l, dtype=float)
        A = np.zeros(l, dtype=float)

        Tr = np.arange(dtp, Tp + dtp, dtp)

        MaxD = np.zeros(len(Tr), dtype=float)
        MaxV = np.zeros(len(Tr), dtype=float)
        MaxA = np.zeros(len(Tr), dtype=float)

        for t, T in enumerate(Tr):
            WN = 2.0 * np.pi / T

            temp = max(1 - damp**2, 1e-12)
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


def load_wave_file(file_path: str) -> np.ndarray:
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        data = pd.read_csv(file_path, header=None).values
    else:
        data = np.loadtxt(file_path)

    data = np.asarray(data, dtype=float)

    if data.size == 0:
        raise ValueError("文件为空，无法计算。")

    return data

def round_list(values, digits=6):
    """
    将 numpy 数组或列表转换为普通 list，并进行四舍五入。
    避免 API 返回 0.12000000000000001 这类长小数。
    """
    return [round(float(v), digits) for v in values]

def calculate_response_spectrum(
    wave,
    damp: float,
    dt: float,
    Tp: float = 6.0,
    dtp: float = 0.02,
):
    """
    反应谱计算核心函数。

    参数：
    wave: 地震波加速度数据，可以是一维或二维数组
    damp: 阻尼比，例如 0.05
    dt: 时程步长
    Tp: 最大周期
    dtp: 周期间隔

    返回：
    dict，可直接给 API 返回 JSON
    """

    if damp < 0:
        raise ValueError("阻尼比 damp 不能小于 0。")

    if dt <= 0:
        raise ValueError("时间步长 dt 必须大于 0。")

    if Tp <= 0:
        raise ValueError("最大周期 Tp 必须大于 0。")

    if dtp <= 0:
        raise ValueError("周期间隔 dtp 必须大于 0。")

    T, ARS, VRS, DRS, PVRS, PARS = Dyn.spectrum(wave, damp, dt, Tp, dtp)

    result = {
        "T": round_list(T[:, 0], 6),
        "ARS": round_list(ARS[:, 0], 6),
        "VRS": round_list(VRS[:, 0], 6),
        "DRS": round_list(DRS[:, 0], 6),
        "PVRS": round_list(PVRS[:, 0], 6),
        "PARS": round_list(PARS[:, 0], 6),
    }
    return result


def result_to_dataframe(result: dict) -> pd.DataFrame:
    return pd.DataFrame(result)


def save_result_csv(result: dict, output_path: str):
    df = result_to_dataframe(result)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")