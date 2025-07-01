import sys
import subprocess
import numpy as np
import math
import copy

# Ensure MindQuantum is installed
try:
    from mindquantum.engine import CircuitEngine
    from mindquantum.core.gates import UnivMathGate, Measure, SWAP,X
    from mindquantum.simulator import Simulator
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mindquantum"])
    from mindquantum.engine import CircuitEngine
    from mindquantum.core.gates import UnivMathGate, Measure
    from mindquantum.simulator import Simulator


from mindquantum.core.gates import X


def pattern_controlled_ops(circ: "Circuit", i: int, n_total: int):
    """根据整数 **n_total - i + 1** 生成两段受控翻转规则。

    设
        val = n_total - i + 1
        w   = bin(val)[2:]          # 去掉 0b 前缀的二进制串（高位在左）
        k   = len(w)

    规则 1
    -------
    以 *第 ``i`` 个 qubit* 为**单控制**；若 ``w`` 的 **第 j 位**\ (j 从 1 起，最低位 j=1) = 1，
    就对 **第 ``n_total - j`` 个 qubit** 执行一次 `X`。

    规则 2
    -------
    以 **最后 ``k`` 个 qubit** 为多控制集，对 *第 ``i`` 个 qubit* 做一次 `X`。
    多控制的每个位按 ``w`` 的对应位决定：

    - 位值 = '1' → **正控**（要求该 qubit 为 |1⟩）
    - 位值 = '0' → **负控**（要求该 qubit 为 |0⟩），实现：操作前后各加一次临时 `X`。

    参数
    ------
    circ     :  目标 `Circuit`
    i        :  既是控制/目标 qubit 的 **索引**，也是决定 `val` 的位置参数  
                (0 ≤ i < n_total)
    n_total  :  电路中 qubit 总数 (必须 ≥ i+1 且 ≥ k)
    """

    # -------- 参数校验 --------
    if not (0 <= i < n_total):
        raise ValueError("i 必须在 0 .. n_total-1 之间")

    val = n_total - i  # 新规则：用 n-i+1 转二进制
    if val <= 2:
        raise ValueError("n_total - i + 1 必须 > 2，才能满足题设")

    w = format(val, "b")   # 二进制字符串，高位在左
    k = len(w)
    if k > n_total:
        raise ValueError("二进制位数 k 超过了 qubit 总数 n_total")

    # -------- 规则 1：单控翻转 --------
    for j, bit in enumerate(reversed(w), start=1):  # j=1 是最低位
        if bit == '1':
            target = n_total - j  # 第 n_total-j 个 qubit（0‑基）
            if target != i:       # 避免控制/目标重合
                circ += X.on(target, i)

    # -------- 规则 2：多控翻转第 i 个 qubit --------
    last_k = list(range(n_total - k, n_total))        # 最后 k 个 qubit (低位在前)
    bits_low_first = list(reversed(w))                # 与 last_k 对位

    # 处理负控位：先翻转成 1，记录待复原列表
    revert = []
    for qb, b in zip(last_k, bits_low_first):
        if b == '0':
            circ += X.on(qb)
            revert.append(qb)

    circ += X.on(i, last_k)  # 多控 X

    # 回复临时翻转
    for qb in revert:
        circ += X.on(qb)



def compress_qubit_compression(qubit_state: np.ndarray, n: int, shots: int = 1024):
    """构造 Schur‑变换压缩电路，输出：
    1. Circuit 对象（不含测量门）
    2. |Ψ⟩ 的完整振幅向量（长度 2^n）
    3. 测量所有 qubit 后的计数统计（shots 次采样）
    """
    if not (isinstance(qubit_state, np.ndarray) and qubit_state.shape == (2,)):
        raise ValueError("qubit_state 必须为长度为 2 的 numpy 数组")
    if not isinstance(n, int) or n < 2:
        raise ValueError("n 必须为 >= 2 的整数")

    # ── 1. 构建电路 ───────────────────────────────────────────
    eng = CircuitEngine()
    eng.allocate_qureg(n)  # qubit_id = 0 … n‑1
    circ = eng.circuit

    # (1) 状态制备 |ψ⟩⊗n
    alpha, beta = qubit_state.astype(complex)
    prep = np.array([[alpha, -np.conj(beta)], [beta, np.conj(alpha)]], dtype=complex)
    for i in range(n):
        circ += UnivMathGate("Prep", prep).on(i)

    # (2) 伪 Schur 变换占位
    Init = np.array([[1,0,0,0],[0,0,0,1],[0,1/np.sqrt(2),1/np.sqrt(2),0],[0,1/np.sqrt(2),-1/np.sqrt(2),0]])
    circ += UnivMathGate("Init", Init).on((0, 1))
    for a in range(3, n + 1):
        for b in range(1, a-1):
            mat_ab = np.eye(8, dtype=complex)
            x = math.sqrt(math.comb(a - 1, b))
            y = math.sqrt(math.comb(a - 1, b + 1))
            z = math.sqrt(math.comb(a, b + 1))
            mat_ab[2, 2] = y / z
            mat_ab[2, 5] = x / z
            mat_ab[5, 2] = -x / z
            mat_ab[5, 5] = y / z

            circ += UnivMathGate(f"G{a}_{b}", mat_ab).on((b - 1, b, a - 1))
        mat2 = np.eye(8, dtype=complex)
        mat2[0, 0] = math.sqrt((a - 1) / a)
        mat2[0, 4] = math.sqrt(1 / a)
        mat2[4, 0] = -math.sqrt(1 / a)
        mat2[4, 4] = math.sqrt((a - 1) / a)
        circ += UnivMathGate(f"H{a}", mat2).on((0, a - 2, a - 1))

    for i in range(math.floor(n/2)):
        circ += SWAP.on((i, n-i-1))


    # i 由 1 到 n-2（题设），实际控制/目标 qubit = i‑1
    for i in range(1, n - 1):
        pattern_controlled_ops(circ, i - 1, n)
        



    # ── 2. 获取完整振幅 ───────────────────────────────────────
    sim_vec = Simulator("mqvector", n)
    sim_vec.apply_circuit(circ)
    full_state = sim_vec.get_qs()  # 长度 2^n

        # ── 3. 添加测量门并采样 ──────────────────────────────────
    circ_meas = copy.deepcopy(circ)
    for i in range(n):
        circ_meas += Measure().on(i)

    # 先让模拟器执行电路，再采样 shots 次
    sim_meas = Simulator("mqvector", n)
    counts = sim_meas.sampling(circuit=circ_meas, shots=shots)


    return circ, full_state, counts


if __name__ == "__main__":
    a = float(input("请输入 alpha（实数）："))
    b = float(input("请输入 beta（实数）："))
    n = int(input("请输入复制个数 n："))
    raw_shots = input("请输入测量 shots 次数 (默认 1024)：").strip()
    shots = 1024 if raw_shots == "" else int(raw_shots)
    print("DEBUG main shots:", shots)

    psi = np.array([a, b])
    circuit, amps, counts = compress_qubit_compression(psi, n, shots)

    print("\n=== 量子线路（不含测量门） ===")
    print(circuit)

    print("\n=== |Ψ⟩ 振幅向量 ===")
    np.set_printoptions(suppress=True)
    print(amps)

    print("\n=== 测量统计 (shots = {}) ===".format(shots))
    print(counts)
