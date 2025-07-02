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
    from mindquantum.core.gates import UnivMathGate, Measure, SWAP,X
    from mindquantum.simulator import Simulator


from mindquantum.core.gates import X



from mindquantum.core.gates import X
from mindquantum.core.circuit import Circuit


def pattern_controlled_ops(circ: "Circuit", i: int, n_total: int):
    #在量子线路 ``circ`` 中追加一些control。

    # ---- 参数校验 ----
    if not (0 <= i < n_total - 2):
        raise ValueError("i 必须满足 0 ≤ i < n_total - 2")


    k = n_total - i
    bits = bin(k)[2:][::-1]           
    n_bits = len(bits)

    controls = []      
    zero_qubits = []   

    # ---- 步骤 1：若 b_j == 1，则以 i 控制翻转 q_j ----
    for j, bit in enumerate(bits):
        q = n_total - 1 - j           
        controls.append(q)
        if bit == '1':
            circ += X.on(q, i)        

    # ---- 步骤 2a：把 |0⟩ 控制位临时翻到 |1⟩ ----
    for j, bit in enumerate(bits):
        if bit == '0':
            q = n_total - 1 - j
            circ += X.on(q)           # |0⟩ → |1⟩
            zero_qubits.append(q)

    # ---- 步骤 2b：多控制 X 翻转目标比特 i ----
    circ += X.on(i, controls)        

    # ---- 步骤 2c：复原临时翻转的控制位 ----
    for q in zero_qubits:
        circ += X.on(q)





def compress_qubit_compression(qubit_state: np.ndarray, n: int, shots: int = 1024):
    #构造压缩线路
    if not (isinstance(qubit_state, np.ndarray) and qubit_state.shape == (2,)):
        raise ValueError("qubit_state 必须为长度为 2 的 numpy 数组")
    if not isinstance(n, int) or n < 2:
        raise ValueError("n 必须为 >= 2 的整数")

    # ── 1. 构建电路 ───────────────────────────────────────────
    eng = CircuitEngine()
    eng.allocate_qureg(n)  # qubit_id = 0 … n‑1
    circ = eng.circuit

    # (1) 状态制备
    alpha, beta = qubit_state.astype(complex)
    prep = np.array([[alpha, -np.conj(beta)], [beta, np.conj(alpha)]], dtype=complex)
    for i in range(n):
        circ += UnivMathGate("Prep", prep).on(i)


    # (2) 压缩
    Init = np.array([[1,0,0,0],[0,0,0,1],[0,1/np.sqrt(2),1/np.sqrt(2),0],[0,1/np.sqrt(2),-1/np.sqrt(2),0]])
    circ += UnivMathGate("Init", Init).on((1, 0))
    gate=circ[-1]
    print(gate.matrix())
    circ += UnivMathGate("Init2", np.eye(2)).on(2)

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
            circ += UnivMathGate(f"G{a}_{b}", mat_ab).on((a - 1, b, b - 1))
            # print(circ[-1].matrix())        
        
        mat2 = np.eye(8, dtype=complex)
        mat2[1, 1] = 0
        mat2[3, 3] = 0
        mat2[1, 3] = 1
        mat2[3, 1] = math.sqrt((a - 1) / a)
        mat2[3, 4] = -math.sqrt(1 / a)
        mat2[4, 1] = math.sqrt(1 / a)
        mat2[4, 4] = math.sqrt((a - 1) / a)
        circ += UnivMathGate(f"H{a}", mat2).on((a - 1, a - 2, 0))
        # print(circ[-1].matrix())


    for i in range(math.floor(n/2)):
        circ += SWAP.on((i, n-i-1))
    # print(circ.matrix())


    # 翻转
    for i in range(n-3, -1, -1):
        pattern_controlled_ops(circ, i , n)
        



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

    # #test
    # running = circ[:0]                        # 空电路
    # for idx, gate in enumerate(circ):
    #     running += gate
    #     try:
    #         U = running.matrix()
    #     except AttributeError:
    #         U = Simulator('mqvector', n).get_u(running)

    #     print(f"\n── Step {idx+1}  Gate: {gate}")
    #     print(U)


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

