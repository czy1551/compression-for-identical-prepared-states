import sys
import subprocess
import numpy as np
import math

# Ensure MindQuantum is installed

try:
    from mindquantum.engine import CircuitEngine
    from mindquantum.core.gates import UnivMathGate, Measure
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mindquantum"])
    from mindquantum.engine import CircuitEngine
    from mindquantum.core.gates import UnivMathGate, Measure


def compress_qubit_compression(qubit_state: np.ndarray,
                               n: int):
    """
    输入:
      qubit_state: np.ndarray, 初始单 qubit 态 [alpha, beta]
      n: int, 重数（复制个数），必须 >= 2
    返回:
      circ: Circuit, 完整的量子线路
      compressed_state: np.ndarray, 测量后压缩寄存器的振幅
    """
    # 参数校验
    if not (isinstance(qubit_state, np.ndarray) and qubit_state.shape == (2,)):
        raise ValueError("qubit_state 必须为长度为2的 numpy 数组")
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n 必须为正整数")
    if n < 2:
        raise ValueError("n 必须大于等于2，以执行压缩操作")

    # 1. 初始化并分配 n 个 qubit
    eng = CircuitEngine()
    data_qubits = eng.allocate_qureg(n)
    circ = eng.circuit

    # 2. 状态制备
    alpha, beta = qubit_state
    prep_mat = np.array([[alpha, -np.conj(beta)],
                         [beta,  np.conj(alpha)]], dtype=complex)
    UnivMathGate('Prep', prep_mat).on(data_qubits)

    # 3. 动态生成并插入 Schur 子门（占位示例）
    for a in range(3, n + 1):
        for b in range(2, a):
            mat_ab = np.eye(8, dtype=complex)
            x = math.sqrt(math.comb(b, a - 1))
            y = math.sqrt(math.comb(b + 1, a - 1))
            z = math.sqrt(math.comb(b + 1, a))
            mat_ab[3, 3] = y / z
            mat_ab[3, 6] = -x / z
            mat_ab[6, 3] = x / z
            mat_ab[6, 6] = y / z
            UnivMathGate(f'G{a}_{b}', mat_ab).on([data_qubits[b-1], data_qubits[b], data_qubits[a]])
        mat2 = np.eye(8, dtype=complex)
        mat2[1, 1] = math.sqrt((a - 1) / a)
        mat2[1, 5] = -math.sqrt(1 / a)
        mat2[5, 1] = math.sqrt(1 / a)
        mat2[5, 5] = math.sqrt((a - 1) / a)
        UnivMathGate(f'H{a}', mat2).on([data_qubits[1], data_qubits[a-1],data_qubits[a]])

    # 4. 测量压缩寄存器前 m 个 qubit
    m = int(math.ceil(math.log2(n + 1)))
    for q in data_qubits:
        circ += Measure.on(q)

    # 5. 运行模拟并读取振幅
    amplitudes = eng.get_qs(circ)
    compressed_state = amplitudes[:2**m]
    return circ, compressed_state


if __name__ == '__main__':
    # 示例用法
    psi = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
    n = 4
    circuit, comp_state = compress_qubit_compression(psi, n)
    print(circuit)
    print('Compressed amplitudes:', comp_state)
