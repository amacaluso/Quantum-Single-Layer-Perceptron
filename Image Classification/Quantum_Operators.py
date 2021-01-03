from qiskit import *
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import BasicAer, execute, IBMQ, Aer
from qiskit.circuit import Gate
from qiskit.quantum_info.operators import Operator
from qiskit.aqua.components.optimizers import ADAM, CG, AQGD


def state_preparation(a, circuit, target):
    a = 2 * a
    circuit.ry(a[0], target[0])

    circuit.cx(target[0], target[1])
    circuit.ry(a[1], target[1])
    circuit.cx(target[0], target[1])
    circuit.ry(a[2], target[1])

    circuit.x(target[0])
    circuit.cx(target[0], target[1])
    circuit.ry(a[3], target[1])
    circuit.cx(target[0], target[1])
    circuit.ry(a[4], target[1])
    circuit.x(target[0])

    return circuit


def get_Sx(ang=None, x=None, pad=True, circuit=False):
    backend = Aer.get_backend('unitary_simulator')

    if pad==True:
        q = QuantumRegister(2)
        qc = QuantumCircuit(q)
        qc = state_preparation(ang, qc, [0, 1])
    elif pad==False:
        x = x.astype(complex)
        q = QuantumRegister(1)
        qc = QuantumCircuit(q)
        qc.initialize(x, [0])

    job = execute(qc, backend)
    result = job.result()
    U = result.get_unitary(qc)
    S = Operator(U)
    
    if circuit==True:
        return qc
    else: 
        return S

def linear_operator(param, pad=True, circuit = False):
    backend = Aer.get_backend('unitary_simulator')
    '''pad variable influences the size of params vector'''
    if pad==True:
        data = QuantumRegister(2)
        qc = QuantumCircuit(data)
        qc.u3(param[0], param[1], param[2], data[0])
        qc.u3(param[3], param[4], param[5], data[1])
        qc.cx(data[0], data[1])
    elif pad==False:
        data = QuantumRegister(1)
        qc = QuantumCircuit(data)
        qc.u3(param[0], param[1], param[2], data)
    
    job = execute(qc, backend)
    result = job.result()
    
    if circuit==True:
        return qc
    else: 
        U = result.get_unitary(qc)
        G = Operator(U)
        return G

def sigma(pad=True, circuit = False):
    
    backend = Aer.get_backend('unitary_simulator')
    
    if pad==True:
        data = QuantumRegister(2)
        qc = QuantumCircuit(data)
        qc.id(data)
    if pad==False:
        data = QuantumRegister(1)
        qc = QuantumCircuit(data)
        qc.id(data)

    job = execute(qc, backend)
    result = job.result()
    
    U = result.get_unitary(qc)
    I = Operator(U)

    if circuit==True:
        return qc
    else: 
        return I

def R_gate(beta, circuit = False):
    
    backend = Aer.get_backend('unitary_simulator')
    
    control = QuantumRegister(1)
    qc = QuantumCircuit(control)
    qc.ry(beta, control)

    job = execute(qc, backend)
    result = job.result()

    U = result.get_unitary(qc)
    R = Operator(U)
    
    if circuit==True:
        return qc
    else: 
        return R


def create_circuit(parameters=None, x=None, pad=True):
    n_params=len(parameters)
    
    beta = parameters[0]
    theta1 = parameters[1:int((n_params+1)/2)]
    theta2 = parameters[int((n_params+1)/2):int(n_params)]

    control = QuantumRegister(1, 'control')
    data = QuantumRegister(2, 'data')
    temp = QuantumRegister(2, 'temp')
    c = ClassicalRegister(1)
    qc = QuantumCircuit(control, data, temp, c)

    S = get_Sx(ang=x, x=None, pad=True, circuit=True)
    R = R_gate(beta, circuit=True)
    sig=sigma(pad, circuit=True)


    G1 = linear_operator(theta1, pad=True, circuit=True)   
    G2 = linear_operator(theta2, pad=True, circuit=True)

    qc.compose(R, qubits=control, inplace=True)
    qc.compose(S, qubits=data, inplace=True)
    
    qc.barrier()
    qc.cswap(control, data[0], temp[0])
    qc.cswap(control, data[1], temp[1])
    qc.barrier()
    
    qc.compose(G1, qubits=data, inplace=True)
    qc.compose(G2, qubits=temp, inplace=True)

    qc.barrier()
    qc.cswap(control, data[1], temp[1])
    qc.cswap(control, data[0], temp[0])

    qc.barrier()

    qc.compose(sig, qubits=data, inplace=True)
    qc.barrier()
    qc.measure(data[0], c)
    return qc