# QuantumSimulator (qsim)

Quantum computers are at the forefront of technology today. They bring with them the promise of huge computational power, enough to rival even the fastest super computers we have today. While these idealistic, fault-tolarent quantum computers are still out of reach we can simulate how they would behave on classical computers today (with sufficient classical resources). In this repo I create a python module to simulate a quantum computer and compare my results with IBM's platform Qiskit.

## Getting started with Qsim 

Simply clone this repository locally, open a terminal in this folder and install it using the command:    `pip install -e .`

To begin, simply import qsim as follows : `from qsim import qsim` 

There are two main classes in this module: Qbit and Circuit. 

## Qbit Class:  
A Qbit object represents a quantum state in 2 dim Hilbert Space. As such it requires two complex coefficients (`c0, c1`) to represent the complex probability amplitudes of the two basis states. *Note the squared sum of these values must be equal to one*. 

An instance of Qbit can be constructed as follows:  ` psi = qsim.Qbit(1/2, (3)**(0.5)/2) `  c0 = 1/2 , c1 = sqrt(3)/2 

We can visually display this instance by printing it:  ` print(psi) ` 

![print qbit](/images/print_qbit.PNG?raw=true)

Once we have constructed a state we can simulate making repeated independant measurements of the state as follows: 

` results = psi.measure(shots=10) `  shots is an optional parameter which determines the number of independant measurements

` print(results) ` 

![print_qbit_results](/images/print_qbit_results.PNG?raw=true)

## Circuit Class:
A Circuit object represents the quantum circuit we would be running on the quantum computer. We initialize it by passing in the number of qbits we want to simulate. More qbits = longer processing time (it grows exponentially). 

An instance of Circuit can be constructed as follows: `circ = qsim.Circuit(3) ` 

We can then visualize our circuit by: ` print(circ) ` 

An important attribute of the circuit is its state vector, which is a quantum state in 2^n dim Hilbert Space, where n represents the number of qbits in the circuit. We can visualize this state vector by calling; ` print(circ.circuit_state) ` 

![circuit and state vector](/images/circ_state.PNG?raw=true)

No quantum circuit is complete without the application of quantum gates. These are normalized unitary matricies which act on the qbits to transform their probability amplitudes. In some cases the quantum gates get more complicated (double and triple qbit gates). I have implemented a series of common gates such as the hadamard gate, the pauli-X, pauli-Y, Pauli-Z gates and the Phase gate. I have also implemented the 2 qbit controlled version of these gates: (CX, CY, CZ, CH, CS). For additional information about these gates check out the wikipedia page: [Quantum Logic Gates](https://en.wikipedia.org/wiki/Quantum_logic_gate)

For the single qbit gates, we simply specify the index of the qbit we wish to apply the gate to.

` circ.x(0) ` applies a pauli-X gate to qbit 1 (index at 0) 

` circ.y([1,2]) ` we can also pass a list of indicies in order to apply the gate to multiple qbits 

Other possible gates include: 
- ` Circuit.z(index) ` pauli-Z
- ` Circuit.h(index) ` hadamard 
- ` Circuit.s(index) ` phase  

We can also apply control-unitary gates as followed: 

` circ.cz(0, 2) `   This is a controlled Z gate with the 1st qbit as the control and the 3rd qbit as the target

` circ.ch(2, 0) `   This is a controlled hadamard gate with the 3rd qbit as control and the 1st qbit as target

Other possible gates include: 
- ` Circuit.cx(control_index, target_index) `
- ` Circuit.cy(control_index, target_index) `
- ` Circuit.cs(control_index, target_index) `

Currently our circuit looks like this: 

![print circuit](/images/display_circuit.PNG?raw=true)

*we have not run the circuit so our circuit state is still initialized to the 0 state for all of our qbits* 

To run the circuit and update the circuit we call: ` circ.run() ` there is an optional boolean parameter called status (default: status=True) which displays the status while it is processing. Once it is finished our circuit state has changed to reflect the application of the circuit. *Note, our circuit has been erased after the run call, but the state vector is safe so that we don't have to re-run the circuit each time we want to measure it.*

We can measure our circuit by: 

` counts = circ.simulate(shots=100) `

` circ.plot_counts(counts) ` 

![final results](/images/results.PNG?raw=true)

## Acknowledgements: 
- Qiskit documentation : For helping me learn some of the mathematical background of quantum computing 
- QOSF : for providing many resources for me to reference 
- Ivan Sharankov : for your code review and assistance implementing the qbit_iterator function 

If you have any further questions about this project, feel free to email me @ jbsoni@uwaterloo.ca 
