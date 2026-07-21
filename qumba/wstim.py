#!/usr/bin/env python


import stim


def test():

    circuit = stim.Circuit()

    circuit.append("R", [0, 1])

    
    # First, the circuit will initialize a Bell pair.
    circuit.append("H", [0])
    circuit.append("X_ERROR", [0, 1], 0.05)
    circuit.append("CNOT", [0, 1])
    
    # Then, the circuit will measure both qubits of the Bell pair in the Z basis.
    circuit.append("M", [0, 1])

    print(circuit)

    print(len(circuit))
    for op in circuit:
        print("\t", op)

    N = 1000
    sampler = circuit.compile_sampler()
    result = (sampler.sample(shots=N))
    result = (result.astype(int))
    stats = ( result.sum(axis=1) % 2 )

    print( stats.sum() / N )



if __name__ == "__main__":

    from time import time
    from qumba.argv import argv

    start_time = time()


    profile = argv.profile
    name = argv.next() or "test"
    _seed = argv.get("seed")
    if _seed is not None:
        print("seed(%s)"%(_seed))
        seed(_seed)

    if profile:
        import cProfile as profile
        profile.run("%s()"%name)

    elif name is not None:
        fn = eval(name)
        fn()

    else:
        test()


    t = time() - start_time
    print("OK! finished in %.3f seconds\n"%t)


