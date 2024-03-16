#!/usr/bin/env python

import warnings

# Settings the warnings to be ignored
warnings.filterwarnings('ignore')


from qjobs import QPU, Batch
qpu = QPU("H1-1E", domain="prod", local=True)
# Setting local=True means that qjobs will use PECOS to run simulations on your device
# Otherwise, it will attempt to connect to a device in the cloud

batch = Batch(qpu)


# create & measure Bell state
qasm = """
OPENQASM 2.0;
include "hqslib1.inc";

qreg q[2];
creg m[2];

h q[0];

CX q[0], q[1];

measure q -> m;
"""

# We can append jobs to the Batch object to run
batch.append(qasm, shots=10, options={"simulator": "stabilizer"})

# Submit all previously unsubmitted jobs to the QPU
batch.submit()
# Note: Each time you submit or retriece jobs, the Batch object will save itself as a pickle

# Retrieve
batch.retrieve()

print(batch.jobs)

if 0:
    #To get an individual job object you can use indexes from
    #this list of jobs. Or use a job's job id like this:
    
    j = batch["local60bb85f56c0b4d8ca6bebe49525c9373"]
    
    print(j.code)
    
    j.results
    
    batch["localeb2dbe1147fe49db80ede0a289c6008f"].params


