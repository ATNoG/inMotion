"""Canonical edge-quantization campaign: accuracy, devices, power, interference.

The submodules are intentionally thin and separately testable:

``contracts``  names, schemas, device specs, power constants, margins
``accuracy``   artifact export, artifact-based metrics, bootstrap CIs, windows
``transport``  local and SSH execution, file transfer, no stored secrets
``bench``      load / latency / RSS / CPU matrices on a device
``replay``     on-device predictions versus Python, label by label
``power``      measured sources first, labelled model last
``stress``     controlled synthetic load and the interference protocol
``analysis``   paired/block-bootstrap statistics and equivalence verdicts
``figures``    the eight publication figures, from the tables only
``campaign``   phase orchestration, run directory, manifest, resume
"""
