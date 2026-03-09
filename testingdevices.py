import jax
import os
import psutil
import subprocess
#os.environ['CUDA_VISIBLE_DEVICES'] = '20'
print(psutil.cpu_count())
print(psutil.Process().cpu_affinity())
print(psutil.Process().cpu_affinity())
print(psutil.cpu_freq(percpu=True))
#print(subprocess.run(["powershell", "-Command", "Get-Counter '\\Processor(_Total)\\% Processor Time' -MaxSamples 1"],capture_output=True))