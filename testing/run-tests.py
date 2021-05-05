import subprocess as sp
import sys

sp.run(["zig", "build", "test"], check=True)
