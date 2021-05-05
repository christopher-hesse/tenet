import subprocess as sp

sp.run(["zig", "build", "test"], check=True)
