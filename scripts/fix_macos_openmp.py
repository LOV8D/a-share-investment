"""Repair only the project's LightGBM wheel search path on macOS."""
import platform
import subprocess
import sysconfig
from pathlib import Path
if platform.system()=="Darwin":
    root=Path(sysconfig.get_paths()["purelib"])
    lib=root/"lightgbm/lib/lib_lightgbm.dylib"
    omp=root/"sklearn/.dylibs/libomp.dylib"
    if lib.exists() and omp.exists():
        rpath="@loader_path/../../sklearn/.dylibs"
        existing=subprocess.check_output(["otool","-l",str(lib)],text=True)
        if rpath not in existing:
            subprocess.run(["install_name_tool","-add_rpath",rpath,str(lib)],check=True)
            subprocess.run(["codesign","--force","--sign","-",str(lib)],check=True)
    print("macOS OpenMP 路径检查完成")
