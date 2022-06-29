from pyinstrument import Profiler

profiler = Profiler()
profiler.start()

# 这里是你要分析的代码

profiler.stop()

profiler.print()