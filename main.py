import runpy

try:
  runpy.run_path("task123.py")
except Exception as e:
  print(f'Task 1~3 failed:\n{e}')