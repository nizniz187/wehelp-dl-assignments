import runpy

try:
  runpy.run_path("task1.py")
except Exception as e:
  print(f'Task 1 failed:\n{e}')