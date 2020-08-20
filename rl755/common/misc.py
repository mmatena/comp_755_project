"""Miscelaneous utilty functions."""


def divide_chunks(lis, n):
  """Yield successive n-sized from list l"""
  for i in range(0, len(lis), n):
    yield lis[i:i + n]
