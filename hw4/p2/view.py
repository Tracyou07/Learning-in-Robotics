"""Launch the dm_control viewer with the trained PPO policy.

Usage:
    py -3.10 view.py             # uses actor.pt by default
    py -3.10 view.py actor.pt    # explicit checkpoint
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    'ppo_main', os.path.join(HERE, '18330723_hw4_p2.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

ckpt = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'actor.pt')
mod.evaluate(ckpt)
