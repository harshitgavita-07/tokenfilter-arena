import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from tokenfilter.data import write_all

d = write_all("runs")
print(f"world: {len(d['world']['train'])} train + {len(d['world']['test'])} test entities")
print(f"pretrain sentences: {len(d['pretrain'])}; sft items: {len(d['sft'])} "
      f"({sum(i['corrupted'] for i in d['sft'])} corrupted)")
