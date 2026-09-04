import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", required=True)
args = parser.parse_args()

obj = torch.load(args.path, weights_only=False)

print("Type:", type(obj))

if isinstance(obj, list):
    print("Length:", len(obj))
    print("First element type:", type(obj[0]))
    print("Attributes of first element:", obj[0].__dict__.keys())

elif isinstance(obj, dict):
    print("Keys:", obj.keys())

else:
    print(obj)
