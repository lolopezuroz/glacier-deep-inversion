from functions.importation import json, argparse
from training.train import train
from drawing.draw import draw

parser = argparse.ArgumentParser()
parser.add_argument("--parameters_path", type=str)
args = parser.parse_args()


with open(args.parameters_path) as file:
    parameters = json.load(file)

train(parameters)
#draw(parameters)
