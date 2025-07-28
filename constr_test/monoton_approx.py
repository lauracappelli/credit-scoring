from src.select_data import *
config = read_config()

dataset = generate_data(config)

print(dataset)