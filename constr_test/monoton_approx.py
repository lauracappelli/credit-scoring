from src.select_data import *
config = read_config()

dataset = generate_or_load_dataset(config)

print(dataset)