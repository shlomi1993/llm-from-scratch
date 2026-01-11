from datasets import load_dataset

ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca")

ds.save_to_disk(".")
