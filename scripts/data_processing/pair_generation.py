import pandas as pd  
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(current_dir, "../../data/raw/data.txt"), 'r', encoding='utf-8') as file:
    pairs = file.read() 

data = []  
for line in pairs.split("\n"):  
    if line.startswith("[formal]"):  
        data.append({"text": line.replace("[formal]", "").strip(), "label": 1})  
    elif line.startswith("[informal]"):  
        data.append({"text": line.replace("[informal]", "").strip(), "label": 0})   

df = pd.DataFrame(data)  
df.to_csv("synthetic_formality_data.csv", index=False)   
