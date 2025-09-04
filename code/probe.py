import random
import pandas as pd
import json



def complexity_function_datast(data_path: str, sample_size: int = 300 , num_class: int = 3, seed: int = 0, test = False, has_label = False):

    with open(data_path, "r") as fp:
        data = pd.DataFrame(json.load(fp))
    fp.close()

    random.seed(seed)
    dataset = {}

    if test == False:

        simple_claims = random.sample(data[data["complexity"] == 0].claim.values.tolist(), sample_size//num_class)
        intermediate_claims = random.sample(data[data["complexity"] == 1].claim.values.tolist(), sample_size//num_class)
        complex_claims = random.sample(data[data["complexity"] >= 2].claim.values.tolist(), sample_size//num_class)

        dataset["simple_claims"] = simple_claims
        dataset["intermediate_claims"] = intermediate_claims
        dataset["complex_claims"] = complex_claims
    
    elif test == True:
        dataset["claims"] = data.claim.values.tolist()
        if has_label == True:
            dataset["labels"] = data.complexity.values.tolist()
    

    return dataset
    