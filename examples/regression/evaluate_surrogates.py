import pickle
import chaospy as cp
import numpy as np
import matplotlib.pyplot as plt

def main():
    lorder = [2, 3, 4, 5]
    lnsamples = [2000, 5000, 10000, 20000, 50000]

    for order in lorder:
        for nsamples in lnsamples:
            with open(f"surrogate_n{nsamples}_o{order}.pkl", "rb") as file:
                vals = pickle.load(file)
                ysurr = vals[1]
            with open(f"./samples_n{nsamples}_o{order}.pkl", "rb") as file:
                vals = pickle.load(file)
                ysamples = vals
            
            surrevals = np.array([ysurr(*samples) for samples in ysamples.T], dtype=float)

            with open(f"surrevals_n{nsamples}_o{order}.pkl", "wb") as file:
                pickle.dump(surrevals, file)

            print("weee")
            
    
if __name__ == "__main__":
    main()

