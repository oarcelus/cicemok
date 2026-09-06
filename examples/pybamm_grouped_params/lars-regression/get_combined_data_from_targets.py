import os
import pickle
import multiprocessing
import dill
import numpy as np
import matplotlib.pyplot as plt


def main():
    lnsamples = [50, 100, 200, 500, 1000, 2000, 5000]

    repetitions = 20
    targets = [
        ([0, 1, 2, 3, 4], "target0-4"),
        ([5, 6, 7, 8, 9], "target5-9"),
        ([10, 11, 12, 13, 14], "target10-14"),
        ([15, 16, 17, 18, 19], "target15-19"),
    ]

    for i in range(repetitions):
        for j, nsamples in enumerate(lnsamples):
            surrlist = []
            fourlist = []
            alphlist = []
            for target in targets:
                indexes = target[0]
                folder = target[1]

                for idx in indexes:
                    surrname = os.path.join(
                        os.getcwd(),
                        os.path.join(
                            folder, f"surrogate_n{nsamples}_target{idx}_reps{i}.pkl"
                        ),
                    )
                    fourname = os.path.join(
                        os.getcwd(),
                        os.path.join(
                            folder, f"fourier_n{nsamples}_target{idx}_reps{i}.pkl"
                        ),
                    )
                    alphname = os.path.join(
                        os.getcwd(),
                        os.path.join(
                            folder, f"alpha_n{nsamples}_target{idx}_reps{i}.pkl"
                        ),
                    )
                    
                    print("rep", i, "nsample", nsamples, "target", target)

                    with open(surrname, "rb") as file:
                        vals = dill.load(file)
                        ysurr = vals[1]
                    with open(fourname, "rb") as file:
                        vals = pickle.load(file)
                        four = vals[1]
                    with open(alphname, "rb") as file:
                        vals = pickle.load(file)
                        alpha = vals[1]

                    x = vals[0]
                    surrlist.append(ysurr)
                    fourlist.append(four)
                    alphlist.append(alpha)

            surrlist = [x, surrlist]
            fourlist = [x, fourlist]
            alphlist = [x, alphlist]

            surrname = os.path.join(os.getcwd(), f"surrogate_n{nsamples}_reps{i}.pkl")
            fourname = os.path.join(os.getcwd(), f"fourier_n{nsamples}_reps{i}.pkl")
            alphname = os.path.join(os.getcwd(), f"alpha_n{nsamples}_reps{i}.pkl")
            with open(surrname, "wb") as file:
                dill.dump((surrlist), file)
            with open(fourname, "wb") as file:
                pickle.dump((fourlist), file)
            with open(alphname, "wb") as file:
                pickle.dump((alphlist), file)


if __name__ == "__main__":
    main()
