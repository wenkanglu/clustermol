import os
import numpy
import preprocessing

def qt_orginal():
    # TODO: implemnet qt orginal
    pass


def qt_like():
    # TODO: Implement qt like
    pass

def runQT(filename, destination, type):
    traj = preprocessing.preprocessing_file(filename)
    rmsd_matrix_temp = preprocessing.preprocessing_general(traj) # Need to write general pre-process.

    if type == "qt_original":
        # TODO: Implement orginal algorithm
        pass
    else:
        # TODO: Implement qt like algorithm
        pass

if __name__ == "__main__":
    runQT("MenY_reduced_100_frames.pdb", "data_dest", "qt_orginal")
