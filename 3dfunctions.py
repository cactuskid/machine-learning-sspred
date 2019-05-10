from Bio.PDB import *
import numpy as np


def generate_DSSP_distmat( pdbdum ):

    p = PDBParser()
    ppb = CaPPBuilder()

    for struct in pdbdump:
        #parse struct

        #run dssp
        name =
        structure = p.get_structure(name, file)
        for model in structure:
            dists = {}
            for alphas in ppb.build_peptides(structure):
                dist = np.asarray([[np.norm(a1.get_vector() - a2.get_vector()) if i < j else 0 for i,a1 in enumerate(alphas)] for j,a2 in enumerate(alphas) ] )
                dist += dist.T
                dists[chain]=(dist)
            #run dssp on struct get solv, SS, phi, psi
            dssp = DSSP(model, file)
            seq =

            #run hhblits her and get HMM

            yield dssp, distmat
