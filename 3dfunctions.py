from Bio.PDB import *
import numpy as np


def DSSP_worker( inq, retq, pdbdum , runhhblits = False):

    print('dssp worker')
    p = PDBParser()
    ppb = CaPPBuilder()

    done = False
    while done == False:
        #parse struct
        #run dssp
        name =
        structure = p.get_structure(name, file)
        dists = {}
        for alphas in ppb.build_peptides(structure):
            #run dssp on struct get solv, SS, phi, psi and distmat
            dssp = DSSP(model, file)
            dssp_df = pd.DataFrame.from_dict({k[1]:dict(zip( header, dssp[k])) for k in list(dssp.keys()) }, orient= 'index')
            alphas = [ r for a in  ppb.build_peptides(structure) for r in a ]
            dist = np.asarray([[np.linalg.norm(a1['CA'].get_vector() - a2['CA'].get_vector()) if i < j else 0 for i,a1 in enumerate(alphas)] for j,a2 in enumerate(alphas) ] )
            dist += dist.T
            #get sequence fasta from model
            query =  '>'+model + '\n' + ''.join( dssp_df.Amino_acid.to_list())

            if runhhblits:
                pass
                #get uniclust or metaclust hmm

            else:
                #or use ecod hmm instead

            retq.put([aln, dssp, distmat])
            #return X = aln , Y = [distmat,dssp]
    print('done generating')
