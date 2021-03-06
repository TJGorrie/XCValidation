from fragalysis_api import Validate, Align
from fragalysis_api.xcimporter.conversion_pdb_mol import set_up
import os
from shutil import copyfile, rmtree
import argparse
from sys import exit
import sys, json, os, glob
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
import pandas as pd
from rdkit.Geometry import Point3D
import logging

def get_3d_distance(coord_a, coord_b):
    sum_ = (sum([(float(coord_a[i])-float(coord_b[i]))**2 for i in range(3)]))
    return np.sqrt(sum_)


def xcvalidate(in_dir, out_dir, target, validate=False):
    if validate:
        validation = Validate(in_dir)
        if not bool(validation.is_pdbs_valid):
            print("Input files are invalid!!")
            logging.error("Files inside: {0}. Are invalid".format(str(in_dir)))
            #exit()
        if not validation.does_dir_exist:
            logging.error("Input Directory: {0}. Does not exist".format(str(in_dir)))
            #exit()
        if not validation.is_there_a_pdb_in_dir:
            logging.error("Input Directory: {0}. Does not contain pdb files".format(str(in_dir)))
            #exit()
    pdb_smiles_dict = {'pdb': [], 'smiles': []}
    # Creating lists of pdbs and smiles
    print('Enumerating pdbs and smiles paths')
    for f in os.listdir(in_dir):
            pdb_smiles_dict['pdb'].append(os.path.join(in_dir, f))
            #print(os.path.join(in_dir, f).replace('.pdb', '_smiles.txt'))
            if os.path.isfile(os.path.join(in_dir, f).replace('.pdb', '_smiles.txt')):
                pdb_smiles_dict['smiles'].append(os.path.join(in_dir, f).replace('.pdb', '_smiles.txt'))
            else:
                pdb_smiles_dict['smiles'].append(None)
                logging.warning("{0} is missing smiles file".format(str(f)))

    # Create output if not exist.
    if not os.path.isdir(out_dir):
        #print('Making Output Dir')
        os.makedirs(out_dir)
        os.makedirs(os.path.join(out_dir, "tmp"))   
    
    # Align structures... Make boundpdb files... Wrap into a try?
    print('Attempt to align pdbs')
    try:
        structure = Align(in_dir, pdb_ref="")
        structure.align(os.path.join(out_dir, "tmp"))
    except:
        logging.error("Unable to align pdbs in {0} together".format(str(in_dir)))
    else:
        logging.info('Successfully aligned')

    print('Copying smiles files')
    for smiles_file in pdb_smiles_dict['smiles']:
        if smiles_file:
            #print(smiles_file)
            try:
                copyfile(smiles_file, os.path.join(os.path.join(out_dir, "tmp", smiles_file.split('/')[-1])))
            #print(os.path.join(out_dir, "tmp", smiles_file.split('/')[-1]))
            except:
                logging.warning("Unable to copy {0} to {1}".format(smiles_file, "tmp"))

    
    aligned_dict = {'bound_pdb':[], 'smiles':[]}
    for f in os.listdir(os.path.join(out_dir, "tmp")):
        if '.pdb' in f:
            aligned_dict['bound_pdb'].append(os.path.join(out_dir, "tmp",f))
            if os.path.isfile(os.path.join(out_dir, "tmp",f).replace('_bound.pdb', '_smiles.txt')):
                aligned_dict['smiles'].append(os.path.join(out_dir, "tmp",f).replace('_bound.pdb', '_smiles.txt'))
            else:
                aligned_dict['smiles'].append(None)
                logging.warning("{0} is still missing smiles file".format(str(f)))
    
    print("Identifying ligands")
    for aligned, smiles in list(zip(aligned_dict['bound_pdb'], aligned_dict['smiles'])):
        try: # Redirect std:out?
            if smiles:
                new = set_up(target_name=target, infile=os.path.abspath(aligned), out_dir=out_dir, smiles_file=os.path.abspath(smiles))
            else:
                new = set_up(target_name=target, infile=os.path.abspath(aligned), out_dir=out_dir)
        except: # AssertionError: # Additional Error caused by innapropriate Valencies?
            print(aligned, "is not suitable, please consider removal or editing")
            logging.warning("{0} isn't suitable, consider removal or editing".format(str(aligned)))
            for file in os.listdir(os.path.join(out_dir, "tmp")):
                if str(aligned) in file:
                    os.remove(os.path.join(out_dir, "tmp", str(file)))
    return


def new_process_covalent(directory):
    for f in [x[0] for x in os.walk(directory)]:

        try:
            covalent = False
            #print(str(f) + '/*_bound.pdb')
            #print(glob.glob(str(f) + '/*_bound.pdb'))
            if glob.glob(str(f) + '/*_bound.pdb'):
                bound_pdb = glob.glob(str(f) + '/*_bound.pdb')[0]
                mol_file = glob.glob(str(f) + '/*.mol')[0]
                pdb = open(bound_pdb, 'r').readlines()
                for line in pdb:
                    if 'LINK' in line:
                        #print('Found Link')
                        #print(str(f))
                        zero = line[13:27]
                        one = line[43:57]
                        if 'LIG' in zero:
                            res = one
                        if 'LIG' in one:
                            res = zero
                        covalent=True

                if covalent:
                    logging.info("Found Covalent Link in " + str(f))
                    #print(str(f))
                    for line in pdb:
                        if 'ATOM' in line and line[13:27]==res:
                            res_x = float(line[31:39])
                            res_y = float(line[39:47])
                            res_z = float(line[47:55])
                            res_atom_sym = line.rsplit()[-1].rstrip()
                            atom_sym_no = pd.read_csv('/dls/science/groups/i04-1/software/tyler/xcvalidation/atom_numbers.csv', index_col=0, parse_dates=True)
                            res_atom_no = atom_sym_no.loc[res_atom_sym].number
                            res_coords = [res_x, res_y, res_z]
                            #print(res_coords)
                            atm = Chem.MolFromPDBBlock(line)
                            atm_trans = atm.GetAtomWithIdx(0)
                    mol = Chem.MolFromMolFile(mol_file)
                    # edmol = Chem.EditableMol(mol)
                    orig_pdb_block = Chem.MolToPDBBlock(mol)
                    lig_block = '\n'.join([l for l in orig_pdb_block.split('\n') if 'COMPND' not in l])
                    lig_lines = [l for l in lig_block.split('\n') if 'HETATM' in l]
                    j = 0
                    old_dist = 100
                    for line in lig_lines:
                        j += 1
                        # print(line)
                        if 'HETATM' in line:
                            coords = [line[31:39].strip(), line[39:47].strip(), line[47:55].strip()]
                            dist = get_3d_distance(coords, res_coords)
                            if dist < old_dist:
                                ind_to_add = j
                                #print(dist)
                                old_dist = dist
                    i = mol.GetNumAtoms()
                    edmol = Chem.EditableMol(mol)
                    edmol.AddAtom(atm_trans)
                    edmol.AddBond(ind_to_add - 1, i, Chem.BondType.SINGLE)
                    new_mol = edmol.GetMol()
                    conf = new_mol.GetConformer()
                    conf.SetAtomPosition(i, Point3D(res_coords[0], res_coords[1], res_coords[2]))
                    try:
                        Chem.MolToMolFile(new_mol, mol_file)
                    except ValueError:
                        Chem.MolToMolFile(new_mol, mol_file, kekulize=False)
                    
        except:
            logging.warning('Unable to create .mol file for {0}'.format(str(f))) 
            # Usually a weird error where Link is established but res does not get defined.
        else:
            logging.info("Created New .mol for " + str(f))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--in_dir",
        default=os.path.join("..", "..", "data", "xcimporter", "input"),
        help="Input directory",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        default=os.path.join("..", "..", "data", "xcimporter", "output"),
        help="Output directory",
        required=True,
    )
    parser.add_argument(
        "-v", "--validate", action="store_true", default=False, help="Validate input"
    )
    parser.add_argument("-t", "--target", help="Target name", required=True)

    args = vars(parser.parse_args())

    # user_id = args['user_id']
    in_dir = args["in_dir"]
    out_dir = args["out_dir"]
    validate = args["validate"]
    target = args["target"]

    # Validate file paths?
    #if in_dir == os.path.join("..", "..", "data", "xcimporter", "input"):
    #    print("Using the default input directory ", in_dir)
    #if out_dir == os.path.join("..", "..", "data", "xcimporter", "output"):
    #    print("Using the default input directory ", out_dir)

    if not os.path.isdir(out_dir):
        print('Making Output Dir')
        os.makedirs(out_dir)
        os.makedirs(os.path.join(out_dir, "tmp"))
        os.makedirs(os.path.join(out_dir, target)) 
    # Log file handler...
    logging.basicConfig(level=logging.DEBUG, filename = os.path.join(out_dir, 'validation.log'), filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    logging.info("Starting Validation Process")
    logging.info("input dir: {0}".format(str(in_dir)))
    logging.info("output dir: {0}".format(str(out_dir)))
    logging.info("Target Name: {0}".format(str(target)))
    logging.info("Running Input Validation?: {0}".format(str(validate)))

    #try:
    xcvalidate(in_dir=in_dir, out_dir=out_dir, target=target, validate=validate)
    #except:
    #    logging.error("Unable to align or create bound_pdb files")
    #else:
    #    logging.info('Step 1 Completed')

    pdb_file_failures = open(os.path.join(out_dir, target, 'pdb_file_failures.txt'), 'w')

    for target_file in os.listdir(os.path.join(out_dir, target)):
        if target_file != 'pdb_file_failures.txt' and len(os.listdir(os.path.join(out_dir, target, target_file))) < 2:
            rmtree(os.path.join(out_dir, target, target_file))
            logging.warning('{0} was not processed...'.format(str(target_file.split('-')[1])))
            pdb_file_failures.write(target_file.split('-')[1]+'\n')


    print('For files that we were unable to process, look at the pdb_file_failures.txt file in your results directory.'
          ' These files were unable to produce RDKit molecules, so the error likely lies in the way the ligand atoms or'
          'the connect files have been written in the pdb file')

    # Go into the output folder and attempt to parse...
    dir2 = os.path.join(out_dir, target)
    # Add more verbose outputs?
    #try:
    new_process_covalent(directory = dir2)
    #except:
    #    logging.error('Errors occured while attempting to create mol files. Either alignment failed upstream or some files were unsuitable')
    #else:
    #    logging.info('Step 2 Completed without ANY errors...')

    pdb_file_failures.close()
    logging.info("End Validation Process")






