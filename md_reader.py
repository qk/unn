#!/usr/bin/env python3
import sys, os
from argparse import ArgumentParser
import re
import numpy as np
import scipy as sp
import scipy.spatial
from collections import defaultdict

# interatom. dist. in graph or normalization before

atoms = re.findall(r'[A-Z][a-z]?',
	"HHeLiBeBCNOFNeNaMgAlSiPSClArKCaScTiVCrMnFeCoNiCuZnGaGeAsSeBrKrRbSrYZrNbMoTcRuRhPdAgCdInSnSbTeIXeCsBa")
charge = defaultdict(int, zip(atoms, range(1, len(atoms)+1))) # defaults to 0

def read(data_dir="md_datasets/benzene/", name="benzene", max_atoms=None):
	# files = sorted([filename for filename in os.listdir(data_dir) if re.search(namefilter, filename)])
	print("listing files ...")
	files = sorted([filename for filename in os.listdir(data_dir)])
	with open("fileorder.txt", "w") as f:
		f.write("\n".join(files))
	if max_atoms is None:
		with open(data_dir + files[0], "r") as f:
			max_atoms = int(f.readline())
		print("automatically detected maxatoms == {}".format(max_atoms))

	N = np.empty(len(files), dtype=np.int32) # number of atoms
	E = np.empty(len(files), dtype=np.float64) # energy
	F = np.zeros([len(files), max_atoms, 3], dtype=np.float64) # forces (per atom)
	R = np.zeros([len(files), max_atoms, 3], dtype=np.float64) # coordinates
	A = np.zeros([len(files), max_atoms], dtype=str) # atom at position label
	D = np.zeros([len(files), max_atoms, max_atoms], dtype=np.float64) # interatomic distances
	Z = np.zeros([len(files), max_atoms], dtype=np.int32) # atomic charges (german: "ordnungszahl")

	num_files = len(files)
	for i,filename in enumerate(files):
		if i%100 == 0:
			print("\r%i / %i"%(i,num_files), end="")
		with open(data_dir + filename, "r") as f:
			content = f.read()
		lines = content.splitlines()
		N[i] = int(lines[0])
		energy, forces = lines[1].split(";")
		E[i] = float(energy)
		F[i] = [
			[float(floatstring) for floatstring in vectorstring.split(",")]
			for vectorstring in re.findall(r'\[(.*?)\]+', forces)
		]
		labels_and_coordinates = [re.split(r"\s+", string) for string in lines[2:]]
		A[i] = [row[0] for row in labels_and_coordinates]
		R[i] = [[float(value) for value in row[1:]] for row in labels_and_coordinates]
		D[i] = sp.spatial.distance.cdist(R[i], R[i], metric="euclidean")
		Z[i] = list(map(lambda a: charge[a], A[i]))
	print("\nsaving %s.npz ... "%name, end="")
	moleculedata = dict(N=N, E=E, F=F, R=R, A=A, D=D, Z=Z)
	np.savez(name, **moleculedata)
	print("done")
	return moleculedata

if __name__ == "__main__":
	parser = ArgumentParser(description='convert dataset to .npz file. already calculates interatomic distances.')
	parser.add_argument(
		'data_dir', type=str,
		help='input dataset folder'
	)
	parser.add_argument(
		'name', type=str,
		help='dataset name'
	)
	parser.add_argument(
		'-d', dest='max_atoms', type=str,
		help='atoms in molecule'
	)
	args = vars(parser.parse_args())
	read(**args)
