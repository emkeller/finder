#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# This file is part of the SiteFinder package for searching
# cationic sites in biomolecules
#
# Copyright (c) 2016-2017, by Mishel Buyanova <emkeller@yandex.ru> and
# Arthur Zalevsky <aozalevsky@fbb.msu.ru>
#
# AffBio is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# AffBio is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public
# License along with AffBio; if not, see
# http://www.gnu.org/licenses, or write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
#

import argparse
import os
from prody import parsePDB, buildDistMatrix, AtomGroup, writePDB
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import least_squares
import sympy.geometry as smp
import networkx as nx
from itertools import product
import cPickle as pickle

class Solution(object):
    
    def __init__(self, clique, coords, dev, scparams, peakdict):
       
        self.coords = coords
        self.dev = dev
        self.scparams = scparams
        self.peakdict = peakdict
        self.sphere = clique
        self.residues = zip(
            clique.getResnames(), 
            clique.getResindices()
        )
        self.atoms = zip(
            clique.getNames(), 
            clique.getIndices()
        )
        self.score = self.count_score()
        
    def get_atom_score(self, atomn, dist):
        mu = self.scparams[atomn]['mu']
        sigma = self.scparams[atomn]['sigma']
        ampl = self.scparams[atomn]['amplitude']
        peak = self.peakdict[atomn]
        return norm.pdf(dist, mu, sigma) * ampl / peak * 100
        
    
    def count_score(self):
        scores_by_atoms = list()
        for atom in self.sphere:
            atomn = atom.getName()            
            atom_coords = atom.getCoords()
            dist = np.linalg.norm(self.coords - atom_coords)
            if atomn != 'O':
                atomscore = self.get_atom_score(atomn, dist)
            else:
                o1 = self.get_atom_score('O_1', dist)
                o2 = self.get_atom_score('O_2', dist)
                if o1 > o2:
                    atomscore = o1
                    atomn = 'O_1'
                else:
                    atomscore = o2
                    atomn = 'O_2'
            scores_by_atoms.append(atomscore)
        return np.mean(scores_by_atoms)
    
    def __len__(self):
        return len(self.sphere)
        
    def __str__(self):
        contact_info = list()
        for i, atom in enumerate(self.sphere):
            resn, resi = self.residues[i]
            atomn, atomi = self.atoms[i]
            atom_coords = atom.getCoords()
            dist = np.linalg.norm(self.coords - atom_coords)
            
            if atomn != 'O':
                mu = self.scparams[atomn]['mu']
            else:
                pref = np.argmax([self.get_atom_score('O_1', dist), 
                                  self.get_atom_score('O_2', dist)]) + 1
                mu = self.scparams['O_%i' % pref]['mu']
                
            contact_info.append(
                '{}\t{:4}\t{:3}\t{:5}\t{:8.6f}\t{:8.6f}'.format(
                    resn, resi, atomn, atomi, dist, mu)
            )
        resstring = '\n'.join(contact_info)
        resstring += '\n' + '-' * 60
        resstring += '\n{:9s} : {:>8.6f}'.format('SCORE', self.score)
        resstring += '\n{:9s} : {:>8.6f}'.format('SOL_DEV', self.dev)
        resstring += '\n{:9s} : {:7.3f}\t{:7.3f}\t{:7.3f}\n'.format('ZN_COORDS', *self.coords) 
        return resstring 
        
        

class SiteFinder(object):

    def __init__(self, threshold=50, support='data/', withlog=False):
        self.SUPPORT = support
        with open('%smudict.dict' % self.SUPPORT, 'r') as f:
            self.distdict = pickle.load(f) 
        with open('%snormgaussparams.dict' % self.SUPPORT, 'r') as f:
            self.scparams = pickle.load(f)
        with open('%speakdict.dict' % self.SUPPORT, 'r') as f:
            self.peakdict = pickle.load(f)
        with open('%srelfreqdict.dict' % self.SUPPORT, 'r') as f:
            self.atomweights = pickle.load(f)
        self.atomweights['O'] = self.atomweights['O_1'] + self.atomweights['O_2']
        self.catoms = ['OP1', 'OP2', 'OD1', 'OE1', 'OG1', 'OH', 'O',
                       'OG', 'SG', 'OE2', 'ND1', 'NE2', 'OD2']
        self.threshold = threshold
        self.structure = None
        self.withlog = withlog

    def InsertZn(self, solutions, outpath):
        lhits = len(solutions)
        solutions_coords = [sol.coords for sol in solutions]
        nozinc = self.structure.select('protein')
        insertzn = AtomGroup('Zincs')
        insertzn.setCoords(solutions_coords)
        zns = ['ZN' for i in range(lhits)]
        insertzn.setChids(['Z' for i in range(lhits)])
        insertzn.setNames(zns)
        insertzn.setElements(zns)
        insertzn.setResnames(zns)
        insertzn.setSerials(
            [len(nozinc.getSerials()) + i + 1 for i in range(lhits)])
        insertzn.setResnums(
            [max(nozinc.getResnums()) + i + 1 for i in range(lhits)])
        writePDB(outpath, nozinc.copy()+insertzn)
            
    def WriteLog(self, solutions, inpath, outpath):
        out_text = list()
        out_text.append(
            'Found %i spheres with score > %f' % (len(solutions), self.threshold)
        )
        if len(solutions) > 0:
            out_text.append(
                '\t'.join(
                    ['RESN', 'RESI', 'NAME', 'INDEX', 'REAL DIST', 'IDEAL DIST']
                )
            )
            for i, sol in enumerate(sorted(solutions, 
                                           key=lambda s: s.score, 
                                           reverse=True)):
                out_text.append(str(sol))
                
        out_str_text = '\n\n'.join(out_text)
        print out_str_text
        
        if self.withlog:
            prefix = os.path.basename(inpath)[:-4]
            logname = 'log_%s' % prefix
            logpath = os.path.join(os.path.dirname(outpath), logname)
            with open(logpath, 'w') as log:
                log.write(out_str_text)
            print 'Wrote log file to %s' % (logpath)
        
        
    
    @staticmethod
    def Distance(a, b):
        dist = np.linalg.norm(a - b)
        return dist
    
    def Reflect(self, x0, coordinators):
        p0 = smp.Point(x0)
        points = [smp.Point3D(ci) for ci in coordinators]
        plane = smp.Plane(points[0], points[1], points[2])
        perp = plane.perpendicular_line(p0)
        i = perp.intersection(plane)[0]
        direct = p0.direction_ratio(i)
        x = i + direct
        x1 = [float(coord) for coord in x]
        return x1

    def Trilaterate(self, coordinators, r): 
        n = len(coordinators)
        c0 = coordinators[0]
        d = [ self.Distance(ci, c0) 
              for ci in coordinators ]
        A = [ ci - c0 
              for ci in coordinators[1:] ]
        b = [ 0.5 * (r[0] ** 2 - r[i] ** 2 + d[i] ** 2) 
              for i in range(1, n)]
        A = np.array(A)
        b = np.array(b)
        if n == 4:
            x = np.linalg.solve(A, b)
        elif n == 3: 
            x = np.linalg.lstsq(A, b, rcond=None)[0]
        return x + c0

    def ResidualVector(self, x, c_coordinates, c_distances):
        resvect = [self.Distance(x, ci) - ri 
                   for ci in c_coordinates 
                   for ri in c_distances]
        return np.array(resvect)
       
    def GetCliques(self, inpath):
        self.structure = parsePDB(inpath).protein

        verts = self.structure.select('name {}'.format(' '.join(self.catoms)))
        vindices = verts.getIndices()
        verts_iter = verts.copy()
        
        disttable = pd.DataFrame(data=buildDistMatrix(verts), 
                                 index=vindices,
                                 columns=vindices)
        disttable = disttable[
            disttable <= 6][
            disttable >= 1].fillna(0)
                
        G = nx.Graph()
        
        with open('%spairwise_boundaries.dict' % self.SUPPORT, 'r') as f:
            pairwise_boundaries = pickle.load(f)
        
        for i in range(len(verts)):
            x = verts_iter[i]
            xn = x.getName()
            xi = vindices[i]
            for j in range(i):
                y = verts_iter[j]
                yn = y.getName()
                yi = vindices[j]
                pair = tuple(sorted((xn, yn)))
                interval = pairwise_boundaries[pair]
                if interval[0] <= disttable[xi][yi] <= interval[1]:
                    G.add_edges_from([(xi, yi)])
        
        cliques = list()
        iter_cliques = nx.find_cliques(G)
    
        for c in iter_cliques:
            lc_ = len(c)
            clique_sele = verts.select(
                    'index {}'.format(' '.join([str(x) for x in c]))
                )
            if lc_  < 3:
                pass
            elif lc_ == 3:
                residues = clique_sele.getResindices()
                atoms = clique_sele.getNames()
                if len(set(residues)) == 3:
                    cliques.append(sorted(c))
            elif lc_ == 4:
                cliques.append(sorted(c))
            elif lc_ > 4:
                cliques.append(sorted(
                    c, 
                    key=lambda x: self.atomweights[
                        clique_sele.select('index %i' % x).getNames()[0]
                    ], 
                    reverse=True
                )[:4])
        if len(cliques) == 0:
            raise ValueError('No cliques found in your structure.')
        cliques_dict = dict()

        for c in cliques:
            clique_sele = verts.select(
                    'index {}'.format(' '.join([str(x) for x in c]))
                )
            wc_ = np.mean([self.atomweights[atom] 
                           for atom in clique_sele.getNames()])
            cliques_dict[clique_sele] = wc_

        return cliques_dict

       
    def MainJob(self, inpath, outpath):
        
        cliques_dict = self.GetCliques(inpath)
        print 'Found %i cliques' % (len(cliques_dict))
        solutions = list()
        
        processed = 0
        for clique in cliques_dict.keys():
            processed += 1
            lc_ = len(clique)
            c_coordinates = [atom.getCoords() 
                             for atom in clique]
            c_ideal_dists_vars = list(
                product(
                    *[self.distdict[atom.getName()] 
                      for atom in clique]
                       )
            )
            j = 0
            dev = 100
            while dev > 1.5 and j < len(c_ideal_dists_vars):
                c_ideal_dists = c_ideal_dists_vars[j]
                x0 = self.Trilaterate(c_coordinates, c_ideal_dists)
                for i in range(5 - lc_): 
                    if i == 1 and around: 
                        x0 = self.Reflect(x0, c_coordinates)
                    lsmodel = least_squares(
                        self.ResidualVector,
                        x0,
                        method='lm',
                        args=(c_coordinates, c_ideal_dists),
                        )
                    coords = lsmodel.x
                    dev = np.linalg.norm(lsmodel.fun)
                    around = self.structure.select('within 1.39 of coords',
                                                   coords=coords)
                    if not around and dev < 1.5:
                        sol = Solution(
                            clique, 
                            coords,
                            dev,
                            scparams=self.scparams,
                            peakdict=self.peakdict
                        )
                        if sol.score > self.threshold:
                            solutions.append(sol)
                    j += 1
            if processed % 10 == 0:
                print 'Processed %i cliques of %i...' % (processed, len(cliques_dict))
                            
        rounded_coords = [(map(lambda x: round(x, 2), sol.coords)) 
                          for sol in solutions]
        _, unique_idxs = np.unique(rounded_coords,
                                   axis=0,
                                   return_index=True)       
        solutions = [solutions[i] for i in unique_idxs]
        
        if len(solutions) > 0:
            self.InsertZn(solutions, outpath)
        self.WriteLog(solutions, inpath, outpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i', '--input',
        help='input PDB file',
        type=str,
        required=True
        )

    parser.add_argument(
        '-o', '--output',
        help='output PDB filename',
        type=str,
        required=True)
    
    parser.add_argument(
        '-t', '--threshold',
        help='set score threshold',
        type=int,
        default=50
    )

    parser.add_argument('-l', '--log', help='write log', action='store_true')
    
    options = parser.parse_args()
    withlog = options.log
    threshold = options.threshold
    inpath = os.path.abspath(options.input)
    outpath = os.path.abspath(options.output)
    

    sf = SiteFinder(threshold=threshold, withlog=withlog)
    sf.MainJob(inpath, outpath)

