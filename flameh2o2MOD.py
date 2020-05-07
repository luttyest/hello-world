"""
A freely-propagating, premixed hydrogen flat flame with multicomponent
transport properties.
h2 to o2 0.5 to 2

task took 18084 seconds
"""

# presurre from 1 to 50    25 times
# alist 1 to 10            10 times
# temperature 300 to 1800 every 50    30 times

import sys
import numpy as np
import matplotlib.pyplot as plt
import cantera as ct
from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired
import csv
import time
import pandas as pd
import os
# Simulation parameters
pressure = ct.one_atm  # pressure [Pa]
Tin = 300.0  # unburned gas temperature [K]
width = 0.03  # m
loglevel = 1  # amount of diagnostic output (0 to 8)

Ilist = 10   #10
aList = np.zeros(Ilist)
flamespeed = []
# IdealGasMix object used to compute mixture properties, set to the state of the
# upstream fuel-air mixture
PressureS = 25  # 25
Ipressure = np.zeros(PressureS)

tempertureS = 9  # 9
Itemperture = np.zeros(tempertureS)


def flamespeedcal(test):
    L = list(test)
    print(L)
    avalue = L[0]
    pressureindex = L[1]
    tempindex = L[2]
    gas = ct.Solution('gri30.xml')
    pressureoutput = pressureindex*ct.one_atm
    gas.TP = tempindex, pressureoutput
    # 1*CH4 + 2*O2 = 2*H2O + 1*CO2
    # 1*H2 + 0.5*O2 = 1*H2O
    gas.X = {'CH4': 1, 'H2': 1, 'O2': 1/avalue, 'N2': 4/avalue}  # premixed gas composition
    # Set up flame object
    f = ct.FreeFlame(gas, width=width)
    f.set_refine_criteria(ratio=3, slope=0.06, curve=0.12)
    #f.show_solution()

    # Solve with mixture-averaged transport model
    #f.transport_model = 'Mix'
    #f.solve(loglevel=loglevel, auto=True)

    # Solve with the energy equation enabled
    #f.save('h2_adiabatic.xml', 'mix', 'solution with mixture-averaged transport')
    #f.show_solution()
    #print('mixture-averaged flamespeed = {0:7f} m/s'.format(f.u[0]))

    # Solve with multi-component transport properties
    f.transport_model = 'Multi'
    f.solve(loglevel)  # don't use 'auto' on subsequent solves
    #f.show_solution()
    #print('multicomponent flamespeed = {0:7f} m/s'.format(f.u[0]))
    #f.save('h2_adiabatic.xml','multi', 'solution with multicomponent transport')

    # write the velocity, temperature, density, and mole fractions to a CSV file
    output = "file_"+str(avalue)+"_"+str(pressureindex) + "_"+str(tempindex)+"_"+".csv"
    f.write_csv(output, quiet=False)
    print('multicomponent flamespeed = {0:7f} m/s'.format(f.u[0]))
    outputlist = []
    #convert csv to numpy
    data = pd.read_csv(output)
    #append order u,temp,rho
    u = data['u (m/s)'][0]
    temp = data['T (K)'][0]
    rho = data['rho (kg/m3)'][0]
    outputlist.extend([u, temp, rho])
    outputlist.append(pressureoutput)
    # append elements max values
    data = data.drop(columns=['z (m)', 'u (m/s)',
                              'V (1/s)', 'T (K)', 'rho (kg/m3)'])
    for name in data.columns:
        maxvalue = data[name].max()
        outputlist.append(maxvalue)

    os.remove(output)
    return outputlist



# muti-processing
def muti():
    results  = []

    with ProcessPool(max_workers = 8) as pool:
        totallist = []
        for i in range(Ilist):
            aList[i] = 0.5+0.15*i
            for m in range(PressureS):
                Ipressure[m] = m*2 + 1
                for t in range(tempertureS):
                    Itemperture[t] = 300+50*t
                    totallist.append((aList[i], Ipressure[m], Itemperture[t]))

        future = pool.map(flamespeedcal, totallist, timeout = 10000)
        iterator = future.result()
        while True:
            try:
                result = next(iterator)
                results.append(result)
            except StopIteration:
                break
            except TimeoutError as error:
                print("function took longer than %d seconds" % error.args[1])
            except ProcessExpired as error:
                print("%s. Exit code: %d" % (error, error.exitcode))
            except Exception as error:
                print("function raised %s" % error)
                print(error.traceback)

    with open("finaloutput2data.csv", 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["u(m/s)", "T(K)", "rho(kg/m3)", "pressure", "H2", "H", "O", "O2", "OH", "H2O", "HO2", "H2O2", "C", "CH", "CH2", "CH2(S)", "CH3", "CH4", "CO", "CO2", "HCO", "CH2O", "CH2OH", "CH3O", "CH3OH", "C2H", "C2H2",
                            "C2H3", "C2H4", "C2H5", "C2H6", "HCCO", "CH2CO", "HCCOH", "N", "NH", "NH2", "NH3", "NNH", "NO", "NO2", "N2O", "HNO", "CN", "HCN", "H2CN", "HCNN", "HCNO", "HOCN", "HNCO", "NCO", "N2", "AR", "C3H7", "C3H8", "CH2CHO", "CH3CHO"])
        writer.writerows(results)

                

#plot


def main():
    tic = time.perf_counter()
    muti()
    #plot()
    toc = time.perf_counter()
    print(f"task took {toc - tic:0.4f} seconds")


def plot():
    plt.plot(aList, flamespeed, '-or')
    plt.xlabel('a = [H2]/([CH4]+[H2]')
    plt.ylabel('flame speed (ms)')
    plt.yscale('log')
    plt.legend('flame speed')
    plt.savefig('CH4-H2.png', dpi=300)


if __name__ == '__main__':
    main()


