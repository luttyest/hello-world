"""
data generation based on cantara code on ignation delay time under different circumstances 
used for neural network 
"""

import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import cantera as ct

import concurrent.futures
import time

from threading import Thread
# Factors:
# Temperature 1000 to 2000 every 100 K
# Pressure 1 to 50 every 1 atm
# CH4:H2 defined by a 0 to 1
# H2:O2--0.5 to 2
# O2:N2--1:4 to 1:2
# fuel air equivalence ratio from 0.5 to 2

def totaldelaydata1(ITemp, IPressure, Ifraction, IER):
    Temp = np.zeros(ITemp)
    a = np.zeros(Ifraction)
    pressure = np.zeros(IPressure)
    ER = np.zeros(IER)
    # write output CSV file for importing into Excel
    gas = ct.Solution('gri30.xml')
    #writer.writerow(gas.species_names)
    for i in range(ITemp):
        Temp[i] = 1000.0 + 50*i
        for m in range(IPressure):
            pressure[m] = 1*ct.one_atm +5.0*m*ct.one_atm
            for n in range(Ifraction):
                a[n] = 0.1 * n
                for e in range(IER):
                    gas.TP = Temp[i], pressure[m]
                    ER[e] = 0.5+0.1*e
                    fO2 = ((1-a[n])*2.0 + a[n]*0.5)*(1/ER[e])
                    CH4_P = 1-a[n]
                    H2_P = a[n]
                    O2_P = fO2
                    N2_P = 3.76*fO2
                    gas.X = {'CH4': 1-a[n], 'H2': a[n],
                                'O2': fO2, 'N2': 3.76*fO2}
                    r = ct.IdealGasConstPressureReactor(gas)
                    sim = ct.ReactorNet([r])
                    time = 0.0
                    states = ct.SolutionArray(gas, extra=['t'])

                    #print('%10s %10s %10s %14s' % ('t [s]','T [K]','P [Pa]','u [J/kg]'))
                    for number in range(10000):
                        time += 1.e-6
                        sim.advance(time)
                        states.append(r.thermo.state, t=time*1e3)
                        #print('%10.3e %10.3f %10.3f %14.6e' % (sim.time, r.T,r.thermo.P, r.thermo.u))

                    X_OH = np.max(states.X[:, gas.species_index('OH')])
                    X_HO2 = np.max(states.X[:, gas.species_index('HO2')])

                    # We will use the 'OH' species to compute the ignition delay
                    max_index = np.argmax(states.X[:, gas.species_index('OH')])
                    Tau1 = states.t[max_index]

                    # We will use the 'HO2' species to compute the ignition delay
                    #max_index = np.argmax(states.X[:, gas.species_index('HO2')])
                    #Tau2[i] = states.t[max_index]

                #    writer.writerow([pressure, Temp[i], CH4_P, H2_P,
                    #                   O2_P, N2_P, ER, X_OH[i], X_HO2[i], Tau1[i]])
                    yield pressure[m], Temp[i], CH4_P, H2_P, O2_P, N2_P, X_OH, X_HO2, Tau1
                    gas = ct.Solution('gri30.xml')
            #plt.plot(states.t, states.T)
        #plt.xlabel('Time (ms)')
        #plt.ylabel('Temperature (K)')
        #plt.draw()

# h2 to o2 ratio 0.5 to 2
def totaldelaydata2(ITemp, IPressure, Ifraction, IER):
    Temp = np.zeros(ITemp)
    a = np.zeros(Ifraction)
    pressure = np.zeros(IPressure)
    ER = np.zeros(IER)
    # write output CSV file for importing into Excel
    gas = ct.Solution('gri30.xml')
    #writer.writerow(gas.species_names)
    for i in range(ITemp):
        Temp[i] = 1000.0 + 50*i
        for m in range(IPressure):
            pressure[m] = 1*ct.one_atm + 5.0*m*ct.one_atm
            for e in range(IER):
                ER[e] = 0.5+0.2*e
                gas.TP = Temp[i], pressure[m]
                CH4_P = 0.5  # percent
                H2_P = 0.5 # percent
                O2_P = 1.25
                N2_P = 3.76*1.25
                gas.X = {'CH4': 0.5*ER[e], 'H2': 0.5*ER[e],
                         'O2': 1.25, 'N2': 3.76*1.25}
                r = ct.IdealGasConstPressureReactor(gas)
                sim = ct.ReactorNet([r])
                time = 0.0
                states = ct.SolutionArray(gas, extra=['t'])

                #print('%10s %10s %10s %14s' % ('t [s]','T [K]','P [Pa]','u [J/kg]'))
                for number in range(10000):
                    time += 1.e-6
                    sim.advance(time)
                    states.append(r.thermo.state, t=time*1e3)
                    #print('%10.3e %10.3f %10.3f %14.6e' % (sim.time, r.T,r.thermo.P, r.thermo.u))

                X_OH = np.max(states.X[:, gas.species_index('OH')])
                X_HO2 = np.max(states.X[:, gas.species_index('HO2')])

                # We will use the 'OH' species to compute the ignition delay
                max_index = np.argmax(states.X[:, gas.species_index('OH')])
                Tau1 = states.t[max_index]

                # We will use the 'HO2' species to compute the ignition delay
                #max_index = np.argmax(states.X[:, gas.species_index('HO2')])
                #Tau2[i] = states.t[max_index]

            #    writer.writerow([pressure, Temp[i], CH4_P, H2_P,
                #                   O2_P, N2_P, ER, X_OH[i], X_HO2[i], Tau1[i]])
                yield pressure[m], Temp[i], CH4_P, H2_P, O2_P, N2_P, X_OH, X_HO2, Tau1
                gas = ct.Solution('gri30.xml')
            

def totaldelaydata3(ITemp, IPressure, Ifraction, IER):
    Temp = np.zeros(ITemp)
    a = np.zeros(Ifraction)
    pressure = np.zeros(IPressure)
    ER = np.zeros(IER)
    # write output CSV file for importing into Excel
    gas = ct.Solution('gri30.xml')
    #writer.writerow(gas.species_names)
    for i in range(ITemp):
        Temp[i] = 1000.0 + 50*i
        for m in range(IPressure):
            pressure[m] = 1*ct.one_atm + 5.0*m*ct.one_atm
            for n in range(Ifraction):
                a[n] = 2.0+0.2*n
                for e in range(IER):
                    gas.TP = Temp[i], pressure[m]
                    ER[e] = 0.5+0.1*e
                    CH4_P = 1/3
                    H2_P = 2/3
                    O2_P = 1
                    N2_P = a[n]
                    gas.X = {'CH4': 1*ER[e], 'H2': 2*ER[e], 'O2': 1, 'N2': a[n]}
                    r = ct.IdealGasConstPressureReactor(gas)
                    sim = ct.ReactorNet([r])
                    time = 0.0
                    states = ct.SolutionArray(gas, extra=['t'])

                    #print('%10s %10s %10s %14s' % ('t [s]','T [K]','P [Pa]','u [J/kg]'))
                    for number in range(10000):
                        time += 1.e-6
                        sim.advance(time)
                        states.append(r.thermo.state, t=time*1e3)
                        #print('%10.3e %10.3f %10.3f %14.6e' % (sim.time, r.T,r.thermo.P, r.thermo.u))

                    X_OH = np.max(states.X[:, gas.species_index('OH')])
                    X_HO2 = np.max(states.X[:, gas.species_index('HO2')])

                    # We will use the 'OH' species to compute the ignition delay
                    max_index = np.argmax(states.X[:, gas.species_index('OH')])
                    Tau1 = states.t[max_index]

                    # We will use the 'HO2' species to compute the ignition delay
                    #max_index = np.argmax(states.X[:, gas.species_index('HO2')])
                    #Tau2[i] = states.t[max_index]

                #    writer.writerow([pressure, Temp[i], CH4_P, H2_P,
                    #                   O2_P, N2_P, ER, X_OH[i], X_HO2[i], Tau1[i]])
                    yield pressure[m], Temp[i], CH4_P, H2_P, O2_P, N2_P, X_OH, X_HO2, Tau1
                    gas = ct.Solution('gri30.xml')
                    

    
def main():
    
    csv_file = 'dataComplextestC1.csv'
    tic = time.perf_counter()   # ITemp = 100
    # IPressure = 50
    # Ifraction = 11
    # IER = 16
    with open(csv_file, 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["pressure", "temp", "CH4_P", "H2_P",
                         "O2_P", "N2_P", "oh", "ho2", "tau1"])
        writer.writerows(totaldelaydata1(20, 10, 11, 16))
        writer.writerows(totaldelaydata2(20, 10, 10, 10))
        writer.writerows(totaldelaydata2(20, 10, 10, 16))
     
     
    # ITemp = 100
    # IPressure = 50
    # Ifraction = 11
    # IER = 16
    
    toc = time.perf_counter()
    print(f"run in {toc - tic:0.4f} seconds")
    
    
    # a = np.zeros(Ifraction)
    # for i in range(ITemp):
    #     gas = ct.Solution('gri30.xml')
    #     Temp[i] = 1000.0 + 10*i
    #     for m in range(IPressure):
    #         pressure = 1 + 1.0*m
    #         gas.TP = Temp[i], ct.one_atm * pressure
    #         for n in range(Ifraction):
    #             a[n] = 0.5+0.15*n
    #             CH4_P = 1
    #             H2_P = 1
    #             O2_P = 1/a[n]
    #             N2_P = 4/a[n]
    #             gas.X  = {'CH4':1,'H2':1,'O2':1/a[i],'N2':4/a[i]}
    #             r = ct.IdealGasConstPressureReactor(gas)
    #             sim = ct.ReactorNet([r])
    #             time = 0.0
    #             states = ct.SolutionArray(gas, extra=['t'])

    #             #print('%10s %10s %10s %14s' % ('t [s]','T [K]','P [Pa]','u [J/kg]'))
    #             for number in range(10000):
    #                 time += 1.e-6
    #                 sim.advance(time)
    #                 states.append(r.thermo.state, t=time*1e3)
    #                 #print('%10.3e %10.3f %10.3f %14.6e' % (sim.time, r.T,r.thermo.P, r.thermo.u))

    #             X_OH[i] = np.max(states.X[:, gas.species_index('OH')])
    #             X_HO2[i] = np.max(states.X[:, gas.species_index('HO2')])

    #             # We will use the 'OH' species to compute the ignition delay
    #             max_index = np.argmax(states.X[:, gas.species_index('OH')])
    #             Tau1[i] = states.t[max_index]

    #             # We will use the 'HO2' species to compute the ignition delay
    #             max_index = np.argmax(states.X[:, gas.species_index('HO2')])
    #             Tau2[i] = states.t[max_index]

    #             writer.writerow([pressure, Temp[i], CH4_P, H2_P,
    #                              O2_P, N2_P, X_OH[i], X_HO2[i], Tau1[i]])
    
    # Ifraction = 10
    # a = np.zeros(Ifraction)
    # for i in range(ITemp):
    #     gas = ct.Solution('gri30.xml')
    #     Temp[i] = 1000.0 + 10*i
    #     for m in range(IPressure):
    #         pressure = 1 + 1.0*m
    #         gas.TP = Temp[i], ct.one_atm * pressure
    #         for n in range(Ifraction):
    #             a[i] = 2.0+0.2*i
    #             CH4_P = 1
    #             H2_P = 2
    #             O2_P = 1
    #             N2_P = a[n]
    #             gas.X = {'CH4': 1, 'H2': 2, 'O2': 1, 'N2': a[i]}
    #             r = ct.IdealGasConstPressureReactor(gas)
    #             sim = ct.ReactorNet([r])
    #             time = 0.0
    #             states = ct.SolutionArray(gas, extra=['t'])

    #             #print('%10s %10s %10s %14s' % ('t [s]','T [K]','P [Pa]','u [J/kg]'))
    #             for number in range(10000):
    #                 time += 1.e-6
    #                 sim.advance(time)
    #                 states.append(r.thermo.state, t=time*1e3)
    #                 #print('%10.3e %10.3f %10.3f %14.6e' % (sim.time, r.T,r.thermo.P, r.thermo.u))

    #             X_OH[i] = np.max(states.X[:, gas.species_index('OH')])
    #             X_HO2[i] = np.max(states.X[:, gas.species_index('HO2')])

    #             # We will use the 'OH' species to compute the ignition delay
    #             max_index = np.argmax(states.X[:, gas.species_index('OH')])
    #             Tau1[i] = states.t[max_index]

    #             # We will use the 'HO2' species to compute the ignition delay
    #             max_index = np.argmax(states.X[:, gas.species_index('HO2')])
    #             Tau2[i] = states.t[max_index]

    #             writer.writerow([pressure, Temp[i], CH4_P, H2_P,
    #                              O2_P, N2_P, X_OH[i], X_HO2[i], Tau1[i]])
                
#plt.plot(Temp, Tau1, '-or', Temp, Tau2, '-ob')
#plt.xlabel('Temperature (K)')
#plt.ylabel('Ignition Delay Time (ms)')
#plt.yscale('log')
#plt.plot(Temp, Tau1)
#plt.legend(('Ignition delay time by OH', 'Ignition delay time by OH'))
#plt.savefig('Fig1.png', dpi=300)
#plt.show()
if __name__ == "__main__":
    main()
