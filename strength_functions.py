# Script by J{\o}rgen Eriksson Midtb{\o}, University of Oslo
# j.e.midtbo@fys.uio.no
# Written September 2017.
from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt 

def div0( a, b ):
  """ division function designed to ignore / 0, i.e. div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
  with np.errstate(divide='ignore', invalid='ignore'):
    c = np.true_divide( a, b )
    c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
  return c

def read_energy_levels(inputfile):
  # Reads levels from a KSHELL summary file, returns Nx3 matrix of 
  # [Ei, 2*Ji, parity], where E is absolute energy of level and parity 1=+, 0=-
  levels = []
  with open(inputfile, 'r') as f:
    lines = f.readlines()
    i_start = -1
    for i in range(len(lines)):
      if len(lines[i].split())<1: continue
      if lines[i].split()[0] == "Energy":
        i_start = i+4
        break
    for i in range(i_start, len(lines)):
      if len(lines[i].split())<1: break
      words = lines[i].split()
      num_parity = 1 if words[2] == "+" else -1
      levels.append([float(words[5]), float(words[1]), num_parity])
    return np.asarray(levels) 

def read_transition_strengths(inputfile, type="M1"):
  transitions = []
  with open(inputfile, 'r') as f:
    lines = f.readlines()
    i_start = -1
    for i in range(len(lines)):
      if len(lines[i].split())<1: continue
      if lines[i].split()[0] == "B({:s})".format(type):
        # print "hello"
        i_start = i+2
        break
    for i in range(i_start, len(lines)):
      # print lines[i]
      if len(lines[i].split())<1: break
      line = lines[i]
      # Returns
      # [2Ji, pi, Ei, 2Jf, pf, Ef, Eg, B(M1,i->f)]  --  beware that the summary file has opposite initial/final convention to this!
      pi = +1 if line[26] == "+" else -1
      pf = +1 if line[4] == "+" else -1
      transitions.append([float(line[22:25]), pi, float(line[34:43]), float(line[0:3]), pi, float(line[12:22]), float(line[43:51]), float(line[67:83])])
    return np.asarray(transitions)




# ===== Begin main program =====

# Initialize figure objects
f_rho, ax_rho =   plt.subplots(1,1)
f_gsf, ax_gsf =   plt.subplots(1,1)
plt.style.use('seaborn-dark-palette')
lw_list = [1.8, 2.3, 2.2, 2.1, 2.0]
linestyles = ['-','-','-','-','-']
dashes_list = [(),(3,1),(4,2),(6,3),(8,4)]


# Set bin width and range
bin_width = 0.20
# Maximum energy of levels that we consider:
Emax = 12
# Define the excitation energy region that we consider for the extraction of the gsf:
Ex_min = 4
Ex_max = 6.5
Nbins = int(np.ceil(Emax/bin_width))
Emax_adjusted = bin_width*Nbins
bins = np.linspace(0,Emax_adjusted,Nbins+1)
bins_middle = (bins[0:-1]+bins[1:])/2

# Find index of first and last bin (lower bin edge) where we put counts.
# It's important to not include the other Ex bins in the averaging later, because they contain zeros which will pull the average down.
i_Exmin = int(np.floor(Ex_min/bin_width)) 
i_Exmax = int(np.floor(Ex_max/bin_width))  


# Set name to be used for saving figures
save_name = "Ni70_shell_model_strength_functions"

# Define list of calculation input files and corresponding label names
inputfile = "summary_Ni70_ca48mh1g.txt"

name = "$\mathrm{ca48mh1g}$"

# Set a spin window by defining list of allowed initial [spins, parities].
Jpi_list = [
              # All spins:
              [0,+1],[2,+1],[4,+1],[6,+1],[8,+1],[10,+1],[12,+1],[14,+1],[16,+1],[18,+1],[20,+1],[22,+1],[24,+1],[26,+1],[28,+1],
              [0,-1],[2,-1],[4,-1],[6,-1],[8,-1],[10,-1],[12,-1],[14,-1],[16,-1],[18,-1],[20,-1],[22,-1],[24,-1],[26,-1],[28,-1],
              # ]
              # # Densest spins:
              # [6,+1],[8,+1],[10,+1],[12,+1],[14,+1],[16,+1],[18,+1],[20,+1],[22,+1],
              # [6,-1],[8,-1],[10,-1],[12,-1],[14,-1],[16,-1],[18,-1],[20,-1],[22,-1],
           ]

levels = read_energy_levels(inputfile)
Egs = levels[0,0] # Read out the absolute ground state energy, so we can get relative energies later
# Read transition strengths
transitions = read_transition_strengths(inputfile, type="M1")



# ==== Calculate rho(Ex,J,pi) ====

# Allocate (Ex,Jpi) matrix to store level density
rho_ExJpi = np.zeros((Nbins,len(Jpi_list)))
# Count number of levels for each (Ex, J, pi) pixel.
for i_l in range(len(levels[:,0])):
  E, J, pi = levels[i_l]
  # Skip if level is outside range:
  if E-Egs >= Emax:
    continue
  i_Ex = int(np.floor((E-Egs)/bin_width))
  try:
    i_Jpi = Jpi_list.index([J,pi])
  except:
    continue
  rho_ExJpi[i_Ex,i_Jpi] += 1
rho_ExJpi /= bin_width # Normalize to bin width, to get density in MeV^-1




# ==== Allocate matrices to store the summed B(M1) values for each (Ex,J,pi)-pixel, and the number of transitions counted ====

B_pixel_sum = np.zeros((Nbins,Nbins,len(Jpi_list)))
B_pixel_count = np.zeros((Nbins,Nbins,len(Jpi_list)))
# Loop over all transitions and put in the correct pixel:
for i_tr in range(len(transitions[:,0])):
  Ex = transitions[i_tr,2] - Egs
  # Check if transition is below Ex_max, skip if not
  if Ex < Ex_min or Ex >= Ex_max:
    continue

  # Get bin index for Eg and Ex (initial). Indices are defined with respect to the lower bin edge.
  i_Eg = int(np.floor(transitions[i_tr,6]/bin_width))
  i_Ex = int(np.floor(Ex/bin_width))

  # Read initial spin and parity of level:
  Ji = int(transitions[i_tr,0])
  pi = int(transitions[i_tr,1])
  # Get index for current [Ji,pi] combination in Jpi_list:
  try:
    i_Jpi = Jpi_list.index([Ji,pi])
  except: 
    continue

  # Add B(M1) value and increment count to pixel, respectively
  # HACK 20170922: Testing a B(M1) threshold, has little effect.
  # if transitions[i_tr,7] < 1e-1:
  #   continue
  B_pixel_sum[i_Ex,i_Eg,i_Jpi] += transitions[i_tr,7]
  B_pixel_count[i_Ex,i_Eg,i_Jpi] += 1




# ==== Calculate gsf by the different methods: ====

# === 1) Calculate the strength function individually for each (Ex,J,pi)-pixel, then average over everything ===
gsf_ExJpi = np.zeros((Nbins,Nbins,len(Jpi_list)))
a = 11.5473e-9 # mu_N^-2 MeV^-2, conversion constant
for i_Jpi in range(len(Jpi_list)):
  for i_Ex in range(Nbins):
                      # a *           <B(M1; Eg, Ex, J, pi)>                             * rho(Ex, J, pi)  
    gsf_ExJpi[i_Ex,:,i_Jpi] = a * div0(B_pixel_sum[i_Ex,:,i_Jpi], B_pixel_count[i_Ex,:,i_Jpi]) * rho_ExJpi[i_Ex, i_Jpi]

gsf_tmp = gsf_ExJpi[i_Exmin:i_Exmax+1,:,:]
gsf_ExJpiavg = div0(gsf_tmp.sum(axis=(0,2)), (gsf_tmp!=0).sum(axis=(0,2)))

linestyle = linestyles[0]
dashes = dashes_list[0]
lw = lw_list[0]
ax_gsf.plot(bins_middle[0:len(gsf_ExJpiavg)], gsf_ExJpiavg, label=r"Avg individual $(E_x,J,\pi)$", linewidth=lw, linestyle=linestyle, dashes=dashes)

# === 2) Average all B(M1) within (Ex, Eg) pixel, multiply by total rho, then average over Ex
gsf_Ex = np.zeros((Nbins,Nbins))
# Sum B(M1) counts over all (J,pi) values, and the same for number of counts:
B_pixel_sum_EgEx = B_pixel_sum.sum(axis=2)
B_pixel_count_EgEx = B_pixel_count.sum(axis=2)
# Sum rho(Ex,J,pi) over (J,pi) to get total level density at Ex.
rho_total = rho_ExJpi.sum(axis=1)
for i_Ex in range(Nbins):
                  # a *           <B(M1; Eg, Ex)>                                 * rho(Ex)  
  gsf_Ex[i_Ex,:] = a * div0(B_pixel_sum_EgEx[i_Ex,:], B_pixel_count_EgEx[i_Ex,:]) * rho_total[i_Ex]

gsf_Exavg =  div0(gsf_Ex[i_Exmin:i_Exmax+1,:].sum(axis=0), (gsf_Ex[i_Exmin:i_Exmax+1,:]!=0).sum(axis=0))

linestyle = linestyles[1]
dashes = dashes_list[1]
lw = lw_list[1]
ax_gsf.plot(bins_middle[0:len(gsf_Exavg)], gsf_Exavg, label=r"Total $\rho$", linewidth=lw, linestyle=linestyle, dashes=dashes)



# === 3) Split rho into parities, average B(M1) within (Ex, Eg)-
# pixel for each parity, then sum the result [or average?]
gsf_pos_Ex = np.zeros((Nbins, Nbins))
gsf_neg_Ex = np.zeros((Nbins, Nbins))
B_pixel_sum_pos_EgEx = np.zeros((Nbins, Nbins))
B_pixel_sum_neg_EgEx = np.zeros((Nbins, Nbins))
B_pixel_count_pos_EgEx = np.zeros((Nbins, Nbins))
B_pixel_count_neg_EgEx = np.zeros((Nbins, Nbins))
rho_pos = np.zeros(Nbins)
rho_neg = np.zeros(Nbins)
# Loop over [J,pi] list and sort the parities:
for i_Jpi in range((len(Jpi_list))):
  J, pi = Jpi_list[i_Jpi]
  # print J, pi
  if pi == +1:
    B_pixel_sum_pos_EgEx += B_pixel_sum[:,:,i_Jpi]
    B_pixel_count_pos_EgEx += B_pixel_count[:,:,i_Jpi]
    rho_pos += rho_ExJpi[:,i_Jpi]
  elif pi == -1:
    B_pixel_sum_neg_EgEx += B_pixel_sum[:,:,i_Jpi]
    B_pixel_count_neg_EgEx += B_pixel_count[:,:,i_Jpi]
    rho_neg += rho_ExJpi[:,i_Jpi]
  else:
    raise Exception("Something is wrong in the parity check")

# Calculate gsf:
for i_Ex in range(Nbins):
                     # a *           <B(M1; Eg, Ex, pi=+)>                                    * rho(Ex, pi=+)  
  gsf_pos_Ex[i_Ex,:] = a * div0(B_pixel_sum_pos_EgEx[i_Ex,:], B_pixel_count_pos_EgEx[i_Ex,:]) * rho_pos[i_Ex]
                     # a *           <B(M1; Eg, Ex, pi=-)>                                    * rho(Ex, pi=-)  
  gsf_neg_Ex[i_Ex,:] = a * div0(B_pixel_sum_neg_EgEx[i_Ex,:], B_pixel_count_neg_EgEx[i_Ex,:]) * rho_neg[i_Ex]


gsf_pos_Exavg =  div0(gsf_pos_Ex[i_Exmin:i_Exmax+1,:].sum(axis=0), (gsf_pos_Ex[i_Exmin:i_Exmax+1,:]!=0).sum(axis=0))
gsf_neg_Exavg =  div0(gsf_neg_Ex[i_Exmin:i_Exmax+1,:].sum(axis=0), (gsf_neg_Ex[i_Exmin:i_Exmax+1,:]!=0).sum(axis=0))

linestyle = linestyles[2]
dashes = dashes_list[2]
lw = lw_list[2]
ax_gsf.plot(bins_middle[0:len(gsf_pos_Exavg)], gsf_pos_Exavg, label="$\pi =+$", linewidth=lw, linestyle=linestyle, dashes=dashes)
linestyle = linestyles[3]
dashes = dashes_list[3]
lw = lw_list[3]
ax_gsf.plot(bins_middle[0:len(gsf_neg_Exavg)], gsf_neg_Exavg, label="$\pi =-$", linewidth=lw, linestyle=linestyle, dashes=dashes)
linestyle = linestyles[3]
dashes = dashes_list[3]
lw = lw_list[3]
ax_gsf.plot(bins_middle[0:len(gsf_neg_Exavg)], gsf_neg_Exavg+gsf_pos_Exavg, label="$\sum \pi$", linewidth=lw, linestyle=linestyle, dashes=dashes)




# General plot settings

# -- rho
ax_rho.step(bins, np.append(0,rho_total), where='pre', label=r"Total $\rho$")
ax_rho.step(bins, np.append(0,rho_pos), where='pre', label=r"$\rho_+$")
ax_rho.step(bins, np.append(0,rho_neg), where='pre', label=r"$\rho_-$")
ax_rho.set_yscale('log')
ax_rho.set_xlabel(r'$E_x \, \mathrm{(MeV)}$', fontsize=22)
ax_rho.set_ylabel(r'$\rho \, \mathrm{(MeV^{-1})}$', fontsize=22)
ax_rho.set_xlim([-0.5,10])
ax_rho.set_ylim([1e-1, 1e5])
ax_rho.legend(loc='best', fontsize=18)
ax_rho.tick_params(labelsize=16)
ax_rho.grid(True)
f_rho.set_size_inches(8,6)
f_rho.subplots_adjust(left=0.15, right=0.95, top=0.98, bottom=0.15)
f_rho.savefig(save_name+"_rho.pdf")

# -- gsf
ax_gsf.set_yscale('log')
ax_gsf.set_ylabel(r'$f_{XL}\, \mathrm{(MeV^{-3})}$', fontsize=22)
ax_gsf.set_xlabel(r'$E_\gamma \,\mathrm{(MeV)}$', fontsize=22)
ax_gsf.set_ylim([1e-11,5e-7])
ax_gsf.set_xlim([0,10])
ax_gsf.set_xticks([0,2,4,6,8,10])
ax_gsf.set_yticks([1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6])
ax_gsf.tick_params(labelsize=16)
ax_gsf.tick_params(labelsize=16)
ax_gsf.grid(True)
ax_gsf.legend(loc='best', fontsize=15)
f_gsf.subplots_adjust(left=0.15, right=0.95, top=0.98, bottom=0.15)
f_gsf.subplots_adjust(hspace=0)
f_gsf.set_size_inches(8,6)
f_gsf.savefig(save_name+"_gsf.pdf")



plt.show()
