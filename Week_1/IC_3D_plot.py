import os
import h5py
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('macosx')

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../Data/')
hosthalo_file = 'hosthalo_IC.hdf5'
subhalo_file = 'subhalo_IC.hdf5'

with h5py.File(data_path + hosthalo_file, 'r') as f:
    # Access the coordinates dataset
    pos = f['PartType1/Coordinates'][:] 

host_x = pos[:, 0]  # X coordinates
host_y = pos[:, 1]  # Y coordinates
host_z = pos[:, 2]  # Z coordinates

with h5py.File(data_path + subhalo_file, 'r') as f:
    # Access the coordinates dataset
    pos = f['PartType1/Coordinates'][:] 

sub_x = pos[:, 0]  
sub_y = pos[:, 1]  
sub_z = pos[:, 2]  

# Create one figure and one set of 3D axes
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the Host Halo particles (Blue)
# We use a very low alpha (transparency) so we can see through the host
ax.scatter(host_x, host_y, host_z, s=0.01, c='royalblue', alpha=0.1, label='Host Halo')

# Plot the Sub Halo particles (Red)
# We use a slightly higher alpha or different color to make it "pop"
ax.scatter(sub_x, sub_y, sub_z, s=0.05, c='crimson', alpha=0.6, label='Sub Halo')

# Add labels and a legend to tell them apart
ax.set_title('Host and Sub Halo Overlap')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend(markerscale=10) # Makes the legend icons big enough to see

plt.show()