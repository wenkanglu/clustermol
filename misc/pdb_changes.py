import mdtraj as md

# Basic Script used for generating and editing .pdb files
# Focus on down-scalling larger files.

menY = "MenY_0_to_1000ns_aligned.pdb"
menW = "MenW_0_to_1000ns_aligned.pdb"

# Load .pdb file
traj = md.load(menW)

# Saving Files MenY
print(traj)
print(traj.time)
print(traj[::25])
traj[::25].save("MenW_0_to_1000ns_aligned(25skip).pdb")
print(traj[::50])
traj[::50].save("MenW_0_to_1000ns_aligned(50skip).pdb")
print(traj[::100])
traj[::100].save("MenW_0_to_1000ns_aligned(100skip).pdb")
print(traj[:100:])
traj[:100:].save("MenW_0_to_1000ns_aligned(100first).pdb")
print(traj[:200:])
traj[:200:].save("MenW_0_to_1000ns_aligned(200first).pdb")
print(traj[:500:])
traj[:500:].save("MenW_0_to_1000ns_aligned(500first).pdb")
print("complete")
