# clustermol
Repository for ClusterMol honours project

# Instructions for Installation & Use
To install, use the following command while in the root directory
> python setup.py install

To run clustermol, change to a directory which contains a folder called "data". Within "data", there should be subdirectories "data_dest", "data_src" and "configs". Place input .pdb files into "data_dest", and config files (with file extension .ini) in "configs". Most outputs will be saved under "data_src".
Clustermol can be run in three modes: conf, clus and prep. Clustering jobs are performed with clus mode and prep jobs are performed in prep mode. In conf mode, either a prep job or a clus job can be done, but parameters are listed in a config file rather than command line arguments.

To run a clustering job, use the following command
> clustermol clus -a <algorithm> -s <pdb source filename> -d <save location> -cvi <CVIs to calculate> [other arguments]

To run a prep job, use the following command
> clustermol prep -s <pdb source filename> -d <pdb save filename> -fs <start frame> <end frame> -ds <frame skip value> [other arguments]

To run from a config file, use the following command
> clustermol conf -c <config file name>

Use the following commands for help
> clustermol -h
> clustermol conf -h
> clustermol clus -h
> clustermol prep -h
