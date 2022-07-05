import synapseclient
import synapseutils

# C:\Users\HiroFoerYou\.synapseCache
syn = synapseclient.Synapse()
syn.login("HiroForYou", "cripta.14")
files = synapseutils.syncFromSynapse(syn, "syn3193805", path="./data")
