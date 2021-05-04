"""
    Check if the data files are found
"""

import os

def Check_if_DATA_files_are_found( ):
    """
        Check if the data files are found
    """

    #global __Data_location__

    this_dir, this_filename = os.path.split(__file__)
    
    Bool_1 = True
    
    arxiv_with_file_names = this_dir + '/../DATA/List_of_DATA_files'
    
    with open( arxiv_with_file_names ) as fd:
    
        for line in fd.readlines():
    
            arxiv_name = line.strip('\n')
    
            Bool_1 = Bool_1 * os.path.isfile( __Data_location__ + '/' + arxiv_name )
    
    return Bool_1

if __name__ == '__main__' : 
    pass
