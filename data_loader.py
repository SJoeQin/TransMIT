'''Data loader for flight test process, industrial 660MW boiler process datasets.
'''

# Necessary packages
import numpy as np

def data_loader(data_name):
  '''Loads datasets and introduce missingness.
  
  Args:
    - data_name: FDT or Boiler
    
  Returns:
    data_x: original data
  '''
  
  # Load data
  file_name = 'data/'+data_name+'.csv'
  data_x = np.loadtxt(file_name, delimiter=",", encoding='utf-8-sig')     # no header

  return data_x

  














