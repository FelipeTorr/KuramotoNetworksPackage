def readline(line):
    """
        Read one line from the *config* file, if the line is not tagged as a comment

        Parameters
        ----------
            line : str
            	A line of the *config* file

        Returns
        -------
        key_name : str 
        	Keyname of the parameter. The name before the "=" symbol. It returns "#" if the line is commented.
        value : str
        	Value of the parameter as written in the *config* file. It returns "#" if the line is commented.
    """
    if line==None:
        return '#','#'
    elif line[0]=='#':
        return '#','#'
    elif line[0]=='':
        return '#','#'
    else:
        tokens=line.split('=')
        if len(tokens)==2:
            return tokens[0],tokens[1].strip()
        else:
            return '#','#'   

def loadData(file_path):
    """
    Recollect the model parameters from the *config* file specified by **file_path**.
    
    Parameters
    ----------
    filepath : str
    	The path with the directory and filename of the __config__ file.
    	
    Returns
    -------
    config : dict
    	A dictionary with the required parameters for the Kuramoto model.
    	Using the template of the __config__ file, there will not be any warning.
    	If you __config__ file lacks one o several parameters, it will be an error.
    	Future release: You can specifiy only the required parameters to change, if there is not in the __config__ file, the model is instatiated with the default parameters.    
    """
    config={}
    with open(file_path) as file:
        while True:
            line=file.readline()
            if not line:
                break
            key,data=readline(line)
            if key== 'struct_connectivity' or key=='delay_matrix':
                if key=='AAL90':
                    config[key]=None
                else:
                    config[key]=data
                continue
            if key=='experiment_name':
                config[key]=data
                continue
            if key=='nat_freqs':
                if data=='' or data==' ':
                    config[key]=None
                else:
                    config[key]=data
                continue
            if key=='ForcingNodes':
                if data=='' or data==' ':
                    config[key]=None
                else:
                    config[key]=data
                continue
            if key=='random_nat_freq': 
                data=eval(data)
                config[key]=data
                continue
            if key=='max_workers' or key =='seed' or key =='n_nodes':
                data=int(data)
                config[key]=data
                continue
            elif key=='#':
                continue
            else:
                data=float(data)
                config[key]=data
                continue
            
    return config

