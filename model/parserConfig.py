def readline(line):
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
    config={}
    with open(file_path) as file:
        while True:
            line=file.readline()
            if not line:
                break
            key,data=readline(line)
            if key== 'struct_connectivity' or key=='delay_matrix' or key=='experiment_name':
                config[key]=data
                continue
            if key=='nat_freqs':
                if data=='' or data==' ':
                    config[key]=None
                else:
                    config[key]=data
                continue
            elif key=='random_nat_freq': 
                data=eval(data)
                config[key]=data
                continue
            elif key=='max_workers' or key =='seed' or key =='n_nodes':
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

