def SelectBestClass(feature_data, classes):
    ranks = []
    for class_ in range(classes):
        headroom = CalculateHeadroom(class_, feature_data)
        classrank = CalculateClassrank(class_, feature_data)
        dpurank = CalculateDPUrank(class_, feature_data)
        rank = CalculateRank(headroom, classrank, dpurank)
        ranks.append(rank)
    
    index = ranks.argmax()
    return classes[index]
    

def CalculateHeadroom(class_, feature_data):
    bandwidth_series = []
    capacity = 0
    for vm in class_.vms:
        if Live(vm, feature_data.duration): 
            bandwidth_series += vm.cur_model.predict(feature_data.duration)
            capacity += vm.bandwidth_capacity
    
    return [capacity - i for i in bandwidth_series]

def CalculateClassrank(class_, feature_data):