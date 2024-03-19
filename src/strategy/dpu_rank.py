def CalculateDPUrank(class_, feature_data):
    bandwidth_series = []
    capacity = 0
    for vm in class_.vms:
        if Live(vm, feature_data.duration): 
            bandwidth_series += vm.cur_model.predict(feature_data.duration)
            capacity += vm.bandwidth_capacity
    
    return [capacity - i for i in bandwidth_series]