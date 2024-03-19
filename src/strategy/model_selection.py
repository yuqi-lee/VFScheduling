def CurrentModelSelection(models, bandwidth_series):
    alpha = 0.25 #最新窗口的成本在总成本中的代价
    w_over = 0.2 #高估成本
    w_under = 0.8 #低估成本
    
    cost = []
    for i in range(len(models)):
        diff = models[i].predict_series - bandwidth_series
        cur_overpredict_cost = sum(diff > 0)
        cur_underpredict_cost = -sum(diff < 0)
        cur_cost = w_over * cur_overpredict_cost + w_under * cur_underpredict_cost
        
        models[i].cost = alpha * cur_cost + (1 - alpha) * models[i].cost
        cost[i] = models[i].cost
        
    cur_model = models[cost.argmin()]
    return cur_model   