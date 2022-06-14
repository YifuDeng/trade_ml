
import matplotlib.pyplot as plt

def turn_around(model, array, seq, num):
    position = 0.0
    profit = 0.0
    total_profit = 0.0
    total_profit_list = [total_profit]
    position_list = [position]
    for i in range(len(array)-seq):
        pred = model(array[i: seq])
        pre_up = pred[0]
        pre_down = pred[1]
        real_up = array[i + seq][2]
        real_down = array[i + seq][3]
        open_price = array[i + seq][0]
        close_price = array[i + seq][1]
        if pre_up < real_up:
            position -= num
            profit += (open_price + pre_up) * num
        if pre_down < real_down:
            position += num
            profit -= (open_price + pre_up) * num
        total_profit = position * close_price + profit
    total_profit_list.append(total_profit)
    position_list.append(position)

    plt.plot(position_list)
    plt.show()

    plt.plot(total_profit_list)
    plt.show()
