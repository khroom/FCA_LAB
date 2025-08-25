import matplotlib.pyplot as plt  # Библиотека для визуализации данных
import json


def save_plot(x, true_positive, false_positive, filename):
    # Визуализация результатов
    fig, ax = plt.subplots()
    # График доли истинно положительных результатов
    ax.plot(x, true_positive, 'g',
            # График доли ложно положительных результатов
            x, false_positive, 'r')
    # Настройка отображения графика
    plt.xlabel("Пороговое значение")
    plt.ylabel("Доля")
    plt.legend(['Истинно положительные', 'Ложно положительные'])
    # Сохранение графика в файл
    plt.savefig(filename)
    plt.clf()  # Очистка текущей фигуры

def save_plot_data(filename, x_list, y_list, row_title, color):

    try:
        with open(filename, 'r') as file:
            traces = json.load(file)
    except Exception:
        traces = {}

    traces[row_title] = {}
    traces[row_title]['x'] = x_list
    traces[row_title]['y'] = y_list
    traces[row_title]['color'] = color

    try:
        with open(filename, 'w') as file:    
            json.dump(traces, file, indent=4)
    except Exception:
        print ('File dump error')