import torch
print(torch.cuda.is_available())  # Проверяет, есть ли доступная GPU
print(torch.cuda.device_count())  # Количество доступных GPU
print(torch.cuda.get_device_name(0))  # Имя первой (0-й) GPU

import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))  # Список доступных GPU
print(tf.test.is_gpu_available())  # Проверяет, доступна ли GPU (старый метод)
print(tf.test.gpu_device_name())  # Выводит имя устройства
