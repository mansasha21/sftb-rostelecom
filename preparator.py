from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import BayesianGaussianMixture
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from category_encoders import CatBoostEncoder
from category_encoders import TargetEncoder
from category_encoders import OrdinalEncoder


def process_period(try_set):
    """
    Обрабатывает периоды в датафрейме
    :param try_set: датафрейм с периодами
    :return: датафрейм с обработанными периодами
    """
    try_set['period_mod'] = pd.to_datetime(try_set['period'])
    try_set['year'] = try_set['period_mod'].dt.year
    try_set['month'] = try_set['period_mod'].dt.month
    try_set['day'] = try_set['period_mod'].dt.day
    try_set['cos_month'] = np.cos(try_set['period_mod'].dt.month)
    drop_cols = ['period_mod', 'day']
    try_set.drop(drop_cols, axis=1, inplace=True)
    return try_set


def add_statistical_feature(df):
    """
    Add statistical feature to the dataframe
    :param df: dataframe
    :return: dataframe with statistical feature
    """
    district = ['Центральный', 'Северо-Западный', 'Южный', 'Северо-Кавказский',
                'Приволжский', 'Уральский', 'Сибирский', 'Дальневосточный']
    mobile_customer_district = [110.7, 110.8, 90, 70.8, 96.1, 100.9, 96.6, 98.7]
    customer_district = [27.9, 24.8, 19.8, 10.2, 22.7, 25.6, 21.4, 19]
    communication_district = [3.3, 3.3, 3.6, 3.8, 3.3, 3, 3.4, 3.7]
    population_district = [39251, 13942, 16482.5, 9967.3, 29070.8, 12329.5, 17003.9, 8124]
    mean_income_district = [46880, 41032.9, 32759.4, 23485.7, 28382, 42555.9, 28406.7, 39822.4]
    children_district = [16.6, 17.3, 18.2, 23.9, 18.7, 20.6, 20.2, 20.8, ]
    services_district = [28.7, 25.5, 24.2, 19.5, 24.3, 26, 23.3, 25.5]
    rural_district = [17.6, 15, 37, 49.6, 27.7, 18.3, 25.7, 27]
    mobile_th_district = [44834.4, 16064.6, 15285.7, 7215.5, 28593.3, 12733.5, 16720.7, 8184]
    pc_district = [76.2, 77.7, 72.4, 63.5, 69.4, 72.7, 67.3, 67.3]
    inter_district = [81.7, 81.3, 80, 82.4, 77.7, 80.8, 77.4, 81.3]
    good_inter_district = [79.6, 79.3, 76.6, 75.8, 75.1, 78.3, 72.8, 75.7]
    amount_inf_district = [18037.9, 9465.4, 4072.9, 1518.3, 12219.2, 5166.4, 5000.1, 3681.]

    subject = ["Белгородская", "Брянская", "Владимирская", "Воронежская", "Ивановская",
               "Калужская", "Костромская", "Курская", "Липецкая", "Московская",
               "Орловская", "Рязанская", "Смоленская", "Тамбовская", "Тверская",
               "Тульская", "Ярославская", "Москва", 'Карелия', 'Коми', 'Архангельская',
               'Ненецкий', 'Вологодская', 'Калининградская', 'Ленинградская',
               'Мурманская', 'Новгородская', 'Псковская', 'Санкт-Петербург',
               'Адыгея', 'Калмыкия', 'Крым', 'Краснодарский', 'Астраханская',
               'Волгоградская', 'Ростовская', 'Севастополь', 'Дагестан', 'Ингушетия',
               'Кабардино-Балкарская', 'Карачаево-Черкесская', 'Северная Осетия - Алания',
               'Чеченская', 'Ставропольский', 'Башкортостан', 'Марий Эл', 'Мордовия',
               'Татарстан', 'Удмуртская', 'Чувашская', 'Пермский', 'Кировская',
               'Нижегородская', 'Оренбургская', 'Пензенская', 'Самарская', 'Саратовская',
               'Ульяновская', 'Курганская', 'Свердловская', 'Тюменская',
               'Ямало-Ненецкий',
               'Ханты-Мансийский Автономный Округ - Югра', 'Челябинская',
               'Алтай', 'Тыва', 'Хакасия', 'Алтайский', 'Красноярский', 'Иркутская',
               'Кемеровская Область-Кузбасс', 'Новосибирская', 'Омская', 'Томская', 'Бурятия',
               'Саха /Якутия/', 'Забайкальский', 'Камчатский', 'Приморский',
               'Хабаровский', 'Амурская', 'Магаданская', 'Сахалинская', 'Еврейская',
               'Чукотский']
    mobile_customer_subject = [91.8, 83.8, 87.9, 89.5, 97.6, 101.7, 88.4, 92.6, 91.5, 100, 91.4, 92.3, 96.7, 81.8, 94.4,
                               101.4, 107.7, 127, 87.8, 95.6, 92.3, 88.7, 88.1, 103.1, 115, 97.6, 90, 85.7, 127.8, 54.8,
                               81.4, 42.9, 119.3, 91.4, 85.9, 88.8, 2.1, 58.4, 51.8, 76.4, 68.4, 84,
                               62.1, 88.4, 91.9, 91.4, 75.8, 108.4, 89.2, 91.2, 94.5, 86.2, 117.9, 94.8, 87.9, 94.6,
                               88.4,
                               86.7, 101.5, 96.1, 105.7, 130.2, 112.2, 94.5, 90.3, 62, 109.4, 89.7, 95.2, 105.9, 91.9,
                               103.1, 96.1, 95.9, 86.6, 95.8, 82.5, 105.7, 107.2, 104.9, 100, 102, 110.8, 83.7, 102.3]
    good_inter_subject = [70.6, 73.9, 75.5, 79.6, 67.4, 74.5, 63.7, 74.0, 68.8, 86.0,
                          70.7, 72.7, 69.9, 72.8, 59.2, 89.1, 72.5, 87.5, 76.5, 78.2, 75.4, 72.5,
                          70.2, 72.4, 77.2, 84.2, 66, 72.5, 87, 78.4, 82.6, 81.6, 71.4, 87.3,
                          74.7, 78.1, 82.7, 74.2, 76.3, 86.6, 68.7, 83.2, 74.6, 74.1, 70.5, 66.6,
                          63, 82.9, 73.7, 68.2, 71.9, 75.3, 75.1, 82.5, 78.2, 81.1, 72.5, 68.6,
                          76.8, 73.1, 78.4, 91.9, 90.9, 78.5, 84.2, 91.8, 67.2, 68.5, 70.1, 71.2,
                          73.5, 77.5, 74, 74.2, 77.6, 81.5, 61.7, 69.6, 79.3, 79.7, 72.4, 86.5,
                          77.4, 65.8, 46.3]
    customer_subject = [20.6, 22.9, 24.1, 28.5, 20.2, 27.4, 26, 24.2, 25.9, 22.5, 27.5, 24.9, 23.5, 20.4, 16.7, 25.6,
                        24.9, 36.8, 33.7, 22.8, 23.7, 21.7, 23.6, 21.7, 12.3, 30.8, 18.8, 21.1, 29.8, 9.2, 14.2, 14.5,
                        20.5, 18.2, 18.8, 23.8, 17.9, 4.2, 1.9, 11.1, 11.2, 17.9, 6, 18.4, 22, 18.3, 19.2, 28, 21.1,
                        22.3, 22, 21.3, 24, 19.2, 19.9, 22.9, 22.5, 21.7, 21.3, 26.6, 27.4, 24, 24.2, 25.5, 8.6, 6.9,
                        13.6, 19.1, 16.8, 20.2, 18.3, 39.2, 17, 21.1, 15.5, 18.6, 16.4, 16.9, 18.3, 26.5, 18.9, 20.7,
                        18.4, 16.4, 12]
    communication_subject = [3.2, 3.7, 4.2, 3.6, 3.1, 3.8, 3.3, 3.4, 3, 2.9, 3.7, 3.7, 3.4,
                             3.8, 3.8, 3.2, 3.2, 3.3, 3.8, 3.1, 2.9, 4.2, 3, 3.5,
                             3.8, 3.6, 3.4, 3.5, 3.1, 3.8, 4.3, 3.3, 4.2, 3.5, 3.5,
                             3.2, 3.6, 3.3, 2.6, 5.4, 4.5, 4.5, 3.5, 4, 3.2, 3.8,
                             4.2, 3.2, 3.3, 4, 3.1, 3.7, 2.9, 3.7, 3.8, 3.2, 3.8, 3.8, 3.2,
                             2.9, 3.9, 4.2, 2.9, 2.8, 2.4, 4.9, 3.5, 4.1, 3.7, 3.4,
                             3.2, 3.3, 2.8, 3.2, 3.5, 5, 3.6, 3.1, 3.5, 3, 4.2, 4.1,
                             3.3, 4.1, 4.6]
    services_subject = [20.4, 19.7, 24.3, 25.6, 22.7, 22.4, 20.8, 20.3, 23.2, 26.3, 20, 19.5,
                        21.6, 22.9, 23.3, 23.3, 30.7, 34.1, 20.6, 27.1, 21.4, 28.1, 22.1,
                        19.3, 23.2, 28.7, 26.2, 23, 27.8, 21.9, 17.7, 21.7, 23.9, 22.8,
                        22.2, 28, 19.3, 13.5, 11.5, 19.9, 24.8, 20.4, 16.7, 26.5, 24, 24.7,
                        21.2, 23.7, 25, 25.2, 25.2, 22.8, 27.5, 22.1, 25.2, 23.2, 23.8, 25.7,
                        19.1, 28.1, 22.1, 28.1, 25.7, 25.4, 22.5, 22.3, 19.8, 22.8, 27.7,
                        21.4, 23.9, 18.7, 24.2, 24.2, 20.6, 25.4, 21.8, 24.8, 27.9, 27.6,
                        26.7, 30.6, 21.8, 21.8, 26.8]
    districts = ['Центральный', 'Центральный', 'Центральный', 'Центральный', 'Центральный',
                 'Центральный', 'Центральный', 'Центральный', 'Центральный', 'Центральный',
                 'Центральный', 'Центральный', 'Центральный', 'Центральный', 'Центральный',
                 'Центральный', 'Центральный', 'Центральный', 'Северо-Западный', 'Северо-Западный',
                 'Северо-Западный', 'Северо-Западный', 'Северо-Западный', 'Северо-Западный',
                 'Северо-Западный', 'Северо-Западный', 'Северо-Западный', 'Северо-Западный',
                 'Северо-Западный', 'Южный', 'Южный', 'Южный', 'Южный', 'Южный', 'Южный', 'Южный',
                 'Южный', 'Северо-Кавказский', 'Северо-Кавказский', 'Северо-Кавказский',
                 'Северо-Кавказский', 'Северо-Кавказский', 'Северо-Кавказский', 'Северо-Кавказский',
                 'Приволжский', 'Приволжский', 'Приволжский', 'Приволжский', 'Приволжский',
                 'Приволжский', 'Приволжский', 'Приволжский', 'Приволжский', 'Приволжский',
                 'Приволжский', 'Приволжский', 'Приволжский', 'Приволжский', 'Уральский', 'Уральский',
                 'Уральский', 'Уральский', 'Уральский', 'Уральский', 'Сибирский', 'Сибирский',
                 'Сибирский', 'Сибирский', 'Сибирский', 'Сибирский', 'Сибирский', 'Сибирский', 'Сибирский', 'Сибирский',
                 'Дальневосточный', 'Дальневосточный', 'Дальневосточный', 'Дальневосточный', 'Дальневосточный',
                 'Дальневосточный', 'Дальневосточный', 'Дальневосточный', 'Дальневосточный', 'Дальневосточный',
                 'Дальневосточный']
    amount_inf_subject = [501.1, 433.3, 512.8, 888.2, 245.9, 362.9, 190.1, 316.8, 426.1, 2556.5,
                          225.6, 470.0, 512.9, 397.1, 291.0, 498.9, 469.2, 8739.4, 382.6,
                          343.6, 396.4, 21.0, 368.9, 283.6, 373.4, 495, 168.2, 105.3,
                          6527.5, 97.8, 53.2, 237.2, 1367.4, 265.7, 566.1, 1440.2, 45.1,
                          257.4, 16.6, 144.9, 74.3, 211.8, 193.2, 620.1, 821.8, 212.2, 197.2,
                          1540.8, 492.9, 349.7, 1604, 390.1, 1214, 434.9, 406.5, 3698.5,
                          558.1, 298.2, 240.5, 1951.5, 661.8, 277.1, 568.1, 1467.3, 38.0,
                          48.6, 100.3, 537.5, 597.6, 608.1, 892.6, 1257.1, 470.9, 449.3,
                          189, 563.3, 183.6, 221.5, 1031.6, 727.2, 393.7, 96.4, 199.6,
                          74.5, 0.7]
    pc_subject = [65.4, 66.4, 67.0, 75.6, 70.0, 73.2, 62.6, 51.0, 65.0, 84.0, 60.2, 73.9,
                  62.8, 70.8, 58.2, 74.8, 65.9, 87.6, 71.4, 70.4, 68.7, 74.5, 66.2, 70.9,
                  79.7, 82.2, 65.4, 66.7, 86.8, 51.2, 61.7, 81.9, 68.3, 79.2, 69.9, 74.9,
                  79.9, 55.0, 66.0, 68, 58.6, 69.4, 62.9, 68.2, 65.3, 54.4, 57.3, 74.5,
                  67.8, 66.2, 67.1, 60.1, 67.1, 79.1, 65.1, 80.7, 73.1, 64.4, 60.2, 70.7,
                  62.8, 88.5, 88.2, 73.9, 54.2, 73.2, 65.7, 71.1, 68.5, 64.7, 65.9, 68.4,
                  65.5, 67.0, 64.3, 62.6, 57.2, 73.1, 67.7, 73.5, 73.3, 86.3, 63.3, 55.3, 89.9]
    inter_subject = [72.5, 74.3, 75.8, 80.6, 75.8, 75.4, 69.3, 74.6, 68.9, 87.5,
                     74.1, 82.3, 75.6, 72.9, 64.2, 91.1, 73.9, 88.8, 77.6, 79.7, 78.4,
                     78.9, 71.6, 80.6, 80.8, 84.9, 69.9, 75.4, 87.4, 79.5, 85.6,
                     81.8, 78.6, 87.9, 75.7, 81.2, 82.7, 82.7, 77.3, 89.8, 85.2, 83.2,
                     89.3, 78.4, 75.2, 70.1, 63.4, 83.3, 76.8, 70.7, 74.6, 75.5,
                     75.1, 85.6, 78.5, 84.9, 80.9, 68.9, 77.4, 76.2, 78.6, 95.4, 93.4, 81.3,
                     84.2, 93.4, 75, 77.3, 76.4, 78.9, 74.9, 79, 77, 76.2, 80.7,
                     91.4, 69.7, 82.7, 79.6, 83.1, 83.5, 95.3, 81.7, 76.3, 89.2]
    mobile_th_subject = [1436.9, 1008.2, 1201.9, 2103.7, 985.3, 1042.2, 564, 1026.7, 1053.1,
                         1500, 674.1, 1037.3, 908.2, 825.4, 1199.9, 1488.3, 1366,
                         26913.3, 552.2, 782.4, 1031.5, 400, 1066, 1069.8, 1000,
                         738.7, 610.5, 541.7, 9661.7, 256.9, 224.8, 973.3,
                         6862.5, 937.3, 2198.8, 3821.1, 10.9, 1873.6, 284.9, 676.5,
                         326, 596.3, 950.1, 2508.3, 3761.4, 624.0, 608.9, 4276.8,
                         1345.4, 1123.8, 2462.6, 1106.7, 3864.3, 1896.3, 1162.0, 3091.2,
                         2181.9, 1088.0, 846.9, 4257, 1664.2, 728.4, 1936.5, 3300.4,
                         201.2, 211.4, 596.2, 2103.8, 2772.1, 2550.8, 2456.7,
                         2925.7, 1852.6, 1050.2, 874.8, 956.6, 898.0, 332.5, 2051,
                         1403.6, 791.3, 142.7, 547.8, 134.5, 51.2]
    children_subject = [16.9, 17, 16.6, 16, 16.4, 17, 18.4, 16.9, 17.2, 18.3, 16.4,
                        16, 15.6, 15.1, 16.9, 15, 17.4, 15.6, 18.3, 20, 18.4, 24.3,
                        19.4, 17.8, 15.6, 18.6, 17.7, 16.7, 16.3, 19.9, 21.4, 18.4,
                        18.9, 20.6, 17.1, 17, 17.8, 25.3, 27.6, 21.6, 20.2, 21.2, 33.1,
                        18.8, 20.4, 19.8, 15, 19.8, 20.4, 19.2, 20.3, 18.2, 17.2, 20.1,
                        16, 17.5, 16.7, 16.7, 19.4, 19.8, 21.9, 23.8, 22.9, 19.5, 27.6,
                        34.1, 21.8, 18.9, 20, 22, 19.4, 19.2, 19.7, 19.1, 24.5, 24.2,
                        22.7, 18.9, 17.9, 19.4, 20.2, 18.6, 19.9, 20.8, 22.2]
    rural_subject = [32.4, 29.6, 21.8, 32, 18.2, 24.2, 27, 31.3, 35.4, 18.3, 33.3,
                     27.8, 28, 38.5, 23.7, 25.3, 18.5, 1.6, 18.8, 21.7, 20.9, 25.8,
                     27.3, 22.2, 32.7, 7.9, 28.3, 29.1, 0, 53, 53.8, 49.2, 44.4,
                     33.4, 22.6, 31.8, 6, 54.7, 44.3, 48, 57.2, 35.7, 62, 40.8,
                     37.4, 32.5, 36, 23.1, 33.8, 36.3, 24.1, 21.8, 20.2, 39.2,
                     30.9, 20.3, 24.3, 23.9, 37.7, 14.9, 32.2, 16.1, 7.4, 17.3,
                     70.8, 45.7, 30, 42.8, 22.4, 22.1, 13.9, 20.7, 27.1, 27.9,
                     40.9, 33.7, 31.7, 21.3, 22.6, 17.9, 32.2, 3.9, 17.6, 31.7,
                     28.8]
    mean_income = [32884, 28636, 25955, 32102, 26284, 32559, 25786, 29791, 32534, 47301, 26990, 27328,
                   28256, 27892, 27692, 29396, 29527, 78106, 32596, 36687, 34857, 84171, 29682, 29621, 33235, 46621,
                   26431, 26444, 49375, 30320, 19816, 23033, 37352, 25206, 24995, 31519, 29970, 27666,
                   16877, 22016, 19101, 23963, 24625, 24188, 30409, 21271, 20635, 35694, 25461, 21165,
                   30237, 24292, 33814, 24731, 24135, 29973, 24095, 24596, 21865, 37447, 31851, 90130,
                   54588, 26647, 21683, 18975, 23843, 23917, 32872, 27577, 25441, 31606, 27377, 28871,
                   26222, 46344, 27048, 55381, 37349, 41751, 35508, 70982, 60797, 28126, 89548]
    population_subject = [1541.3, 1182.7, 1342.1, 2305.6, 987, 1001, 628.4, 1096.5, 1128.2,
                          7708.5, 724.7, 1098.3, 921.1, 994.4, 1245.6, 1449.1, 1241.4, 12655.1,
                          609.1, 813.6, 1082.7, 44.4, 1151, 1018.7, 1892.7, 732.9, 592.4, 620.2,
                          5384.3, 463.2, 270, 1901.5, 5683.9, 997.8, 2474.6, 4181.5, 510, 3133.3,
                          515.5, 869.2, 465.4, 693.1, 1498, 2792.8, 4013.8, 675.3, 779, 3894.1,
                          1493.4, 1207.9, 2579.2, 1250.2, 3176.5, 1942.9, 1290.9, 3154.2, 2395.1,
                          1218.3, 818.6, 4290, 1543.4, 547, 1687.7, 3442.8, 221, 330.4, 532,
                          2296.4, 2855.9, 2375, 2633.4, 2785.8, 1903.7, 1070.3, 985.4, 982,
                          1053.5, 311.7, 1877.8, 1301.1, 781.9, 139, 485.6, 156.5, 49.5]
    dict_subject = {'subject_name': subject, 'mean_income_subject': mean_income,
                    'subject_population': population_subject, 'district': districts,
                    'children_subject': children_subject, 'rural_subject': rural_subject,
                    'services_subject': services_subject,
                    'communication_subject': communication_subject,
                    'customer_subject': customer_subject,
                    'mobile_customer': mobile_customer_subject,
                    'mobile_th_subject': mobile_th_subject,
                    'pc_subject': pc_subject,
                    'inter_subject': inter_subject,
                    'good_inter_subject': good_inter_subject,
                    'amount_inf_subject': amount_inf_subject}
    dict_district = {'district_population': population_district, 'district': district,
                     'mean_income_district': mean_income_district,
                     'children_district': children_district, 'rural_district': rural_district,
                     'services_district': services_district,
                     'communication_district': communication_district,
                     'customer_district': customer_district,
                     'mobile_customer_district': mobile_customer_district,
                     'mobile_th_district': mobile_th_district,
                     'pc_district': pc_district,
                     'inter_district': inter_district,
                     'good_inter_district': good_inter_district,
                     'amount_inf_district': amount_inf_district}
    subject_df = pd.DataFrame(data=dict_subject)
    district_df = pd.DataFrame(data=dict_district)
    df = df.merge(subject_df, how='inner')
    df = df.merge(district_df, how='inner')
    return df


def add_tariff_price_feature(test_init):
    """
    Add tariff price feature to the test set
    :param test_init: test set
    :return: test set with tariff price feature
    """
    dfs = pd.read_csv("important/additional_data_all_cities.csv", sep=",", index_col=0)
    test_init['city'] = test_init['city_name']
    df3 = dfs.merge(test_init, on='city', how='right')
    df3['Апгрейд'] = df3.groupby('subject_name')['Апгрейд'].transform(lambda x: x.fillna(x.min()))
    df3['Игровой'] = df3.groupby('subject_name')['Игровой'].transform(lambda x: x.fillna(x.min()))
    df3['Технологии доступа'] = df3.groupby('subject_name')['Технологии доступа'].transform(lambda x: x.fillna(x.min()))
    df3['Технологии доступа PRO'] = df3.groupby('subject_name')['Технологии доступа PRO'].transform(
        lambda x: x.fillna(x.min()))
    df3['Технологии контроля'] = df3.groupby('subject_name')['Технологии контроля'].transform(
        lambda x: x.fillna(x.min()))
    df3.drop(['city'], axis=1, inplace=True)
    return df3


def add_covid_cases_feature(df_test):
    """
    Добавляет количество зараженных как фичу
    :param df_test: датафрейм с данными о пользователях
    :return: датафрейм с добавленной фичей
    """
    df_covid = pd.read_csv('important/owid-covid-data.csv')
    df_covid.drop(['continent', 'location', 'new_cases_smoothed', 'total_deaths', 'new_deaths',
                   'new_deaths_smoothed', 'total_cases_per_million',
                   'new_cases_per_million', 'new_cases_smoothed_per_million',
                   'total_deaths_per_million', 'new_deaths_per_million',
                   'new_deaths_smoothed_per_million', 'reproduction_rate', 'icu_patients',
                   'icu_patients_per_million', 'hosp_patients',
                   'hosp_patients_per_million', 'weekly_icu_admissions',
                   'weekly_icu_admissions_per_million', 'weekly_hosp_admissions',
                   'weekly_hosp_admissions_per_million', 'total_tests', 'new_tests',
                   'total_tests_per_thousand', 'new_tests_per_thousand',
                   'new_tests_smoothed', 'new_tests_smoothed_per_thousand',
                   'positive_rate', 'tests_per_case', 'tests_units', 'total_vaccinations',
                   'people_vaccinated', 'people_fully_vaccinated', 'total_boosters',
                   'new_vaccinations', 'new_vaccinations_smoothed',
                   'total_vaccinations_per_hundred', 'people_vaccinated_per_hundred',
                   'people_fully_vaccinated_per_hundred', 'total_boosters_per_hundred',
                   'new_vaccinations_smoothed_per_million',
                   'new_people_vaccinated_smoothed',
                   'new_people_vaccinated_smoothed_per_hundred', 'stringency_index',
                   'population', 'population_density', 'median_age', 'aged_65_older',
                   'aged_70_older', 'gdp_per_capita', 'extreme_poverty',
                   'cardiovasc_death_rate', 'diabetes_prevalence', 'female_smokers',
                   'male_smokers', 'handwashing_facilities', 'hospital_beds_per_thousand',
                   'life_expectancy', 'human_development_index',
                   'excess_mortality_cumulative_absolute', 'excess_mortality_cumulative',
                   'excess_mortality', 'excess_mortality_cumulative_per_million', ], axis=1, inplace=True)
    df_covid.drop(["iso_code"], axis=1, inplace=True)
    df_covid['period'] = df_covid['date']
    df_covid.drop(["date"], axis=1, inplace=True)
    df3 = df_covid.merge(df_test, on='period', how='right')
    return df3


def add_rozn_feature(df_test):
    """
    Добавляет фичу розничной цены
    :param x: датафрейм с данными о пользователях
    :return: датафрейм с добавленной фичей
    """
    df_rozn = pd.read_csv('important/rozn.csv')
    df_test = df_test.merge(df_rozn, left_on=['subject_name', 'year'], right_on=['region', 'year']).drop(
        ['region'], axis=1)
    return df_test


def add_population_feature(df_test):
    """
    Добавляет фичу населения
    :param x: датафрейм с данными о пользователях
    :return: датафрейм с добавленной фичей
    """
    df_population = pd.read_csv('important/growing_population.csv')
    df_population.value_population = df_population.value_population.astype(np.float32)

    df_test = df_test.merge(df_population, left_on=['subject_name', 'year'], right_on=['territory', 'year']).drop(
        ['territory'], axis=1)

    return df_test


def add_salary_feature(df_test):
    """
    Добавляет фичу о заработной плате
    :param x: датафрейм с данными о пользователях
    :return: датафрейм с добавленной фичей
    """
    df_salary = pd.read_csv('important/salary_data.csv')
    df_salary.value_salary = df_salary.value_salary.astype(np.float32)
    df_test = df_test.merge(df_salary, left_on=['subject_name', 'year'], right_on=['region', 'year']).drop(
        ['region'], axis=1)

    return df_test


class DataPreparator:
    def __init__(self):
        self.add_covid_data = None
        self.add_rt_tariff_data = None
        self.add_region_statistical_data = None
        self.fill_missing_numerical_by = None
        self.fill_missing_categorical_by = None
        self.is_cluster = False

        self.encoders = {"ordinal": OrdinalEncoder(),
                         "cat_boost": CatBoostEncoder(),
                         "target": TargetEncoder()
                         }
        self.scalers = {"standard_scaler": StandardScaler(),
                        "min_max_scaler": MinMaxScaler(),
                        }
        self.clustelizer = None
        self.type_of_encoder = None
        self.type_of_scaler = None

    def transform(self, df,
                  fill_missing_categorical_by=None,
                  fill_missing_numerical_by=None,
                  add_region_statistical_data=True,
                  add_rt_tariff_data=False,
                  add_covid_data=False,
                  add_rozn_data=False,
                  add_salary_data=False,
                  add_growing_population_data=False,
                  type_data='train'):
        """
        Transform the data to the model
        :param add_salary_data: добавлять данные о зарплате
        :param add_rozn_data: добавлять данные о розничной цене
        :param add_covid_data: добавлять данные о коронавирусе
        :param add_rt_tariff_data: добавлять данные о тарифах
        :param add_region_statistical_data: Is include region statistical data
        :param type_data: train or test
        :param df: Dataframe to transform
        :param fill_missing_categorical_by: Fill missing categorical values by [NaN, mean, median, mode] of subject_name
        :param fill_missing_numerical_by: Should fill missing numerical values by [np.mean, np.min, np.median, np.mode] of subject_name
        """
        self.fill_missing_categorical_by = fill_missing_categorical_by
        self.fill_missing_numerical_by = fill_missing_numerical_by
        self.add_region_statistical_data = add_region_statistical_data
        self.add_rt_tariff_data = add_rt_tariff_data
        self.add_covid_data = add_covid_data

        new_df = df.copy()
        new_df = process_period(new_df)
        if add_region_statistical_data:
            new_df = add_statistical_feature(new_df)
        if add_rt_tariff_data:
            new_df = add_tariff_price_feature(new_df)
        if add_covid_data:
            new_df = add_covid_cases_feature(new_df)
        if add_rozn_data:
            new_df = add_rozn_feature(new_df)
        if add_growing_population_data:
            new_df = add_population_feature(new_df)
        if add_salary_data:
            new_df = add_salary_feature(new_df)

        for con in new_df.select_dtypes(include=['float64', 'int64']).columns:
            new_df[con] = new_df[con].astype(np.float32)
        print(new_df.info())
        # Fill categorical missing values
        cat_cols = new_df.select_dtypes(include=['object']).columns.tolist()
        if fill_missing_categorical_by is not None:
            if fill_missing_categorical_by == "NaN":
                for col in cat_cols:
                    new_df[col] = new_df[col].fillna('NaN')
            else:
                for col in cat_cols:
                    new_df[col] = new_df.groupby('subject_name')[col].transform(lambda x: x.fillna(x.mode()))

        # Encoder categorical values
        if self.type_of_encoder:
            new_df[cat_cols] = self.encoders[self.type_of_encoder].transform(new_df[cat_cols])

        # Fill numerical missing values
        num_cols = new_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if fill_missing_numerical_by is not None:
            for col in num_cols:
                new_df[col] = new_df[col].fillna(
                    new_df.groupby('subject_name')[col].transform(fill_missing_numerical_by))
            for col in num_cols:
                new_df[col] = new_df[col].fillna(new_df.groupby('district')[col].transform(fill_missing_numerical_by))

        # Scaling numerical values
        if self.type_of_scaler:
            new_df[num_cols] = self.scalers[self.type_of_scaler].transform(new_df[num_cols])

        # TODO: как то переписать type_data
        # TODO: add month and powertranform
        f_cols = [i for i in num_cols if i.startswith('f')]
        if self.is_cluster and type_data == 'train':
            self._fit_cluster(new_df[f_cols])
            new_df['cluster'] = self._clusterize_data_(new_df[f_cols])
        elif self.is_cluster and type_data == 'test':
            new_df['cluster'] = self._clusterize_data_(new_df[f_cols])
        drop_cols = ['period']
        new_df.drop(drop_cols, axis=1, inplace=True)
        return new_df

    def fit(self,
            X,
            y,
            is_clustering=True,
            type_of_scaler=None,
            type_of_encoder=None,
            ):
        """
        Fit the data to the model
        :param X: Dataframe to fit
        :param y: Target to fit
        :param is_clustering: Is the model a clustering model
        :param type_of_scaler: Type of scaler to use
        :param type_of_encoder: Type of encoder to use
        """
        if type_of_scaler:
            self.type_of_scaler = type_of_scaler
            self.scalers[type_of_scaler].fit(X)

        if type_of_encoder:
            self.type_of_encoder = type_of_encoder
            self.encoders[type_of_encoder].fit(X, y)

        if is_clustering:
            self.is_cluster = True

    def _fit_cluster(self, df: pd.DataFrame):
        BGM = BayesianGaussianMixture(n_components=7, covariance_type='full', max_iter=300, n_init=5)
        BGM.fit(df)
        self.clustelizer = BGM

    def _clusterize_data_(self, df: pd.DataFrame):
        cluster_predict = self.clustelizer.predict(df)
        return cluster_predict
