import re

import requests
from bs4 import BeautifulSoup
import pandas as pd
from transliterate import translit


def get_rt_tariffs(read_path,
                   save_path,
                   save_every=25):
    '''
    Получает тарифы с сайта rt-internet.ru и сохраняет их в csv файл
    :param read_path: путь к файлу с данными
    :param save_path: путь куда все сохранить
    :param save_every: каждый какой тариф сохранять
    '''
    df = pd.DataFrame(columns=['city', 'latina_name', 'tariff_name',
                               'tariff_price', 'tariff_speed', 'wifi_router_price',
                               'is_first_month_free', "is_correct_info", 'additional_info'])
    data_df = pd.read_csv(read_path, sep=";")

    from_start = 0
    all_cities = data_df.city_name.unique().tolist()[from_start:]
    i = from_start

    for city in all_cities:
        if str(city) == "nan":
            continue
        my_name = translit(city, "ru", reversed=True, strict=False).replace("'", "").replace(" ", "-").replace(".", ""). \
            replace(",", "").replace('/', '').replace('(', '').replace(')', '').replace('[', '').replace(']', ''). \
            replace('«', '').replace('»', '')
        url = "https://" + my_name + ".rt-internet.ru/"
        page_response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        page_content = BeautifulSoup(page_response.content, "html.parser")
        reviews_stars_for_page = page_content.find_all("div", class_="tarifscard")
        is_correct = str(page_content).find("самый популярный интернет-провайдер в") != -1
        for review_stars in reviews_stars_for_page:
            price = review_stars.find("div", class_="tarif-ap").text.strip()
            speed = re.findall("\d+", review_stars.find("div", class_="attr-int").text.strip())[0]
            try:
                wifi_router_price = re.findall("\d+", review_stars.find("div", class_="equip-attr").text.strip())[0]
            except Exception as e:
                wifi_router_price = "-1"
            is_month_free = review_stars.find("div", class_="tarif-comap") is not None
            name = review_stars.find("div", class_="tarifs-name").text.strip()
            try:
                additional_info = \
                    str(review_stars.find("div", class_="tarifs-name")).split("data-desc=\"")[1].split("\">")[0]
            except Exception as e:
                additional_info = ""
            tmp_df = pd.DataFrame([{
                "city": city,
                "latina_name": my_name,
                "tariff_name": name,
                "tariff_price": price,
                "tariff_speed": speed,
                "wifi_router_price": wifi_router_price,
                "is_first_month_free": is_month_free,
                "is_correct_info": is_correct,
                "additional_info": additional_info,
            }])
            df = df.append(tmp_df, ignore_index=True)
        i += 1
        if i % save_every == 0:
            df.to_csv(save_path + str(i) + ".csv", index=False)
    return df


if __name__ == "__main__":
    get_rt_tariffs(read_path="./data/train.csv",
                   save_path="./data/rt_parsed.csv",
                   save_every=25)
