import re

import requests
from bs4 import BeautifulSoup
import pandas as pd
from transliterate import translit


def get_context():
    df = pd.DataFrame(
        columns=['city', 'latina_name', 'tariff_name', 'tariff_price', 'tariff_speed', 'wifi_router_price',
                 'is_first_month_free', "is_correct_info", 'additional_info'])
    data_df = pd.read_csv("C:/Users/Sergey/PycharmProjects/RZD_Digital/SFTB/data/train.csv", sep=";")
    from_start = 0
    all_cities = data_df.city_name.unique().tolist()[from_start:]

    i = from_start
    for city in all_cities:
        if str(city) == "nan":
            continue
        my_name = translit(city, "ru", reversed=True, strict=False).replace("'", "").replace(" ", "-").replace(".", ""). \
            replace(",", "").replace('/', '').replace('(', '').replace(')', '').replace('[', '').replace(']',
                                                                                                         '').replace(
            '«', '').replace('»', '')
        url = "https://" + my_name + ".rt-internet.ru/"
        print(my_name + " " + city)
        page_response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        print(page_response)
        page_content = BeautifulSoup(page_response.content, "html.parser")
        # print(page_content)
        reviews_stars_for_page = page_content.find_all("div", class_="tarifscard")
        is_correct = str(page_content).find("самый популярный интернет-провайдер в") != -1
        for review_stars in reviews_stars_for_page:
            price = review_stars.find("div", class_="tarif-ap").text.strip()
            speed = re.findall("\d+", review_stars.find("div", class_="attr-int").text.strip())[0]
            # print(review_stars.find("div", class_="attr-int").text.strip())
            try:
                wifi_router_price = re.findall("\d+", review_stars.find("div", class_="equip-attr").text.strip())[0]
            except:
                wifi_router_price = "-1"
            is_month_free = review_stars.find("div", class_="tarif-comap") is not None
            name = review_stars.find("div", class_="tarifs-name").text.strip()
            try:
                additional_info = \
                    str(review_stars.find("div", class_="tarifs-name")).split("data-desc=\"")[1].split("\">")[0]
            except:
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
        if i % 25:
            df.to_csv(
                r"C:\Users\Sergey\PycharmProjects\Test\additional_data_from_test_" + str(from_start) + "_to_" + str(
                    i) + ".csv", index=False)

        print(i)
    return df


df = get_context("https://snt-yujnyy.rt-internet.ru/")
