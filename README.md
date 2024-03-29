# Предсказатель потенциальных клиентов для Ростелекома от команды SFTB

Реализован пайплайн для прогноз потенциальных клиентов Ростелекома с использованием технологий искусственного интеллекта, основываясь на метаинформации о них.

# Установка
- `git clone https://github.com/mansasha21/sftb-rostelecom.git`

# Запуск
```bash
python train.py && python eval.py
```

# Используемое решение

* Для исходных данных проводится обработка пропущенных значений, основанная на статистических распределениях и очистка данных от дубликатов. 
* В исходный датасет добавлены агрегированные статистики по субъектам РФ. Также, он обогащен дополнительными значениями из внешних источников:

  * экономические и демографические показатели по субъектам РФ;
  * статистика использования населением инфокоммуникационных технологий;
  * информация о характеристиках доступных в регионе тарифов Ростелеком;
  * данные о курсах валют, фондовых рынках и ситуации с COVID-19.

* На обработанных данных обучены модели машинного обучения, в частности, AutoML и градиентный бустинг с последующим блендингом в единый алгоритм определения потенциальных клиентов.
* Для последующей оценки полученных предсказаний, разработанный пайплайн включает в себя модуль по выводу мета информации о потенциальных клиентах. Это в совокупности представляет собой систему поддержки принятия решений.

# Уникальность:

Разработанный пайплайн является уникальным решением на рынке за счет использования для обучения обогащенного набор данных, разработанного алгоритма предобработки исходных данных, а также имплементированного механизма блендинга полученных моделей.

# Стек используемых технологий:

`Python3`, `git`, `GitHub` - инструменты разработки  
`LightGBM`, `LAMA`, `CatBoost`, `Scikit-Learn`, `SciPy` - фреймворки машинного обучения  
`Plotly`, `Seaborn` - инструменты визуализации  

# Сравнение моделей

В качестве устойчивого классификационного решения был выбран ансамбль из 5 моделей градиентного бустинга, с временем инференса 93.4 мс, так как он решает прогнозирует потенциальных клиентов с высоким (более 10% на отложенной выборке) результатом по предложенной метрике.

# Проводимые исследования

- `research/catboost.ipynb` - исследования с моделями градиентного бустинга
- `research/signal_eda.ipynb` и `research/spec_eda.ipynb` - анализ исходных данных 


# Разработчики
| Имя                  | Роль           | Контакт               |
|----------------------|----------------|-----------------------|
| Суржиков Александр   | Data Scientist | https://t.me/mansasha |
| ---                  | ---            | ---                   |
| Назаренко Екатерина  | Data Scientist | https://t.me/cutttle  |
| ---                  | ---            | ---                   |
| Чаусов Дмитрий       | Data Scientist | -                     |
| ---                  | ---            | ---                   |
| Ванданов Сергей      | Data Scientist | -                     |
| ---                  | ---            | ---                   |
| Кочетков Максим      | Data Scientist | https://t.me/mahhets  |
| ---                  | ---            | ---                   |
