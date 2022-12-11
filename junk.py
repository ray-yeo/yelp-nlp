    nightlife_restaurants_df = restaurants_df[restaurants_df['Nightlife'] == True]
    nightlife_restaurant_id_list = nightlife_restaurants_df['business_id']  

    bars_restaurants_df = restaurants_df[restaurants_df['Bars'] == True]
    bars_restaurant_id_list = bars_restaurants_df['business_id']  

    sandwiches_restaurants_df = restaurants_df[restaurants_df['Sandwiches'] == True]
    sandwiches_restaurant_id_list = sandwiches_restaurants_df['business_id']  

    pizza_restaurants_df = restaurants_df[restaurants_df['Pizza'] == True]
    pizza_restaurant_id_list = pizza_restaurants_df['business_id']  

    american_new_restaurants_df = restaurants_df[restaurants_df['American (New)'] == True]
    american_new_restaurant_id_list = american_new_restaurants_df['business_id']  

    breakfast_and_brunch_restaurants_df = restaurants_df[restaurants_df['Breakfast & Brunch'] == True]
    breakfast_and_brunch_restaurant_id_list = breakfast_and_brunch_restaurants_df['business_id'] 

    american_traditional_restaurants_df = restaurants_df[restaurants_df['American (Traditional)'] == True]
    american_traditional_restaurant_id_list = american_traditional_restaurants_df['business_id']  

    coffee_and_tea_restaurants_df = restaurants_df[restaurants_df['Coffee & Tea'] == True]
    coffee_and_tea_restaurant_id_list = coffee_and_tea_restaurants_df['business_id']  

    italian_restaurants_df = restaurants_df[restaurants_df['Italian'] == True]
    italian_restaurant_id_list = italian_restaurants_df['business_id']  

    chinese_restaurants_df = restaurants_df[restaurants_df['Chinese'] == True]
    chinese_restaurant_id_list = chinese_restaurants_df['business_id']

    fastfood_restaurants_df = restaurants_df[restaurants_df['Fast Food'] == True]
    fastfood_restaurant_id_list = fastfood_restaurants_df['business_id']  

    burgers_restaurants_df = restaurants_df[restaurants_df['Burgers'] == True]
    burgers_restaurant_id_list = burgers_restaurants_df['business_id']  

    seafood_restaurants_df = restaurants_df[restaurants_df['Seafood'] == True]
    seafood_restaurant_id_list = seafood_restaurants_df['business_id']  

    cafes_restaurants_df = restaurants_df[restaurants_df['Cafes'] == True]
    cafes_restaurant_id_list = cafes_restaurants_df['business_id']  

    mexican_restaurants_df = restaurants_df[restaurants_df['Mexican'] == True]
    mexican_restaurant_id_list = mexican_restaurants_df['business_id']  

    delis_restaurants_df = restaurants_df[restaurants_df['Delis'] == True]
    delis_restaurant_id_list = delis_restaurants_df['business_id']  

    event_planning_restaurants_df = restaurants_df[restaurants_df['Event Planning & Services'] == True]
    event_planning_restaurant_id_list = event_planning_restaurants_df['business_id']  

    salad_restaurants_df = restaurants_df[restaurants_df['Salad'] == True]
    salad_restaurant_id_list = salad_restaurants_df['business_id'] 

    specialty_restaurants_df = restaurants_df[restaurants_df['Specialty Food'] == True]
    specialty_restaurant_id_list = specialty_restaurants_df['business_id'] 

    chicken_wings_restaurants_df = restaurants_df[restaurants_df['Chicken Wings'] == True]
    chicken_wings_restaurant_id_list = chicken_wings_restaurants_df['business_id'] 

    bakeries_restaurants_df = restaurants_df[restaurants_df['Bakeries'] == True]
    bakeries_restaurant_id_list = bakeries_restaurants_df['business_id']  

    japanese_restaurants_df = restaurants_df[restaurants_df['Japanese'] == True]
    japanese_restaurant_id_list = japanese_restaurants_df['business_id']

    asian_fusion_restaurants_df = restaurants_df[restaurants_df['Asian Fusion'] == True]
    asian_fusion_restaurant_id_list = asian_fusion_restaurants_df['business_id']  

    vegetarian_restaurants_df = restaurants_df[restaurants_df['Vegetarian'] == True]
    vegetarian_restaurant_id_list = vegetarian_restaurants_df['business_id'] 

    caterers_restaurants_df = restaurants_df[restaurants_df['Caterers'] == True]
    caterers_restaurant_id_list = caterers_restaurants_df['business_id']

    desserts_restaurants_df = restaurants_df[restaurants_df['Desserts'] == True]
    desserts_restaurant_id_list = desserts_restaurants_df['business_id'] 

    sushi_bars_restaurants_df = restaurants_df[restaurants_df['Sushi Bars'] == True]
    sushi_bars_restaurant_id_list = sushi_bars_restaurants_df['business_id']

    mediterranean_restaurants_df = restaurants_df[restaurants_df['Mediterranean'] == True]
    mediterranean_restaurant_id_list = mediterranean_restaurants_df['business_id'] 

    cheesesteaks_restaurants_df = restaurants_df[restaurants_df['Cheesesteaks'] == True]
    cheesesteaks_restaurant_id_list = cheesesteaks_restaurants_df['business_id'] 

    pubs_restaurants_df = restaurants_df[restaurants_df['Pubs'] == True]
    pubs_restaurant_id_list = pubs_restaurants_df['business_id'] 









        # get just the restaurants
    nightlife_df = df[df['business_id'].isin(nightlife_restaurant_id_list)]
    bars_df = df[df['business_id'].isin(bars_restaurant_id_list)]
    sandwiches_df = df[df['business_id'].isin(sandwiches_restaurant_id_list)]
    pizza_df = df[df['business_id'].isin(pizza_restaurant_id_list)]
    american_new_df = df[df['business_id'].isin(american_new_restaurant_id_list)]
    breakfast_and_brunch_df = df[df['business_id'].isin(breakfast_and_brunch_restaurant_id_list)]
    american_traditional_df =  df[df['business_id'].isin(american_traditional_restaurant_id_list)]
    coffee_and_tea_df =  df[df['business_id'].isin(coffee_and_tea_restaurant_id_list)]
    italian_df = df[df['business_id'].isin(italian_restaurant_id_list)]
    chinese_df = df[df['business_id'].isin(chinese_restaurant_id_list)]
    fastfood_df = df[df['business_id'].isin(fastfood_restaurant_id_list)]
    burgers_df = df[df['business_id'].isin(burgers_restaurant_id_list)]
    seafood_df = df[df['business_id'].isin(seafood_restaurant_id_list)]
    cafes_df = df[df['business_id'].isin(cafes_restaurant_id_list)]
    mexican_df = df[df['business_id'].isin(mexican_restaurant_id_list)]
    delis_df = df[df['business_id'].isin(delis_restaurant_id_list)]
    event_planning_df = df[df['business_id'].isin(event_planning_restaurant_id_list)]
    salad_df = df[df['business_id'].isin(salad_restaurant_id_list)]
    specialty_df = df[df['business_id'].isin(specialty_restaurant_id_list)]
    chicken_wings_df = df[df['business_id'].isin(chicken_wings_restaurant_id_list)]
    bakeries_df = df[df['business_id'].isin(bakeries_restaurant_id_list)]
    japanese_df = df[df['business_id'].isin(japanese_restaurant_id_list)]
    asian_fusion_df = df[df['business_id'].isin(asian_fusion_restaurant_id_list)]
    vegetarian_df = df[df['business_id'].isin(vegetarian_restaurant_id_list)]
    caterers_df = df[df['business_id'].isin(caterers_restaurant_id_list)]
    desserts_df = df[df['business_id'].isin(desserts_restaurant_id_list)]
    sushi_bars_df = df[df['business_id'].isin(sushi_bars_restaurant_id_list)]
    mediterranean_df = df[df['business_id'].isin(mediterranean_restaurant_id_list)]
    cheeseteaks_df = df[df['business_id'].isin(cheesesteaks_restaurant_id_list)]
    pubs_df = df[df['business_id'].isin(pubs_restaurant_id_list)]