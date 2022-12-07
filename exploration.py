import json
from collections import Counter




# businesses = []
# with open('yelp_dataset/yelp_academic_dataset_business.json') as f:
#     for jsonObj in f:
#         businessDict = json.loads(jsonObj)
#         businesses.append(businessDict)

# print("Printing each JSON Decoded Object")

# state_list = []
# city_list = []
# zip_list = []

# for business in businesses:
#     state_list.append(business["state"])
#     city_list.append(business["city"])
#     zip_list.append(business["postal_code"])

# print(Counter(state_list))
# print(Counter(city_list))
# print(Counter(zip_list))

"""
In ascending order: the number of businesses in each area
{'PA': 34039, 'FL': 26330, 'TN': 12056, 'IN': 11247, 'MO': 10913, 'LA': 9924, 'AZ': 9912, 'NJ': 8536, 'NV': 7715, 'AB': 5573, 'CA': 5203, 'ID': 4467, 'DE': 22
{'Philadelphia': 14569, 'Tucson': 9250, 'Tampa': 9050, 'Indianapolis': 7540, 'Nashville': 6971, 'New Orleans': 6209, 'Reno': 5935, 'Edmonton': 5054, 'Saint Louis': 4827, 'Santa Barbara': 3829, 'Boise': 2937, 'Clearwater': 2221, 'Saint Petersburg': 1663, 'Metairie': 1643}
{'93101': 1866, '89502': 1804, '70130': 1512, '19103': 1362, '19107': 1353, '19147': 1255, '37203': 1179, '85705': 1069, '33511': 940, '89431': 915, '93117': 901, '46032': 888, '19106': 851, '85719': 835}
"""




# create dictionary between business ID and zip for use in reviews (i.e. look only at reviews in Philly)
# businesses = []
# with open('yelp_dataset/yelp_academic_dataset_business.json') as f:
#     for jsonObj in f:
#         businessDict = json.loads(jsonObj)
#         businesses.append(businessDict)

# # business_to_state_dict = {}
# business_to_zip_dict = {}

# for business in businesses:
#     if business['city'] == "Philadelphia":
#         if business['business_id'] not in business_to_zip_dict:
#             business_to_zip_dict[business['business_id']] = business['postal_code']

# print(business_to_zip_dict)




# reviews


#reading in reviews
philly_reviews = []
people_who_left_review_in_philly = set()
with open('yelp_dataset/yelp_academic_dataset_review.json') as f:
    for jsonObj in f:
        reviewsDict = json.loads(jsonObj)
        if reviewsDict['business_id'] in business_to_zip_dict:
            philly_reviews.append(reviewsDict)
            people_who_left_review_in_philly.add(reviewsDict['user_id'])
            # print("philly!")

final = json.dumps(philly_reviews)
print(len(philly_reviews))

# writing to json, phillyReviews.json holds all the reviews made in philadelphia
with open("phillyReviews.json", "w") as outfile:
    outfile.write(final)

# now, we have a subset of our data which consists of philly reviews only
# from here, we can track individual reviewers
# we have zip code as well, which allows us to use census data


# reviews hold user id which maps to interesting characteristics about the reviewer




#exploration about users
#reading in reviews

# philly_reviewers = []
# with open('yelp_dataset/yelp_academic_dataset_user.json') as f:
#     for jsonObj in f:
#         userDict = json.loads(jsonObj)
#         if userDict['user_id'] in people_who_left_review_in_philly:
#             philly_reviewers.append(userDict)
#             # print("philly!")

# final = json.dumps(philly_reviewers)
# print(len(philly_reviewers))

# # writing to json, phillyReviewers.json holds individuals who have made at least one review in Philly
# with open("phillyReviewers.json", "w") as outfile:
#     outfile.write(final)




philly_busineses = []
with open('yelp_dataset/yelp_academic_dataset_business.json') as f:
    print("read in file")
    for jsonObj in f:
        businessDict = json.loads(jsonObj)
        if businessDict['city'] == "Philadelphia":
            philly_busineses.append(businessDict)
final = json.dumps(philly_busineses)
print(len(philly_busineses))
with open("phillyBusinesses.json", "w") as outfile:
    print("writing to file")
    outfile.write(final)