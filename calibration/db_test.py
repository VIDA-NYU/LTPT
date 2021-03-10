from pymongo import MongoClient

client = MongoClient("mongodb+srv://yurii:yuriimongo@ltpt.qsvio.mongodb.net")
db = client.ltpt
collection = db["videos"]

videos = list(collection.find({}))
print(len(videos))
print(videos[0])
