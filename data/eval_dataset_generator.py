import instructor
from pydantic import BaseModel
from openai import OpenAI
from src.models import EvalQueryList
from dotenv import load_dotenv
import os
import random
import json

load_dotenv
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

message_style = """
Thank👍🏾👍🏾👍🏾 this is Mr.
Chimbanga
Yes it can be Mr.victor
Hello victor
How is weekend?
I am ok.when are you coming shaaban Robert sec.school again victor
Hello victor how is your weekend going on
Welcome my home bagamoyo,so you can see my family
Waoo have a great weekend and Easter
Happy Easter victor, how about Sweden
Tanzania is okay but very 🔥hot
Hi Victor, are you still in the class??
Yes, I am waiting you here at form two A.
A student will come to take you
Thank you Victor for the nice photos. It was wonderful.
I hope your having a good weekend Victor.
You are
Hi Victor. I hope you are doing good.
I have a gift for you as I promised. It was wonderful and cheerful time being with you in my class.
Please make a time for me so that we can meet.🥂
Hi victor..
about your breakfast you'll be having it everyday 8:00PM,  toninght here at the counter...
For a dinner you have go down at the reception to make your Order
Good morning Victor,
Well,we meet in the TRC
OK am coming
Hello Victor,
We can meet at 9am, the lesson starts at 11.00am
Yes, we can meet there
Ok Victor, see you tomorrow
I'm assigned another work to do, from the academic office
Please, can we meet at 10.20?
Ok sir, you can join
I'm in T2
Can we do the interview tomorrow,
I'm on duty today
sounds good, we can meet at this time
Ooo am sorry do day i was not online through out the whole day
Can you come and bring the documents to the head of school
Sorry  have you got the permission to conduct research
Can you come to submit to the head of school
You have postponed your  research..what happened  bro?
Why not bro! Let us make it
You have to come early before 8am we will fix the time schedule....
Hello Victor. 
I am very happy to receive your message. You are mostly welcome
Regard
Mr stedius
Good evening. You are mostly welcome
So far, I don't know. Possibly, he will reply to your msg anytime from today.
I have not seen him at work. I will try to look for him today
Good morning, Mr Victor. Headmaster received your documents,  but it seems he is not convinced with the documents. This is the way I see. Otherwise, he could have replied your email.
OK, sending another documents will be a good idea. Let us convince him
Please send it to email
That document from municipal council is what actually needed
You have to come and request for it. He will give you the time to meet him
Please come on tuesday. If possible, come with your documents , specifically from the municipal council addressing being permitted to do research at Shaaban robert Sec.  This will make headmaster to be comfortable.
If you want to make an appointment on tuesday then you must make it on Monday. If you will come on tuesday, then you will meet headmaster on Wednesday.
Make sure you have permission from the municipal council of ilala. Allowing you to do research at shaaban robert sec school. Otherwise, it will not be possible to it.
Aga Khan is totally different to shaaban robert Sec School. Aga Khan is taking the Cambridge curriculum, which is not under Necta. Shaaban robert Secondary is taking a necta curriculum. Aga Khan can have teachers from Abroad with minor restrictions, but shaaban robert must have very genuine reason to have an expert from abroad.
So bring it at shaaban robert
Sorry for the late reply. My timetable was very tight.
Please come and submit to the secretary of the headmaster. There is no problem. I will make sure it reaches the table of headmaster
Mathematics lessons are normally conducted in the morning , first lesson, 2nd ,3rd and 4th
First lesson start at 7.25 am
You are mostly welcome
Hello Mr Voctor
Hopefully, everything is fine on your side. Just saying hi to you
"""

topics = [
    {
        "chapter": "Chapter One (Human Activities)"
    },
    {
        "chapter": "Chapter Two (Agriculture)"
    },
    {
        "chapter": "Chapter Three (Water management for economic development)"
    },
    {
        "chapter": "Chapter Four (Sustainable use of forest resources)"
    },
    {
        "chapter": "Chapter Five (Mining Industry)"
    },
    {
        "chapter": "Chapter six (Tourism)"
    },
    {
        "chapter": "Chapter Seven (Manufacturing Industry)"
    },
    {
        "chapter": "Chapter Eight (Sustainable use of power and energy resources)"
    },
    {
        "chapter": "Chapter Nine (Transport)"
    }
]
list_of_topics = [
    "The concept of human activities",
    "Types of human activities",
    "Importance of human activities",
    "The concept of agriculture",
    "Crop cultivation",
    "Large-scale crop cultivation",
    "Coffee",
    "Oil palm",
    "Tea",
    "Conditions for growing tea",
    "Sisal",
    "Clove",
    "Cereal crops",
    "Contribution of crop production to the economy of Tanzania",
    "Challenges facing large-scale crop cultivation in Tanzania",
    "Ways to improve large-scale crop cultivation in Tanzania",
    "Contribution of crop production to the economy of the United States of America",
    "Challenges facing large-scale agriculture in USA",
    "Livestock-keeping",
    "Nomadic pastoralism/True pastoralism",
    "Commercial livestock-keeping",
    "The concept of water management",
    "Sources of water",
    "Fresh water and hydrological cycle",
    "Evaporation",
    "Evapotranspiration",
    "Condensation",
    "Precipitation",
    "Economic importance of water",
    "Family size, water supply and quality of life",
    "Effects of distance to water sources on the girl-child in Tanzania",
    "Relationship between vegetation and water supply",
    "River basin development",
    "Land reclamation",
    "Sustainable use of water resources",
    "Water pollution",
    "Ways of conserving water",
    "The concept of forest",
    "Timber production in the world",
    "Transportation of timber",
    "Challenges facing timber transportation in the world",
    "Challenges facing the forestry industry",
    "Addressing challenges facing forest resources",
    "The concept of mining industry",
    "Methods of mining",
    "Methods of mineral processing",
    "Contribution of the mining industry to the economy of Tanzania",
    "Ways of minimising the effects of mining on the environment",
    "The concept of tourism",
    "Factors for the development and growth of tourism in the world",
    "Impact of tourism in the world",
    "Focal studies on the tourism industry",
    "Tourism in Switzerland",
    "Tourism in Namibia",
    "The concept of manufacturing industry",
    "The importance of manufacturing industries",
    "Types of manufacturing industries",
    "Pollutants from manufacturing industries",
    "Major ways of reducing industrial pollution",
    "Car manufacturing industries in Japan",
    "Factors contributing to the development of industries in Japan",
    "Electronic equipment industry in South Korea",
    "Textile industries in Tanzania",
    "Lessons from Japanese and South Korean industries for Tanzania",
    "The concept of Power and Energy",
    "Major sources of power and energy",
    "Importance of power and energy",
    "Challenges facing power and energy production",
    "Addressing the challenges of energy and power production",
    "Hydro-electric power and biogas in Tanzania",
    "Importance of HEP and biogas energy resources in Tanzania",
    "Challenges to harnessing power and energy in Tanzania",
    "Addressing challenges to power and energy harnessing in Tanzania",
    "Solar and wind energy harnessing in USA",
    "The concept of transport",
    "Main types of transport",
    "Importance of the transport industry in East Africa",
    "Advantage and disadvantages of the transport sector",
    "Challenges facing the transport industry in East Africa",
    "Measures to address transport challenges in Tanzania",
    "Primary activities",
    "Natural forests",
    "Secondary activities",
    "Tertiary activities",
    "Small-scale crop cultivation",
    "Shifting cultivation",
    "Crop rotation",
    "Bush fallowing",
    "Characteristics of small-scale crop cultivation",
    "Relationship between population growth and small-scale crop production",
    "Effects of rapid population growth on small-scale crop cultivation",
    "Advantages of small-scale crop cultivation",
    "Disadvantages of small-scale crop cultivation",
    "Ways of improving small-scale crop cultivation",
    "Characteristics of large-scale crop cultivation",
    "Types of large-scale crop cultivation",
    "Conditions necessary for growing coffee",
    "Farm preparation, planting, and care of coffee",
    "Harvesting, processing, storage and transportation",
    "Uses of coffee",
    "Conditions necessary for growing cotton",
    "Farm preparation, planting and care of cotton",
    "Harvesting, processing, storage and transportation of cotton",
    "Uses of cotton",
    "Conditions necessary for growing oil palm",
    "Farm preparation, planting and care of oil palm",
    "Harvesting, processing, storage, and transportation of oil palm",
    "Uses of palm oil",
    "Farm preparation, planting and caring of tea",
    "Harvesting, processing, storage and transportation of tea",
    "Uses of tea",
    "Conditions for growing sisal",
    "Farm preparation, planting and caring of sisal",
    "Harvesting, processing and transportation of sisal",
    "Uses of sisal",
    "Conditions for growing cloves",
    "Farm preparation, planting and care of cloves",
    "Harvesting, processing and transportation of cloves",
    "Uses of cloves",
    "Maize",
    "Wheat",
    "Rice",
    "Characteristics of nomadic pastoralism",
    "Advantages of nomadic pastoralism",
    "Disadvantages of nomadic pastoralism",
    "Semi-nomadism",
    "Transhumance",
    "Characteristics of commercial livestock-keeping",
    "Advantages of commercial livestock-keeping",
    "Disadvantages of commercial livestock-keeping",
    "Benefits of livestock-keeping",
    "Challenges facing livestock-keeping",
    "Ways of solving problems facing pastoralists",
    "Livestock-keeping in Tanzania",
    "Challenges facing livestock-keeping in Tanzania",
    "Livestock-keeping in Australia",
    "Similarities between livestock -keeping in Tanzania and Australia",
    "Differences between livestock-keeping in Tanzania and Australia",
    "Economic importance of livestock-keeping",
    "Temperature",
    "Humidity",
    "Wind",
    "Soil",
    "Types of plants",
    "Infiltration",
    "Requirements for establishing a river basin project",
    "Benefits of river basin development projects",
    "Challenges facing river basin projects",
    "Rufiji Basin in Tanzania",
    "Other river basin development projects",
    "Methods used in land reclamation",
    "Land reclamation in Tanzania",
    "Salt water",
    "Underground water",
    "Surface water",
    "Boreholes",
    "Underground water",
    "Springs",
    "Wells",
    "Resources obtained from water",
    "Other water resources",
    "Problems resulting from the extraction of water resources",
    "Natural forests",
    "Equatorial rainforest",
    "Characteristics of an equatorial rainforest",
    "Mangrove forests",
    "Characteristics of a mangrove forest",
    "Tropical monsoon forests",
    "Characteristics of a tropical monsoon forest",
    "Deciduous forests",
    "Characteristics of a deciduous forest",
    "Coniferous forests",
    "Characteristics of coniferous forests",
    "Planted forests",
    "Characteristics of planted forests",
    "Factors influencing distribution of forests",
    "Importance of forest resources",
    "Importance of forests in ecological and environmental balance",
    "Mining",
    "Mining industry",
    "Coal mining industry",
    "Metal mining industry",
    "Non-metallic mineral mining industry",
    "Oil and gas extraction industry",
    "The occurrence of minerals",
    "Distribution of minerals in the world",
    "Uses of some minerals",
    "Surface mining",
    "Underground or shaft mining",
    "Alluvial/Placer mining",
    "In-situ mining",
    "Crushing and grinding",
    "Separation",
    "Sizing",
    "Floatation",
    "Oil production in the Middle East",
    "Importance of oil production to Middle East countries",
    "Challenges associated with oil production in the Middle East",
    "Natural gas production in Tanzania",
    "Advantages of natural gas in Tanzania",
    "Challenges of natural gas production in Tanzania",
    "Tanzanias efforts in addressing the challenges of natural gas production",
    "Positive impact of tourism",
    "Negative impact of tourism",
    "Ways of addressing the negative impact of tourism",
    "Factors for development of tourism in Switzerland",
    "Importance of tourism in Switzerland",
    "Challenges facing the tourism industry in Switzerland",
    "Promoting access to tourist attraction centers",
    "Protecting the biodiversity",
    "Improving the tourist environment",
    "Factors for the development of tourism in Namibia",
    "Improved infrastructure",
    "Strong tourism policy",
    "Namibia Tourism Board",
    "Regional cooperation",
    "Importance of tourism to Namibia",
    "Challenges facing tourism in Namibia",
    "Management of national parks and game reserves in Tanzania",
    "Factors for the development of tourism in Tanzania",
    "Importance of tourism in Tanzania",
    "Challenges facing tourism in Tanzania",
    "Processing industries",
    "Fabrication industries",
    "Types of manufacturing industries in East Africa",
    "Factors for the location of manufacturing industries",
    "Industrial pollution",
    "Air pollution",
    "Gases",
    "Industrial solid waste",
    "Liquid pollutants",
    "Noise pollution",
    "Factors contributing to the development of electronic manufacturing industries in South Korea",
    "Importance of textile industries in Tanzania",
    "Challenges facing textile industries in Tanzania",
    "Non-renewable energy sources",
    "Importance of solar and wind energy in USA",
    "Challenges facing solar and wind energy in USA",
    "Addressing challenges facing solar and wind energy in USA"
]

def generate_educator_queries(num_queries, topics, question_types):
    system_prompt = """
    Generate a series of questions as requested by Tanzanian educators on specific topics for secondary school geography. 
    The queries should reflect a variety of teaching styles and may include multiple choice, true/false, or short answer questions.
    """

    queries = []
    for _ in range(num_queries):
        topic = random.choice(topics)
        question_type = random.choice(question_types)
        prompt = f"Generate a {question_type} question about {topic['chapter']}."
        
        try:
            response = client.Completions.create(
                model="text-davinci-002",
                prompt=system_prompt + prompt,
                max_tokens=100
            )
            question = response.choices[0].text.strip()
            queries.append({
                "prompt": prompt,
                "question": question,
                "metadata": {
                    "topic": topic['chapter'],
                    "type": question_type
                }
            })
        except Exception as e:
            print(f"Error generating query: {e}")
    
    return queries


topics = [
    {"chapter": "Chapter One (Human Activities)"},
    {"chapter": "Chapter Two (Agriculture)"},
    # Add the rest of your chapters here
]

question_types = ["multiple choice", "true/false", "short answer"]

def save_queries_to_json(queries, filename='educator_queries.json'):
    with open(filename, 'w') as f:
        json.dump(queries, f, indent=4)

# Generate the queries
total_queries = 300
queries_per_type = total_queries // len(question_types)
all_queries = []

for q_type in question_types:
    generated_queries = generate_educator_queries(queries_per_type, topics, [q_type])
    all_queries.extend(generated_queries)

# Save to JSON
save_queries_to_json(all_queries)

if __name__ == "__main__":
    total_queries = 300
    queries_per_type = total_queries // len(question_types)
    all_queries = []

    for q_type in question_types:
        generated_queries = generate_educator_queries(queries_per_type, topics, [q_type])
        all_queries.extend(generated_queries)

    # Save to JSON
    save_queries_to_json(all_queries)






# # Patch the OpenAI client
client = instructor.from_openai(OpenAI(api_key=OPENAI_API_KEY))

system_prompt = """
    Create a json file containing a bunch of example prompts from Tanzanian teachers asking a chatbot to generate questions on various topics from Form 2 Geography. 
    They do not have fluent english when writing the prompts. In addition, each prompt example has associated metadata such as topic, subject, and most importantly, 
    what type of question they're looking for (eg. long-answer, short-answer, or true-false).
"""

completion = client.chat.completions.create(
  model="gpt-4-turbo",
  response_model=EvalQueryList,
  messages=[
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "Please generate 10 queries from tanzanian educators on the following topics."}
  ]
)

print(completion.choices[0].message)
