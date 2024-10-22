from typing import List
import instructor
from openai import OpenAI
from src.models import EvalQuery
from dotenv import load_dotenv
import os
import random
import json
from tqdm import tqdm
from src.utils import save_objects_as_json
from concurrent.futures import ThreadPoolExecutor, as_completed

"""
This file can generate test prompts from Tanzanian teachers to my pipeline. There are geography topics that exist within the textbook but also a list of non-geography topics that are from all over
the place for when I want to generate a control eval set. 

TODO: For the future I could experiment with using openai's batching feature instead of parallelizing many API calls to reduce the risk for openai.RateLimitError
"""


load_dotenv
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATA_DIR = os.getenv("DATA_DIR_PATH")

# This was taken from my texts with secondary school teachers but it has been anonymized
teacher_writing_style = """
    Thank👍🏾👍🏾👍🏾 this is Mr. X. Yes it can be Mr.victor. Hello victor. How is weekend? I am ok.when are you coming School name sec.school again victor. Hello victor how is your weekend going on. Welcome my home,so you can see my family
    Waoo have a great weekend and Easter. Happy Easter victor, how about Sweden. Tanzania is okay but very 🔥hot. Hi Victor, are you still in the class?? Yes, I am waiting you here at form two A.
    A student will come to take you. Thank you Victor for the nice photos. It was wonderful. I hope your having a good weekend Victor. You are. Hi Victor. I hope you are doing good.
    I have a gift for you as I promised. It was wonderful and cheerful time being with you in my class.
    Hi victor..about your breakfast you'll be having it everyday 8:00PM,  toninght here at the counter...For a dinner you have go down at the reception. Good morning Victor,
    Well,we meet in the TRC. OK am coming. Hello Victor. We can meet at 9am, the lesson starts at 11.00am. Yes, we can meet there. Ok Victor, see you tomorrow. I'm assigned another work to do, from the academic office
    Please, can we meet at 10.20? Ok sir, you can join. I'm in T2. Can we do the interview tomorrow. I'm on duty today. sounds good, we can meet at this time. Ooo am sorry do day i was not online through out the whole day
    Can you come and bring the documents to the head of school. Sorry  have you got the permission to conduct research Can you come to submit to the head of school
    You have postponed your  research..what happened  bro? Why not bro! Let us make it. You have to come early before 8am we will fix the time schedule....
    Hello Victor. I am very happy to receive your message. You are mostly welcome. Regard.. Good evening. You are mostly welcome. So far, I don't know. Possibly, he will reply to your msg anytime from today.
    I have not seen him at work. I will try to look for him today. Good morning, Mr Victor. Headmaster received your documents,  but it seems he is not convinced with the documents. This is the way I see. Otherwise, he could have replied your email.
    OK, sending another documents will be a good idea. Let us convince him. Please send it to email. That document from municipal council is what actually needed
    You have to come and request for it. He will give you the time to meet him. Please come on tuesday. If possible, come with your documents , specifically from the municipal council addressing being permitted to do research at School name Sec.  This will make headmaster to be comfortable.
    If you want to make an appointment on tuesday then you must make it on Monday. If you will come on tuesday, then you will meet headmaster on Wednesday. Make sure you have permission from the municipal council of name. Allowing you to do research at School name sec school. Otherwise, it will not be possible to it.
    school is totally different to school Sec School. school is taking the Cambridge curriculum, which is not under Necta. School name Secondary is taking a necta curriculum. The school can have teachers from Abroad with minor restrictions, but School name must have very genuine reason to have an expert from abroad.
    So bring it at school name. Sorry for the late reply. My timetable was very tight. Please come and submit to the secretary of the headmaster. There is no problem. I will make sure it reaches the table of headmaster
    Mathematics lessons are normally conducted in the morning , first lesson, 2nd ,3rd and 4th. First lesson start at 7.25 am. You are mostly welcome. Hello Mr Voctor. Hopefully, everything is fine on your side. Just saying hi to you
"""

list_of_unrelated_topics = [
    "Classical Music Composition",
    "Quantum Computing Basics",
    "Modern Art Movements",
    "The Science of Sleep",
    "Ethics in Artificial Intelligence",
    "Film Noir Characteristics",
    "Advancements in Robotics",
    "History of the Olympic Games",
    "DNA Replication Processes",
    "The Psychology of Color",
    "Fundamentals of Baking",
    "Ancient Egyptian Mythology",
    "Stock Market Trends",
    "Animal Behavioral Studies",
    "Theories of Personality",
    "Introduction to Philosophy",
    "Mechanisms of Breathing",
    "Evolution of Video Games",
    "Cybersecurity Threats",
    "Child Language Acquisition",
    "Techniques in Modern Painting",
    "Nutritional Science",
    "Fundamentals of Jazz Music",
    "Biographies of Inventors",
    "The Sociology of Fashion",
    "Molecular Gastronomy",
    "Blockchain and Cryptocurrency",
    "Major Theories of Motivation",
    "History of Comic Books",
    "Understanding Wine Terroir"
]

# These topics were generated by modifying a bunch of subtopics that existed in the geography textbook
list_of_modified_topics = [
    "The concept of human activities",
    "Varieties of human activities",
    "Significance of human activities",
    "Basics of agriculture",
    "Growing crops",
    "Extensive crop cultivation",
    "Coffee",
    "Palm oil",
    "Tea",
    "Environmental requirements for tea cultivation",
    "Sisal",
    "Cloves",
    "Cereal crops",
    "Economic impacts of agriculture in Tanzania",
    "Obstacles in extensive farming in Tanzania",
    "Enhancing large-scale farming in Tanzania",
    "Economic impacts of agriculture in the USA",
    "Issues in American large-scale farming",
    "Raising livestock",
    "Traditional pastoralism",
    "Industrial livestock farming",
    "Basics of water management",
    "Water sources",
    "Freshwater sources and the water cycle",
    "Evaporation",
    "Plant Transpiration",
    "Formation of Condensate",
    "Types of Precipitation",
    "Economic value of water",
    "Impact of family size on water access and quality",
    "Influence of water source distance on Tanzanian girls",
    "Interplay between vegetation and water availability",
    "Development of river catchments",
    "Restoration of land",
    "Sustainable management of water resources",
    "Contamination of water sources",
    "Techniques for water conservation",
    "Understanding forests",
    "Global timber industry",
    "Logistics of timber transportation",
    "Global challenges in timber logistics",
    "Problems in the forestry sector",
    "Strategies to overcome forest resource challenges",
    "Fundamentals of the mining sector",
    "Mining techniques",
    "Mineral refinement methods",
    "Mining sector's economic contributions in Tanzania",
    "Reducing environmental impacts of mining",
    "Basics of tourism",
    "Drivers of global tourism growth",
    "Global impacts of tourism",
    "Case studies in the tourism sector",
    "Tourism in Switzerland",
    "Tourism in Namibia",
    "Basics of the manufacturing sector",
    "Significance of manufacturing to the economy",
    "Categories of manufacturing",
    "Emissions from manufacturing activities",
    "Methods to mitigate industrial pollution",
    "Automobile industry in Japan",
    "Factors in Japanese industrial success",
    "South Korea's electronics industry",
    "Textile sector in Tanzania",
    "Industrial insights from Japan and South Korea for Tanzania",
    "Fundamentals of Power and Energy",
    "Primary sources of energy",
    "Significance of energy resources",
    "Obstacles in energy production",
    "Mitigation strategies for energy production challenges",
    "Hydroelectric and biogas energy in Tanzania",
    "Significance of hydroelectric power and biogas in Tanzania",
    "Difficulties in energy acquisition in Tanzania",
    "Solutions to energy acquisition challenges in Tanzania",
    "Utilization of solar and wind power in the USA",
    "Overview of transportation",
    "Key transport modalities",
    "Role of transportation in East Africa's economy",
    "Pros and cons of the transportation industry",
    "Problems encountered by the East African transport sector",
    "Strategies to improve transport in Tanzania",
    "Primary economic activities",
    "Undisturbed forests",
    "Secondary economic activities",
    "Tertiary economic activities",
    "Subsistence crop farming",
    "Nomadic farming",
    "Practice of crop rotation",
    "Traditional fallowing techniques",
    "Traits of subsistence farming",
    "Impact of population dynamics on small-scale farming",
    "Population growth effects on subsistence farming",
    "Benefits of subsistence farming",
    "Drawbacks of subsistence farming",
    "Enhancement methods for subsistence farming",
    "Features of commercial farming",
    "Varieties of commercial farming",
    "Essential conditions for coffee cultivation",
    "Procedures for coffee farming",
    "Post-harvest processing of coffee",
    "Applications of coffee",
    "Essential conditions for cotton cultivation",
    "Procedures for cotton farming",
    "Post-harvest processing of cotton",
    "Applications of cotton",
    "Essential conditions for oil palm cultivation",
    "Procedures for oil palm farming",
    "Post-harvest processing of oil palm",
    "Applications of palm oil",
    "Cultivation techniques for tea",
    "Post-harvest processing of tea",
    "Applications of tea",
    "Growing conditions for sisal",
    "Cultivation techniques for sisal",
    "Post-harvest processing of sisal",
    "Applications of sisal",
    "Growing conditions for cloves",
    "Cultivation techniques for cloves",
    "Post-harvest processing and transport of cloves",
    "Applications of cloves",
    "Maize cultivation",
    "Wheat cultivation",
    "Rice cultivation",
    "Traits of nomadic pastoralism",
    "Pros of nomadic pastoralism",
    "Cons of nomadic pastoralism",
    "Semi-nomadic lifestyle",
    "Transhumance practices",
    "Features of commercial livestock farming",
    "Benefits of commercial livestock farming",
    "Drawbacks of commercial livestock farming",
    "Advantages of raising livestock",
    "Issues in livestock farming",
    "Solutions for challenges faced by pastoralists",
    "Livestock farming in Tanzania",
    "Issues in Tanzanian livestock farming",
    "Livestock farming practices in Australia",
    "Comparing livestock farming in Tanzania and Australia",
    "Contrasts in livestock farming between Tanzania and Australia",
    "Economic role of livestock farming",
    "Ambient temperature",
    "Atmospheric humidity",
    "Wind patterns",
    "Soil types",
    "Varieties of plant life",
    "Soil absorption",
    "Prerequisites for initiating a river basin project",
    "Advantages of river basin developments",
    "Obstacles in river basin projects",
    "The Rufiji Basin in Tanzania",
    "Examples of river basin projects",
    "Techniques for land recovery",
    "Land restoration in Tanzania",
    "Ocean water",
    "Groundwater sources",
    "Types of surface water",
    "Water wells",
    "Groundwater sources",
    "Natural springs",
    "Water wells",
    "Benefits from water resources",
    "Additional water resources",
    "Consequences of water resource extraction",
    "Untouched forests",
    "Tropical rainforest",
    "Properties of tropical rainforests",
    "Mangrove ecosystems",
    "Properties of mangrove forests",
    "Tropical monsoon forests",
    "Properties of tropical monsoon forests",
    "Deciduous woodlands",
    "Traits of deciduous forests",
    "Coniferous woodlands",
    "Traits of coniferous forests",
    "Cultivated forests",
    "Traits of cultivated forests",
    "Determinants of forest distribution",
    "Value of forest resources",
    "Ecological importance of forests",
    "Extraction industries",
    "Sector of extraction industries",
    "Coal extraction sector",
    "Metal extraction sector",
    "Non-metal mineral extraction sector",
    "Petroleum and natural gas industry",
    "Mineral occurrences",
    "Global mineral distribution",
    "Applications of various minerals",
    "Opencast mining",
    "Shaft mining",
    "Placer mining",
    "In-situ leach mining",
    "Ore crushing and milling",
    "Mineral separation",
    "Particle sizing",
    "Ore flotation process",
    "Petroleum extraction in the Middle East",
    "Economic impact of petroleum in Middle Eastern countries",
    "Problems in Middle Eastern oil industries",
    "Natural gas sector in Tanzania",
    "Benefits of natural gas in Tanzania",
    "Problems in Tanzanian natural gas sector",
    "Tanzania's strategies for natural gas challenges",
    "Positive effects of tourism",
    "Negative effects of tourism",
    "Mitigating negative tourism impacts",
    "Tourism growth factors in Switzerland",
    "Tourism's role in Switzerland's economy",
    "Problems in Switzerland's tourism sector",
    "Enhancing tourist access to attractions",
    "Conservation of biodiversity",
    "Enhancements to tourist sites",
    "Tourism development factors in Namibia",
    "Infrastructure improvements",
    "Robust tourism policies",
    "Namibia's Tourism Authority",
    "Regional tourism collaboration",
    "Tourism's contribution to Namibia",
    "Tourism industry challenges in Namibia",
    "Wildlife reserve management in Tanzania",
    "Tourism growth factors in Tanzania",
    "Tourism's significance in Tanzania",
    "Tourism industry problems in Tanzania",
    "Assembly industries",
    "Manufacturing sectors",
    "Manufacturing sectors in East Africa",
    "Location determinants for manufacturing",
    "Environmental pollution from industries",
    "Air contaminants",
    "Emission gases",
    "Solid industrial waste",
    "Industrial wastewater",
    "Industrial noise issues",
    "Growth factors for South Korean electronic industries",
    "Significance of the textile sector in Tanzania",
    "Textile industry challenges in Tanzania",
    "Non-renewable resources",
    "Significance of renewable energies in the USA",
    "Renewable energy challenges in the USA",
    "Solutions to renewable energy challenges in the USA"
    ]


def create_teacher_request(topic, question_type, system_prompt, client):
    prompt = f"Generate a request for a {question_type} question about {topic}."
    try:
        teacher_request = client.chat.completions.create(
            model="gpt-4-turbo",
            response_model=EvalQuery,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return teacher_request
    except Exception as e:
        print(f"Error generating query: {e}")
        return None
    
def generate_educator_queries(num_queries, topics, question_types, client):
    system_prompt = f"""
        You are a Tanzanian secondary school form 2 geography teacher that is texting a WhatsApp bot that is capable of generating questions for you to give to students as homework, classwork, or exams. 
        Each message you send comprises of a requests for a question on a topic that is provided to you. In the request you specify both the topic of interest and the type of question you want the bot
        to generate (the possible choices are true-false statements, long-answer questions like 'describe', 'explain', 'outline' etc..., and short-answer questions like 'define', 'state', 'list', etc...). 
        Be as varied as possible in how you formulate the request.

        Remember that examples of ways to describe long-answer questions are:
        - describing questions, explaining questions, outlining questions, etc...
        
        Short-answer questions:
        - defining questions, state questions, listing questions, etc...

        True-false questions:
        - statements, assertions, etc...
        
        The queries should reflect a typical way that Tanzanian secondary school teachers would text, including any grammar mistakes and their style of writing. Here is a group of text messages from Tanzanian secondary school teachers. 
        Try to emulate their grammar and writing style when you generate example queries:
        WRITING STYLE:
        {teacher_writing_style}
        
        
        EXAMPLE:
        USER: Generate a request for a short-answer question about wind and solar in the US.
        TEACHER: hi, I need a listing question about on renewable energy in USA

        EXAMPLE:
        USER: Generate a request for a true-false question about air contaminants.
        TEACHER: Give a right or wrong statement about air contaminants for my students form 2 exam

        EXAMPLE:
        USER: Generate a request for a long-answer question about industrial wastewater.
        TEACHER: good afternoon, i need exercise about industrial wastewater where students need to write a lot
    """

    queries: List[EvalQuery] = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(create_teacher_request, random.choice(topics), random.choice(question_types), system_prompt, client) for _ in range(num_queries)]
        for future in tqdm(as_completed(futures), total=num_queries, desc="Generating queries"):
            result = future.result()
            if result:
                queries.append(result)

    return queries

question_types = ["long-answer", "true-false", "short-answer"]

if __name__ == "__main__":
    content_path = os.path.join(DATA_DIR, "datasets", "test-prompts.json")

    # Patch the OpenAI client
    client = instructor.from_openai(OpenAI(api_key=OPENAI_API_KEY))

    num_to_generate = 10

    generated_queries = generate_educator_queries(num_to_generate, list_of_modified_topics, question_types, client)

    # Save to JSON
    save_objects_as_json(generated_queries, content_path, rewrite=False)