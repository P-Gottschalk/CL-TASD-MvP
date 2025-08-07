import json

sentiments = [
    "positive", 
    "negative", 
    "neutral",
]

rest_aspect_cate_list=[      
    "drinks prices",
    "restaurant prices",
    "restaurant miscellaneous",
    "ambience general",
    "restaurant general",
    "food general",
    "food quality",
    "location general",
    "drinks quality",
    "food prices",
    "food style_options",
    "drinks style_options",
    "service general"
]

force_words = sentiments + rest_aspect_cate_list + ["[SSEP]"]